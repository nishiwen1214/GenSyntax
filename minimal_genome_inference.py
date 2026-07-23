"""Context-aware iterative minimal-genome inference with GenSyntax."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_THRESHOLDS = (0.5, 0.4, 0.3, 0.2, 0.05)


def setup_environment(gpu_ids: str, tensor_parallel_size: int) -> None:
    """Set visible CUDA devices and validate tensor parallelism."""
    devices = [item.strip() for item in gpu_ids.split(",") if item.strip()]
    if not devices:
        raise ValueError("--gpu-ids must contain at least one CUDA device.")
    if len(devices) != tensor_parallel_size:
        raise ValueError(
            f"Number of GPUs ({len(devices)}) does not match "
            f"tensor_parallel_size ({tensor_parallel_size})."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)


def extract_product_names(
    record: dict[str, Any], record_index: int | None = None
) -> list[str]:
    """Extract product names while preserving the original product records."""
    products = record.get("Protein_products")
    location = f"Record {record_index}" if record_index is not None else "Record"
    if not isinstance(products, list) or not products:
        raise ValueError(f"{location} must contain a non-empty Protein_products list.")

    names: list[str] = []
    for product_index, product in enumerate(products):
        if isinstance(product, str):
            name = product
        elif isinstance(product, (list, tuple)) and len(product) >= 3:
            name = product[2]
        elif isinstance(product, dict):
            name = product.get("product") or product.get("Product")
        else:
            name = None
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"{location}, Protein_products[{product_index}] has no product name."
            )
        names.append(name.strip())
    return names


def load_genomes(path: Path) -> list[dict[str, Any]]:
    """Load and validate a JSON object or array of genome records."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    records = data if isinstance(data, list) else [data]
    if not records:
        raise ValueError(f"{path} contains no genome records.")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not a JSON object.")
        extract_product_names(record, index)
    return records


def organism_name(record: dict[str, Any]) -> str:
    """Return the organism label used in the model prompt."""
    for key in ("Source", "source", "Organism", "organism", "species"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "the supplied bacterial organism"


def build_instruction(
    organism: str,
    target_product: str,
    retained_product_names: Sequence[str],
    target_position: int,
) -> str:
    """Construct an essentiality prompt in the current genome context."""
    context = "".join(
        f"[{product}]"
        for position, product in enumerate(retained_product_names)
        if position != target_position
    )
    return (
        f"The following list presents the protein products encoded by "
        f"{organism} chromosome. Please predict whether the gene corresponding "
        f"to the protein product $${target_product}$$ is essential for this "
        f"organism? Answer strictly in the following format: non-essential or "
        f"essential\n{context}"
    )


def format_messages(instruction: str) -> list[dict[str, str]]:
    """Format one instruction for vLLM chat inference."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
    ]


def normalize_label_token(token: str) -> str | None:
    """Map a first generated token to one of the two constrained labels."""
    normalized = re.sub(r"^[^a-z]+", "", token.lower())
    if normalized.startswith("non"):
        return "non-essential"
    if normalized.startswith("essential"):
        return "essential"
    return None


def essential_probability_from_logprobs(token_logprobs: dict[Any, Any]) -> float:
    """Calculate P(essential) conditional on the two allowed first tokens."""
    values: dict[str, list[float]] = {"essential": [], "non-essential": []}
    for item in token_logprobs.values():
        token = getattr(item, "decoded_token", None)
        logprob = getattr(item, "logprob", None)
        if not isinstance(token, str) or not isinstance(logprob, (int, float)):
            continue
        label = normalize_label_token(token)
        if label is not None:
            values[label].append(float(logprob))

    missing = [label for label, candidates in values.items() if not candidates]
    if missing:
        raise RuntimeError(
            "Could not recover both answer-label probabilities from vLLM "
            f"logprobs; missing {', '.join(missing)}. Increase --top-logprobs "
            "or verify the model chat template."
        )

    essential_logprob = max(values["essential"])
    nonessential_logprob = max(values["non-essential"])
    offset = max(essential_logprob, nonessential_logprob)
    essential_mass = math.exp(essential_logprob - offset)
    nonessential_mass = math.exp(nonessential_logprob - offset)
    return essential_mass / (essential_mass + nonessential_mass)


def predict_essential_probability(
    llm: Any, instruction: str, sampling_params: Any
) -> float:
    """Run one model call and return normalized essentiality probability."""
    outputs = llm.chat([format_messages(instruction)], sampling_params)
    if len(outputs) != 1 or not outputs[0].outputs:
        raise RuntimeError("vLLM returned no output for an essentiality prompt.")
    positions = outputs[0].outputs[0].logprobs
    if not positions:
        raise RuntimeError("vLLM returned no generated-token log probabilities.")
    return essential_probability_from_logprobs(positions[0])


def iterative_reduction(
    llm: Any,
    record: dict[str, Any],
    threshold: float,
    rng: random.Random,
    sampling_params: Any,
) -> dict[str, Any]:
    """Run one randomized IRA replicate for one genome and threshold."""
    original_products = record["Protein_products"]
    product_names = extract_product_names(record)
    retained_indices = list(range(len(product_names)))
    evaluations = 0
    deletions: list[dict[str, Any]] = []

    while retained_indices:
        traversal = retained_indices.copy()
        rng.shuffle(traversal)
        deleted = False

        for original_index in traversal:
            current_position = retained_indices.index(original_index)
            retained_names = [product_names[index] for index in retained_indices]
            instruction = build_instruction(
                organism_name(record),
                product_names[original_index],
                retained_names,
                current_position,
            )
            probability = predict_essential_probability(
                llm, instruction, sampling_params
            )
            evaluations += 1

            # Revised definition: retain only when P(essential) > threshold.
            if probability <= threshold:
                retained_indices.remove(original_index)
                deletions.append(
                    {
                        "original_index": original_index,
                        "product": product_names[original_index],
                        "essential_probability": probability,
                        "retained_gene_count_after_deletion": len(retained_indices),
                    }
                )
                deleted = True
                break

        if not deleted:
            break

    result = copy.deepcopy(record)
    result["predicted_necessary_products"] = [
        original_products[index] for index in retained_indices
    ]
    result["minimal_genome_metadata"] = {
        "essentiality_confidence_threshold": threshold,
        "initial_gene_count": len(product_names),
        "retained_gene_count": len(retained_indices),
        "deleted_gene_count": len(product_names) - len(retained_indices),
        "model_evaluations": evaluations,
        "retained_original_indices": retained_indices,
        "deletion_trace": deletions,
    }
    return result


def parse_thresholds(values: Iterable[float]) -> list[float]:
    """Validate thresholds and remove duplicates while preserving order."""
    thresholds: list[float] = []
    for value in values:
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1: {threshold}")
        if threshold not in thresholds:
            thresholds.append(threshold)
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def threshold_slug(threshold: float) -> str:
    return format(threshold, "g").replace(".", "_")


def write_results(
    output_dir: Path,
    model_name: str,
    threshold: float,
    replicate: int,
    seed: int,
    records: list[dict[str, Any]],
) -> Path:
    """Write one independent threshold/replicate result atomically."""
    destination = (
        output_dir
        / model_name
        / f"threshold_{threshold_slug(threshold)}"
        / f"replicate_{replicate:02d}_seed_{seed}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    payload = {
        "model": model_name,
        "essentiality_confidence_threshold": threshold,
        "replicate": replicate,
        "seed": seed,
        "records": records,
    }
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive candidate minimal genomes with the GenSyntax IRA."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/minimal_genome")
    )
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Default: 0.5 0.4 0.3 0.2 0.05.",
    )
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replicates < 1:
        raise ValueError("--replicates must be at least 1.")
    thresholds = parse_thresholds(args.thresholds)
    records = load_genomes(args.input_json)
    setup_environment(args.gpu_ids, args.tensor_parallel_size)

    # Lazy import keeps schema/probability helpers testable without a GPU stack.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=args.top_logprobs,
    )
    model_name = Path(args.model_path.rstrip("/")).name

    for threshold in thresholds:
        for replicate_offset in range(args.replicates):
            replicate = replicate_offset + 1
            replicate_seed = args.seed + replicate_offset
            rng = random.Random(replicate_seed)
            reduced_records = [
                iterative_reduction(
                    llm, record, threshold, rng, sampling_params
                )
                for record in records
            ]
            path = write_results(
                args.output_dir,
                model_name,
                threshold,
                replicate,
                replicate_seed,
                reduced_records,
            )
            print(
                f"[INFO] threshold={threshold:g} replicate={replicate}/"
                f"{args.replicates} seed={replicate_seed} saved={path}"
            )


if __name__ == "__main__":
    main()
