#!/usr/bin/env python3
"""Evaluate Task 2 multiple-choice gene-product predictions.

The evaluator supports both four-option (A-D) and eight-option (A-H) test
sets. Unresolved or out-of-range predictions remain in the denominator and are
counted as incorrect. Confidence intervals are calculated by sample-level
percentile bootstrap resampling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_BOOTSTRAP_REPLICATES = 100
DEFAULT_CONFIDENCE_LEVEL = 0.90
DEFAULT_RANDOM_SEED = 42
OPTION_LABELS = tuple("ABCDEFGH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate accuracy and percentile bootstrap confidence intervals "
            "for four-option or eight-option GenSyntax Task 2 predictions."
        )
    )
    parser.add_argument(
        "--references",
        type=Path,
        required=True,
        help="JSON array containing reference labels and multiple-choice inputs.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Text file containing one model prediction per line.",
    )
    parser.add_argument(
        "--num-options",
        choices=("auto", "4", "8"),
        default="auto",
        help=(
            "Number of options per sample. 'auto' detects the count from each "
            "record and requires all records to agree (default: auto)."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/task2_accuracy.csv"),
        help="Destination for the summary CSV file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/task2_accuracy.json"),
        help="Destination for results and evaluation settings.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
        help="Number of sample-level bootstrap replicates (default: 100).",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL,
        help="Two-sided percentile confidence level (default: 0.90).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Bootstrap random seed (default: 42).",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=0,
        help="Print parsing details for the first N records (default: 0).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress bootstrap progress output.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate numerical command-line settings before loading large files."""
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be greater than zero")
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("--confidence-level must be between zero and one")
    if args.debug_samples < 0:
        raise ValueError("--debug-samples must be zero or greater")


def extract_answer_candidates(text: Any) -> list[str]:
    """Extract answer labels using fixed, ordered parsing rules.

    Parsing priority follows the manuscript description: bracketed labels,
    numbered labels, explicit ``Answer: X`` text, and finally a standalone
    option letter.
    """
    value = "" if text is None else str(text).strip()
    if not value:
        return []

    patterns = (
        r"\[\s*([A-H])\s*\]",
        r"(?:^|\n|\s)\d+\s*[.):-]?\s*([A-H])(?!\w)",
        r"answer[^\w]{0,3}([A-H])(?!\w)",
        r"(?<!\w)([A-H])(?!\w)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, value, flags=re.IGNORECASE)
        if matches:
            return [match.upper() for match in matches]
    return []


def extract_single_answer(text: Any, allowed_labels: set[str]) -> str | None:
    """Return the first parsed answer if it is valid for the option count."""
    candidates = extract_answer_candidates(text)
    if not candidates:
        return None
    answer = candidates[0]
    return answer if answer in allowed_labels else None


def get_reference_text(record: dict[str, Any]) -> Any:
    for key in ("label", "Label", "Output", "output", "answer", "Answer"):
        if key in record:
            return record[key]
    raise ValueError(
        "Reference record has no label field. Expected one of: "
        "label, Label, Output, output, answer, Answer"
    )


def get_prompt_text(record: dict[str, Any]) -> str:
    for key in ("Input", "input", "instruction", "prompt", "question"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def detect_option_count(record: dict[str, Any]) -> int | None:
    """Detect whether a record contains four or eight answer options."""
    for key in ("num_options", "option_count", "n_options"):
        value = record.get(key)
        if str(value) in {"4", "8"}:
            return int(value)

    options = record.get("options", record.get("Options"))
    if isinstance(options, (list, tuple, dict)) and len(options) in {4, 8}:
        return len(options)

    prompt = get_prompt_text(record)
    if not prompt:
        return None

    labels: set[str] = set()
    marker_patterns = (
        r"\[\s*([A-H])\s*\]",
        r"(?:^|\n)\s*([A-H])\s*[.):-]",
        r"(?:^|\n)\s*\(([A-H])\)",
        r"(?:^|[.;\n])\s*([A-H])\s*:",
    )
    for pattern in marker_patterns:
        labels.update(
            match.upper()
            for match in re.findall(pattern, prompt, flags=re.IGNORECASE)
        )

    if set("ABCDEFGH").issubset(labels):
        return 8
    if set("ABCD").issubset(labels) and not labels.intersection(set("EFGH")):
        return 4
    return None


def resolve_option_count(records: Sequence[dict[str, Any]], mode: str) -> int:
    if mode in {"4", "8"}:
        return int(mode)

    detected = [detect_option_count(record) for record in records]
    missing_indices = [index + 1 for index, value in enumerate(detected) if value is None]
    detected_counts = {value for value in detected if value is not None}

    if missing_indices:
        preview = ", ".join(map(str, missing_indices[:10]))
        raise ValueError(
            "Could not automatically detect the option count for reference "
            f"records: {preview}. Pass --num-options 4 or --num-options 8."
        )
    if len(detected_counts) != 1:
        raise ValueError(
            "Auto-detection found mixed option counts in one evaluation file: "
            f"{sorted(detected_counts)}. Evaluate four-option and eight-option "
            "files separately."
        )
    option_count = detected_counts.pop()
    if option_count not in {4, 8}:
        raise ValueError(f"Unsupported detected option count: {option_count}")
    return option_count


def load_references(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Reference JSON must contain a top-level array: {path}")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Reference record {index + 1} is not a JSON object")
    return records


def load_predictions(path: Path, expected_count: int) -> list[str]:
    """Load predictions while preserving internal blank lines."""
    with path.open("r", encoding="utf-8") as handle:
        predictions = [line.rstrip("\r\n") for line in handle]
    while len(predictions) > expected_count and not predictions[-1].strip():
        predictions.pop()
    if len(predictions) != expected_count:
        raise ValueError(
            "Reference and prediction counts differ: "
            f"{expected_count} references versus {len(predictions)} predictions"
        )
    return predictions


def parse_reference_answers(
    records: Sequence[dict[str, Any]], allowed_labels: set[str]
) -> list[str]:
    answers: list[str] = []
    for index, record in enumerate(records):
        reference_text = get_reference_text(record)
        candidates = extract_answer_candidates(reference_text)
        valid_candidates = [value for value in candidates if value in allowed_labels]
        if len(valid_candidates) != 1:
            raise ValueError(
                f"Reference record {index + 1} must contain exactly one valid "
                f"answer in {sorted(allowed_labels)}; parsed {valid_candidates} "
                f"from {reference_text!r}"
            )
        answers.append(valid_candidates[0])
    return answers


def evaluate_indices(
    reference_answers: Sequence[str],
    prediction_answers: Sequence[str | None],
    indices: Iterable[int],
) -> dict[str, float | int]:
    total = 0
    correct = 0
    unresolved = 0
    for index in indices:
        total += 1
        prediction = prediction_answers[index]
        if prediction is None:
            unresolved += 1
        elif prediction == reference_answers[index]:
            correct += 1
    return {
        "total": total,
        "correct": correct,
        "unresolved": unresolved,
        "accuracy_percent": correct / total * 100.0 if total else float("nan"),
    }


def bootstrap_accuracy(
    reference_answers: Sequence[str],
    prediction_answers: Sequence[str | None],
    replicates: int,
    confidence_level: float,
    seed: int,
    show_progress: bool,
) -> dict[str, float | int]:
    if not reference_answers:
        raise ValueError("The reference dataset is empty")
    if len(reference_answers) != len(prediction_answers):
        raise ValueError("Reference and prediction counts must be identical")
    if replicates <= 0:
        raise ValueError("--bootstrap-replicates must be greater than zero")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("--confidence-level must be between zero and one")

    sample_count = len(reference_answers)
    rng = np.random.default_rng(seed)
    accuracies: list[float] = []
    unresolved_counts: list[int] = []

    for replicate_index in range(replicates):
        indices = rng.integers(0, sample_count, size=sample_count)
        result = evaluate_indices(reference_answers, prediction_answers, indices)
        accuracies.append(float(result["accuracy_percent"]))
        unresolved_counts.append(int(result["unresolved"]))
        if show_progress:
            print(
                f"\rBootstrap progress: {replicate_index + 1}/{replicates}",
                end="",
                flush=True,
            )
    if show_progress:
        print()

    values = np.asarray(accuracies, dtype=float)
    alpha = 1.0 - confidence_level
    return {
        "mean_percent": float(np.mean(values)),
        "standard_deviation_percent": float(
            np.std(values, ddof=1) if values.size > 1 else 0.0
        ),
        "ci_lower_percent": float(np.percentile(values, alpha / 2.0 * 100.0)),
        "ci_upper_percent": float(
            np.percentile(values, (1.0 - alpha / 2.0) * 100.0)
        ),
        "mean_unresolved": float(np.mean(unresolved_counts)),
        "valid_replicates": int(values.size),
    }


def write_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)


def build_result(
    option_count: int,
    sample_count: int,
    original: dict[str, float | int],
    bootstrap: dict[str, float | int],
) -> dict[str, Any]:
    """Combine original-sample and bootstrap statistics for serialization."""
    return {
        "option_count": option_count,
        "sample_count": sample_count,
        "correct": original["correct"],
        "total": original["total"],
        "unresolved": original["unresolved"],
        "original_accuracy_percent": original["accuracy_percent"],
        **bootstrap,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def print_debug_records(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[str],
    reference_answers: Sequence[str],
    prediction_answers: Sequence[str | None],
    count: int,
) -> None:
    for index in range(min(count, len(records))):
        print("=" * 72)
        print(f"Record: {index + 1}")
        print(f"Reference text: {get_reference_text(records[index])}")
        print(f"Prediction text: {predictions[index]}")
        print(f"Parsed reference: {reference_answers[index]}")
        print(f"Parsed prediction: {prediction_answers[index]}")


def print_results(result: dict[str, Any], confidence_level: float) -> None:
    """Print a compact result table consistent with the Task 1 evaluator."""
    confidence_percent = int(round(confidence_level * 100))
    interval = (
        f"{result['mean_percent']:.2f}% "
        f"[{result['ci_lower_percent']:.2f}, "
        f"{result['ci_upper_percent']:.2f}]"
    )
    print("\nTask 2 gene-product disambiguation accuracy")
    print("-" * 94)
    print(
        f"{'Options':<10} {'Original':>12} {'Correct/Total':>16} "
        f"{'Unresolved':>12} {f'Bootstrap mean ({confidence_percent}% CI)':>36}"
    )
    print(
        f"{result['option_count']:<10} "
        f"{result['original_accuracy_percent']:>11.2f}% "
        f"{result['correct']:>7}/{result['total']:<8} "
        f"{result['unresolved']:>12} {interval:>36}"
    )


def main() -> None:
    args = parse_args()
    validate_arguments(args)
    records = load_references(args.references)
    predictions = load_predictions(args.predictions, len(records))
    option_count = resolve_option_count(records, args.num_options)
    allowed_labels = set(OPTION_LABELS[:option_count])

    reference_answers = parse_reference_answers(records, allowed_labels)
    prediction_answers = [
        extract_single_answer(prediction, allowed_labels)
        for prediction in predictions
    ]
    original = evaluate_indices(
        reference_answers, prediction_answers, range(len(records))
    )
    bootstrap = bootstrap_accuracy(
        reference_answers=reference_answers,
        prediction_answers=prediction_answers,
        replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
        show_progress=not args.quiet,
    )

    result = build_result(option_count, len(records), original, bootstrap)
    prediction_distribution = Counter(
        answer if answer is not None else "unresolved"
        for answer in prediction_answers
    )
    payload = {
        "task": "Task 2: multiple-choice gene-product disambiguation",
        "settings": {
            "option_count": option_count,
            "bootstrap_replicates": args.bootstrap_replicates,
            "confidence_level": args.confidence_level,
            "confidence_interval_method": "two-sided percentile bootstrap",
            "random_seed": args.seed,
            "references": str(args.references),
            "predictions": str(args.predictions),
        },
        "prediction_distribution": dict(sorted(prediction_distribution.items())),
        "results": result,
    }

    if args.debug_samples > 0:
        print_debug_records(
            records,
            predictions,
            reference_answers,
            prediction_answers,
            args.debug_samples,
        )

    print(f"Reference records: {len(records)}")
    print(f"Prediction records: {len(predictions)}")
    print(f"Detected option count: {option_count}")
    print_results(result, args.confidence_level)

    write_csv(args.output_csv, result)
    write_json(args.output_json, payload)
    print(f"CSV results: {args.output_csv}")
    print(f"JSON results: {args.output_json}")


if __name__ == "__main__":
    main()
