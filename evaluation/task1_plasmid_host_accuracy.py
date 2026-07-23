#!/usr/bin/env python3
"""Evaluate Task 1 plasmid-host predictions across taxonomic ranks.

Class, order, family, genus, and species use every test record as the
denominator. Strain accuracy uses only records whose reference answer includes
strain information. Unresolved predictions remain in the relevant denominator
and are counted as incorrect.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


RANKS = ("class", "order", "family", "genus", "species", "strain")
DEFAULT_BOOTSTRAP_REPLICATES = 100
DEFAULT_CONFIDENCE_LEVEL = 0.90
DEFAULT_RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate taxonomic accuracy and percentile bootstrap confidence "
            "intervals for GenSyntax Task 1 plasmid-host predictions."
        )
    )
    parser.add_argument(
        "--references",
        type=Path,
        required=True,
        help="JSON array containing reference labels in an Output/output field.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Text file containing one prediction per line.",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        required=True,
        help="CSV file with genus,class,order,family columns.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/task1_accuracy.csv"),
        help="Destination for rank-level results.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/task1_accuracy.json"),
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
        "--quiet",
        action="store_true",
        help="Suppress taxonomy diagnostics and bootstrap progress.",
    )
    return parser.parse_args()


def normalize_taxon(value: Any) -> str:
    """Normalize a non-strain taxonomic name to lowercase ASCII letters."""
    if value is None:
        return ""
    return re.sub(r"[^A-Za-z]", "", str(value).strip()).lower()


def normalize_strain(value: Any) -> str:
    """Normalize a strain name while retaining letters and digits."""
    if value is None:
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", str(value).strip()).lower()


def parse_label(value: Any) -> tuple[str, str, str]:
    """Return genus, species epithet, and strain text from a model label.

    Both ``Genus species strain`` and ``[Genus, species, strain]`` are
    supported. If a response contains a bracketed answer, the bracketed text is
    preferred over surrounding explanation.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        return "", "", ""

    text = re.sub(r"^\s*(?:final\s+)?answer\s*:\s*", "", text, flags=re.I)

    # Treat brackets as answer delimiters only when they enclose the complete
    # response. Internal brackets may be part of a valid strain or serovar name
    # (for example, Salmonella serovar 1,4,[5],12:i:-).
    bracketed = re.fullmatch(r"\s*\[([^\[\]]+)\]\s*", text)
    is_bracketed_answer = bracketed is not None
    if bracketed:
        text = bracketed.group(1).strip()

    if is_bracketed_answer and "," in text:
        parts = [part.strip() for part in text.split(",")]
        genus = parts[0] if len(parts) >= 1 else ""
        species = parts[1] if len(parts) >= 2 else ""
        strain = " ".join(parts[2:]).strip() if len(parts) >= 3 else ""
        return genus, species, strain

    parts = text.split()
    if len(parts) >= 3 and parts[0].lower() == "candidatus":
        # "Candidatus" is a provisional-status qualifier rather than the
        # genus. For "Candidatus Accumulibacter phosphatis", use
        # Accumulibacter as the genus and phosphatis as the species epithet.
        genus = parts[1]
        species = parts[2]
        strain = " ".join(parts[3:]) if len(parts) >= 4 else ""
        return genus, species, strain
    genus = parts[0] if len(parts) >= 1 else ""
    species = parts[1] if len(parts) >= 2 else ""
    strain = " ".join(parts[2:]) if len(parts) >= 3 else ""
    return genus, species, strain


def load_references(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Reference JSON must contain a top-level array: {path}")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Reference record {index} is not a JSON object")
    return records


def load_predictions(path: Path, expected_count: int) -> list[str]:
    """Load predictions without dropping internal blank lines."""
    with path.open("r", encoding="utf-8") as handle:
        predictions = [line.rstrip("\r\n") for line in handle]

    # Text files often contain an additional final blank line. Remove only
    # blank records beyond the expected number so internal empty predictions
    # remain aligned with their reference records.
    while len(predictions) > expected_count and not predictions[-1].strip():
        predictions.pop()

    if len(predictions) != expected_count:
        raise ValueError(
            "Reference and prediction counts differ: "
            f"{expected_count} references versus {len(predictions)} predictions"
        )
    return predictions


def load_taxonomy(path: Path) -> dict[str, dict[str, str]]:
    taxonomy: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Taxonomy CSV has no header: {path}")
        canonical_headers = {name.strip().lower(): name for name in reader.fieldnames}
        required = {"genus", "class", "order", "family"}
        missing = sorted(required - canonical_headers.keys())
        if missing:
            raise ValueError(
                f"Taxonomy CSV is missing required columns {missing}: {path}"
            )

        for row in reader:
            genus = normalize_taxon(row[canonical_headers["genus"]])
            if not genus:
                continue
            taxonomy[genus] = {
                rank: normalize_taxon(row[canonical_headers[rank]])
                for rank in ("class", "order", "family")
            }

    if not taxonomy:
        raise ValueError(f"Taxonomy CSV contains no usable genus records: {path}")
    return taxonomy


def empty_statistics() -> dict[str, dict[str, int]]:
    return {
        rank: {"correct": 0, "total": 0, "unresolved": 0}
        for rank in RANKS
    }


def evaluate_predictions(
    references: list[dict[str, Any]],
    predictions: list[str],
    taxonomy: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, int]], Counter[str], Counter[str]]:
    if len(references) != len(predictions):
        raise ValueError("Reference and prediction counts must be identical")

    statistics = empty_statistics()
    missing_reference_genera: Counter[str] = Counter()
    missing_prediction_genera: Counter[str] = Counter()

    for record, prediction in zip(references, predictions):
        for rank in ("class", "order", "family", "genus", "species"):
            statistics[rank]["total"] += 1

        reference_label = record.get("Output", record.get("output", ""))
        true_genus_raw, true_species_raw, true_strain_raw = parse_label(reference_label)
        pred_genus_raw, pred_species_raw, pred_strain_raw = parse_label(prediction)

        true_genus = normalize_taxon(true_genus_raw)
        pred_genus = normalize_taxon(pred_genus_raw)

        if not true_genus or not pred_genus:
            statistics["genus"]["unresolved"] += 1
        elif true_genus == pred_genus:
            statistics["genus"]["correct"] += 1

        true_taxonomy = taxonomy.get(true_genus)
        pred_taxonomy = taxonomy.get(pred_genus)
        if true_genus and true_taxonomy is None:
            missing_reference_genera[true_genus] += 1
        if pred_genus and pred_taxonomy is None:
            missing_prediction_genera[pred_genus] += 1

        for rank in ("class", "order", "family"):
            true_value = true_taxonomy.get(rank, "") if true_taxonomy else ""
            pred_value = pred_taxonomy.get(rank, "") if pred_taxonomy else ""
            if not true_value or not pred_value:
                statistics[rank]["unresolved"] += 1
            elif true_value == pred_value:
                statistics[rank]["correct"] += 1

        true_species = normalize_taxon(f"{true_genus_raw} {true_species_raw}")
        pred_species = normalize_taxon(f"{pred_genus_raw} {pred_species_raw}")
        if not true_genus or not true_species_raw or not pred_genus or not pred_species_raw:
            statistics["species"]["unresolved"] += 1
        elif true_species == pred_species:
            statistics["species"]["correct"] += 1

        true_strain = normalize_strain(true_strain_raw)
        if true_strain:
            statistics["strain"]["total"] += 1
            pred_strain = normalize_strain(pred_strain_raw)
            if not pred_strain:
                statistics["strain"]["unresolved"] += 1
            # Preserve the original evaluation rule: a prediction may include
            # explanatory suffix text after the correct strain designation.
            elif true_strain in pred_strain:
                statistics["strain"]["correct"] += 1

    return statistics, missing_reference_genera, missing_prediction_genera


def bootstrap_accuracy(
    references: list[dict[str, Any]],
    predictions: list[str],
    taxonomy: dict[str, dict[str, str]],
    replicates: int,
    confidence_level: float,
    seed: int,
    show_progress: bool,
) -> dict[str, dict[str, float | int]]:
    if not references:
        raise ValueError("The reference dataset is empty")
    if replicates <= 0:
        raise ValueError("--bootstrap-replicates must be greater than zero")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("--confidence-level must be between zero and one")

    rng = np.random.default_rng(seed)
    sample_count = len(references)
    scores = {rank: [] for rank in RANKS}
    totals = {rank: [] for rank in RANKS}
    unresolved_counts = {rank: [] for rank in RANKS}

    for replicate_index in range(replicates):
        indices = rng.integers(0, sample_count, size=sample_count)
        sampled_references = [references[index] for index in indices]
        sampled_predictions = [predictions[index] for index in indices]
        sampled_stats, _, _ = evaluate_predictions(
            sampled_references, sampled_predictions, taxonomy
        )

        for rank in RANKS:
            correct = sampled_stats[rank]["correct"]
            total = sampled_stats[rank]["total"]
            totals[rank].append(total)
            unresolved_counts[rank].append(sampled_stats[rank]["unresolved"])
            if total > 0:
                scores[rank].append(correct / total * 100.0)

        if show_progress:
            print(
                f"\rBootstrap progress: {replicate_index + 1}/{replicates}",
                end="",
                flush=True,
            )
    if show_progress:
        print()

    alpha = 1.0 - confidence_level
    lower_percentile = alpha / 2.0 * 100.0
    upper_percentile = (1.0 - alpha / 2.0) * 100.0
    results: dict[str, dict[str, float | int]] = {}

    for rank in RANKS:
        rank_scores = np.asarray(scores[rank], dtype=float)
        if rank_scores.size == 0:
            results[rank] = {
                "mean_percent": float("nan"),
                "standard_deviation_percent": float("nan"),
                "ci_lower_percent": float("nan"),
                "ci_upper_percent": float("nan"),
                "mean_total": float("nan"),
                "mean_unresolved": float("nan"),
                "valid_replicates": 0,
            }
            continue

        results[rank] = {
            "mean_percent": float(np.mean(rank_scores)),
            "standard_deviation_percent": float(
                np.std(rank_scores, ddof=1) if rank_scores.size > 1 else 0.0
            ),
            "ci_lower_percent": float(
                np.percentile(rank_scores, lower_percentile)
            ),
            "ci_upper_percent": float(
                np.percentile(rank_scores, upper_percentile)
            ),
            "mean_total": float(np.mean(totals[rank])),
            "mean_unresolved": float(np.mean(unresolved_counts[rank])),
            "valid_replicates": int(rank_scores.size),
        }
    return results


def build_results(
    statistics: dict[str, dict[str, int]],
    bootstrap: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    for rank in RANKS:
        correct = statistics[rank]["correct"]
        total = statistics[rank]["total"]
        results[rank] = {
            "correct": correct,
            "total": total,
            "unresolved": statistics[rank]["unresolved"],
            "original_accuracy_percent": correct / total * 100.0 if total else float("nan"),
            **bootstrap[rank],
        }
    return results


def write_csv(path: Path, results: dict[str, dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank", *next(iter(results.values())).keys()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank in RANKS:
            writer.writerow({"rank": rank, **results[rank]})


def write_json(
    path: Path,
    results: dict[str, dict[str, float | int]],
    args: argparse.Namespace,
    sample_count: int,
) -> None:
    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "Task 1: plasmid host taxonomic classification",
        "settings": {
            "sample_count": sample_count,
            "bootstrap_replicates": args.bootstrap_replicates,
            "confidence_level": args.confidence_level,
            "confidence_interval_method": "two-sided percentile bootstrap",
            "random_seed": args.seed,
            "references": str(args.references),
            "predictions": str(args.predictions),
            "taxonomy": str(args.taxonomy),
        },
        "results": json_safe(results),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def print_diagnostics(
    taxonomy_count: int,
    statistics: dict[str, dict[str, int]],
    missing_reference_genera: Counter[str],
    missing_prediction_genera: Counter[str],
) -> None:
    print(f"Loaded taxonomy records: {taxonomy_count}")
    print(f"Reference genera missing from taxonomy: {len(missing_reference_genera)}")
    print(f"Predicted genera missing from taxonomy: {len(missing_prediction_genera)}")
    print(f"Reference records with strain labels: {statistics['strain']['total']}")
    print(f"Unresolved strain predictions: {statistics['strain']['unresolved']}")

    for label, counter in (
        ("Most frequent missing reference genera", missing_reference_genera),
        ("Most frequent missing predicted genera", missing_prediction_genera),
    ):
        if counter:
            print(f"\n{label}:")
            for genus, count in counter.most_common(20):
                print(f"  {genus}: {count}")


def print_results(
    results: dict[str, dict[str, float | int]], confidence_level: float
) -> None:
    confidence_percent = int(round(confidence_level * 100))
    print("\nTask 1 taxonomic accuracy")
    print("-" * 88)
    print(
        f"{'Rank':<10} {'Original':>12} {'Correct/Total':>16} "
        f"{'Unresolved':>12} {f'Bootstrap mean ({confidence_percent}% CI)':>32}"
    )
    for rank in RANKS:
        result = results[rank]
        interval = (
            f"{result['mean_percent']:.2f}% "
            f"[{result['ci_lower_percent']:.2f}, {result['ci_upper_percent']:.2f}]"
        )
        print(
            f"{rank.title():<10} "
            f"{result['original_accuracy_percent']:>11.2f}% "
            f"{result['correct']:>7}/{result['total']:<8} "
            f"{result['unresolved']:>12} {interval:>32}"
        )


def main() -> None:
    args = parse_args()
    references = load_references(args.references)
    predictions = load_predictions(args.predictions, len(references))
    taxonomy = load_taxonomy(args.taxonomy)

    statistics, missing_reference_genera, missing_prediction_genera = (
        evaluate_predictions(references, predictions, taxonomy)
    )
    if not args.quiet:
        print(f"Reference records: {len(references)}")
        print(f"Prediction records: {len(predictions)}")
        print_diagnostics(
            len(taxonomy),
            statistics,
            missing_reference_genera,
            missing_prediction_genera,
        )

    bootstrap = bootstrap_accuracy(
        references=references,
        predictions=predictions,
        taxonomy=taxonomy,
        replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
        show_progress=not args.quiet,
    )
    results = build_results(statistics, bootstrap)
    print_results(results, args.confidence_level)
    write_csv(args.output_csv, results)
    write_json(args.output_json, results, args, len(references))
    print(f"\nCSV results: {args.output_csv}")
    print(f"JSON results: {args.output_json}")


if __name__ == "__main__":
    main()
