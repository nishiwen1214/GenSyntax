#!/usr/bin/env python3
"""Evaluate Task 4 binary gene-essentiality predictions.

The evaluator reports accuracy, class-specific precision/recall/F1, and macro
precision/recall/F1. Unresolved predictions remain in the accuracy denominator
and count as false negatives for the corresponding reference class.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ESSENTIAL = "essential"
NON_ESSENTIAL = "non-essential"
CLASSES = (ESSENTIAL, NON_ESSENTIAL)
DEFAULT_BOOTSTRAP_REPLICATES = 100
DEFAULT_CONFIDENCE_LEVEL = 0.90
DEFAULT_RANDOM_SEED = 42
METRIC_NAMES = (
    "accuracy",
    "essential_precision",
    "essential_recall",
    "essential_f1",
    "nonessential_precision",
    "nonessential_recall",
    "nonessential_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate classification metrics and percentile bootstrap "
            "confidence intervals for GenSyntax Task 4 predictions."
        )
    )
    parser.add_argument(
        "--references",
        type=Path,
        required=True,
        help="JSON or JSONL file containing reference essentiality labels.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Text file containing one prediction per line.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/task4_metrics.csv"),
        help="Destination for the metric-level CSV file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/task4_metrics.json"),
        help="Destination for results and evaluation settings.",
    )
    parser.add_argument(
        "--errors-csv",
        type=Path,
        default=None,
        help="Optional destination for incorrect and unresolved records.",
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
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be greater than zero")
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("--confidence-level must be between zero and one")
    if args.debug_samples < 0:
        raise ValueError("--debug-samples must be zero or greater")


def parse_label(text: Any) -> str | None:
    """Parse essential or non-essential from a label or model response.

    Non-essential patterns are evaluated first because the string
    ``non-essential`` contains the substring ``essential``.
    """
    value = "" if text is None else str(text).strip().lower()
    if not value:
        return None

    value = value.replace("–", "-").replace("—", "-").replace("_", "-")
    nonessential_patterns = (
        r"\bnon[\s-]*essential\b",
        r"\bnot\s+essential\b",
        r"\binessential\b",
    )
    if any(re.search(pattern, value) for pattern in nonessential_patterns):
        return NON_ESSENTIAL
    if re.search(r"\bessential\b", value):
        return ESSENTIAL
    return None


def get_reference_text(record: dict[str, Any]) -> Any:
    for key in ("label", "Label", "Output", "output", "answer", "Answer"):
        if key in record:
            return record[key]
    raise ValueError(
        "Reference record has no label field. Expected one of: "
        "label, Label, Output, output, answer, Answer"
    )


def load_references(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSONL at line {line_number}: {error}"
                    ) from error
                records.append(record)
    else:
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError(f"Reference data must contain a top-level array: {path}")
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


def parse_reference_labels(records: Sequence[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for index, record in enumerate(records):
        raw_label = get_reference_text(record)
        label = parse_label(raw_label)
        if label is None:
            raise ValueError(
                f"Reference record {index + 1} has an invalid label: {raw_label!r}"
            )
        labels.append(label)
    return labels


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def calculate_metrics(
    reference_labels: Sequence[str], prediction_labels: Sequence[str | None]
) -> dict[str, float | int | dict[str, dict[str, int]]]:
    """Calculate accuracy, per-class metrics, macro metrics, and counts."""
    if len(reference_labels) != len(prediction_labels):
        raise ValueError("Reference and prediction counts must be identical")
    if not reference_labels:
        raise ValueError("The evaluation dataset is empty")
    invalid_references = [label for label in reference_labels if label not in CLASSES]
    if invalid_references:
        raise ValueError(f"Invalid reference labels: {sorted(set(invalid_references))}")

    total = len(reference_labels)
    correct = sum(
        truth == prediction
        for truth, prediction in zip(reference_labels, prediction_labels)
    )
    unresolved = sum(prediction is None for prediction in prediction_labels)
    class_metrics: dict[str, dict[str, float | int]] = {}

    for target in CLASSES:
        true_positive = sum(
            truth == target and prediction == target
            for truth, prediction in zip(reference_labels, prediction_labels)
        )
        false_positive = sum(
            truth != target and prediction == target
            for truth, prediction in zip(reference_labels, prediction_labels)
        )
        false_negative = sum(
            truth == target and prediction != target
            for truth, prediction in zip(reference_labels, prediction_labels)
        )
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)
        class_metrics[target] = {
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "accuracy": safe_divide(correct, total),
        "essential_precision": class_metrics[ESSENTIAL]["precision"],
        "essential_recall": class_metrics[ESSENTIAL]["recall"],
        "essential_f1": class_metrics[ESSENTIAL]["f1"],
        "nonessential_precision": class_metrics[NON_ESSENTIAL]["precision"],
        "nonessential_recall": class_metrics[NON_ESSENTIAL]["recall"],
        "nonessential_f1": class_metrics[NON_ESSENTIAL]["f1"],
        "macro_precision": float(
            np.mean([class_metrics[label]["precision"] for label in CLASSES])
        ),
        "macro_recall": float(
            np.mean([class_metrics[label]["recall"] for label in CLASSES])
        ),
        "macro_f1": float(np.mean([class_metrics[label]["f1"] for label in CLASSES])),
        "total": total,
        "correct": correct,
        "unresolved": unresolved,
        "confusion_counts": {
            label: {
                "tp": int(class_metrics[label]["tp"]),
                "fp": int(class_metrics[label]["fp"]),
                "fn": int(class_metrics[label]["fn"]),
            }
            for label in CLASSES
        },
    }


def bootstrap_metrics(
    reference_labels: Sequence[str],
    prediction_labels: Sequence[str | None],
    replicates: int,
    confidence_level: float,
    seed: int,
    show_progress: bool,
) -> dict[str, dict[str, float | int]]:
    if len(reference_labels) != len(prediction_labels):
        raise ValueError("Reference and prediction counts must be identical")
    if not reference_labels:
        raise ValueError("The evaluation dataset is empty")
    if replicates <= 0:
        raise ValueError("--bootstrap-replicates must be greater than zero")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("--confidence-level must be between zero and one")

    references = np.asarray(reference_labels, dtype=object)
    predictions = np.asarray(prediction_labels, dtype=object)
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in METRIC_NAMES}

    for replicate_index in range(replicates):
        indices = rng.integers(0, len(references), size=len(references))
        metrics = calculate_metrics(
            references[indices].tolist(), predictions[indices].tolist()
        )
        for metric in METRIC_NAMES:
            values[metric].append(float(metrics[metric]))
        if show_progress:
            print(
                f"\rBootstrap progress: {replicate_index + 1}/{replicates}",
                end="",
                flush=True,
            )
    if show_progress:
        print()

    alpha = 1.0 - confidence_level
    results: dict[str, dict[str, float | int]] = {}
    for metric in METRIC_NAMES:
        metric_values = np.asarray(values[metric], dtype=float) * 100.0
        results[metric] = {
            "mean_percent": float(np.mean(metric_values)),
            "standard_deviation_percent": float(
                np.std(metric_values, ddof=1) if metric_values.size > 1 else 0.0
            ),
            "ci_lower_percent": float(
                np.percentile(metric_values, alpha / 2.0 * 100.0)
            ),
            "ci_upper_percent": float(
                np.percentile(metric_values, (1.0 - alpha / 2.0) * 100.0)
            ),
            "valid_replicates": int(metric_values.size),
        }
    return results


def build_metric_rows(
    original: dict[str, Any], bootstrap: dict[str, dict[str, float | int]]
) -> list[dict[str, Any]]:
    return [
        {
            "metric": metric,
            "original_percent": float(original[metric]) * 100.0,
            **bootstrap[metric],
        }
        for metric in METRIC_NAMES
    ]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_errors_csv(
    path: Path,
    reference_labels: Sequence[str],
    prediction_texts: Sequence[str],
    prediction_labels: Sequence[str | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "record_index",
            "reference_label",
            "prediction_label",
            "prediction_text",
            "resolved",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (truth, text, prediction) in enumerate(
            zip(reference_labels, prediction_texts, prediction_labels), start=1
        ):
            if truth != prediction:
                writer.writerow(
                    {
                        "record_index": index,
                        "reference_label": truth,
                        "prediction_label": prediction or "",
                        "prediction_text": text,
                        "resolved": prediction is not None,
                    }
                )


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


def print_results(rows: Sequence[dict[str, Any]], confidence_level: float) -> None:
    confidence_percent = int(round(confidence_level * 100))
    print("\nTask 4 gene-essentiality metrics")
    print("-" * 94)
    print(
        f"{'Metric':<24} {'Original':>12} "
        f"{f'Bootstrap mean ({confidence_percent}% CI)':>42}"
    )
    for row in rows:
        interval = (
            f"{row['mean_percent']:.2f}% "
            f"[{row['ci_lower_percent']:.2f}, {row['ci_upper_percent']:.2f}]"
        )
        print(
            f"{row['metric']:<24} {row['original_percent']:>11.2f}% "
            f"{interval:>42}"
        )


def main() -> None:
    args = parse_args()
    validate_arguments(args)
    records = load_references(args.references)
    prediction_texts = load_predictions(args.predictions, len(records))
    reference_labels = parse_reference_labels(records)
    prediction_labels = [parse_label(text) for text in prediction_texts]

    original = calculate_metrics(reference_labels, prediction_labels)
    bootstrap = bootstrap_metrics(
        reference_labels=reference_labels,
        prediction_labels=prediction_labels,
        replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
        show_progress=not args.quiet,
    )
    rows = build_metric_rows(original, bootstrap)
    payload = {
        "task": "Task 4: binary gene-essentiality prediction",
        "settings": {
            "positive_class_for_essential_metrics": ESSENTIAL,
            "unresolved_prediction_policy": (
                "Counted as incorrect for accuracy and as a false negative for "
                "the corresponding reference class"
            ),
            "bootstrap_replicates": args.bootstrap_replicates,
            "confidence_level": args.confidence_level,
            "confidence_interval_method": "two-sided percentile bootstrap",
            "random_seed": args.seed,
            "references": str(args.references),
            "predictions": str(args.predictions),
        },
        "counts": {
            "sample_count": original["total"],
            "correct": original["correct"],
            "unresolved": original["unresolved"],
            "reference_essential": reference_labels.count(ESSENTIAL),
            "reference_nonessential": reference_labels.count(NON_ESSENTIAL),
            "prediction_essential": prediction_labels.count(ESSENTIAL),
            "prediction_nonessential": prediction_labels.count(NON_ESSENTIAL),
            "confusion_counts": original["confusion_counts"],
        },
        "results": {row["metric"]: row for row in rows},
    }

    if args.debug_samples > 0:
        for index in range(min(args.debug_samples, len(records))):
            print("=" * 72)
            print(f"Record: {index + 1}")
            print(f"Reference: {reference_labels[index]}")
            print(f"Prediction text: {prediction_texts[index]}")
            print(f"Parsed prediction: {prediction_labels[index]}")

    print(f"Reference records: {len(records)}")
    print(f"Prediction records: {len(prediction_texts)}")
    print(f"Unresolved predictions: {original['unresolved']}")
    print_results(rows, args.confidence_level)

    write_csv(args.output_csv, rows)
    write_json(args.output_json, payload)
    if args.errors_csv is not None:
        write_errors_csv(
            args.errors_csv, reference_labels, prediction_texts, prediction_labels
        )
    print(f"CSV results: {args.output_csv}")
    print(f"JSON results: {args.output_json}")
    if args.errors_csv is not None:
        print(f"Error records: {args.errors_csv}")


if __name__ == "__main__":
    main()
