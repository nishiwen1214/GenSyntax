#!/usr/bin/env python3
"""Evaluate Task 3 circular contig-order predictions.

Predictions are correct when they are cyclic rotations of the reference order.
Reverse order is not accepted by default. Missing, duplicated, unexpected, or
additional contig identifiers remain in the denominator and count as errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_BOOTSTRAP_REPLICATES = 100
DEFAULT_CONFIDENCE_LEVEL = 0.90
DEFAULT_RANDOM_SEED = 42
SUPPORTED_CONTIG_COUNTS = (3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate circular contig-order accuracy and percentile bootstrap "
            "confidence intervals for GenSyntax Task 3 predictions."
        )
    )
    parser.add_argument(
        "--references",
        type=Path,
        required=True,
        help="JSON array containing reference contig orders.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Text file containing one predicted contig order per line.",
    )
    parser.add_argument(
        "--num-contigs",
        choices=("auto", "3", "4", "5"),
        default="auto",
        help=(
            "Expected contig count. 'auto' reads every reference order and "
            "requires all records to agree (default: auto)."
        ),
    )
    parser.add_argument(
        "--allow-reverse",
        action="store_true",
        help=(
            "Also accept cyclic rotations of the reversed order. Disabled by "
            "default and not used for the manuscript results."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/task3_accuracy.csv"),
        help="Destination for the summary CSV file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/task3_accuracy.json"),
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


def extract_contig_order(text: Any) -> list[int]:
    """Extract ordered contig identifiers from common output formats."""
    value = "" if text is None else str(text).strip()
    if not value:
        return []
    return [
        int(identifier)
        for identifier in re.findall(
            r"contig[\s_-]*(\d+)", value, flags=re.IGNORECASE
        )
    ]


def is_valid_order(order: Sequence[int], contig_count: int) -> bool:
    """Return whether an order contains each identifier from 1 to N once."""
    return len(order) == contig_count and set(order) == set(range(1, contig_count + 1))


def is_circular_match(
    reference: Sequence[int], prediction: Sequence[int], allow_reverse: bool = False
) -> bool:
    """Compare two orders under cyclic rotation, optionally allowing reversal."""
    if len(reference) != len(prediction) or not reference:
        return False
    if sorted(reference) != sorted(prediction):
        return False

    candidates = [list(reference)]
    if allow_reverse:
        candidates.append(list(reversed(reference)))

    prediction_list = list(prediction)
    for candidate in candidates:
        for shift in range(len(candidate)):
            if candidate[shift:] + candidate[:shift] == prediction_list:
                return True
    return False


def get_reference_text(record: dict[str, Any]) -> Any:
    for key in ("Output", "output", "label", "Label", "answer", "Answer"):
        if key in record:
            return record[key]
    raise ValueError(
        "Reference record has no order field. Expected one of: "
        "Output, output, label, Label, answer, Answer"
    )


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


def parse_reference_orders(records: Sequence[dict[str, Any]]) -> list[list[int]]:
    orders: list[list[int]] = []
    for index, record in enumerate(records):
        order = extract_contig_order(get_reference_text(record))
        if len(order) not in SUPPORTED_CONTIG_COUNTS:
            raise ValueError(
                f"Reference record {index + 1} contains {len(order)} contigs; "
                f"expected one of {SUPPORTED_CONTIG_COUNTS}"
            )
        if not is_valid_order(order, len(order)):
            raise ValueError(
                f"Reference record {index + 1} is not a permutation of "
                f"1..{len(order)}: {order}"
            )
        orders.append(order)
    return orders


def resolve_contig_count(reference_orders: Sequence[Sequence[int]], mode: str) -> int:
    detected_counts = {len(order) for order in reference_orders}
    if mode != "auto":
        expected = int(mode)
        mismatches = [
            index + 1
            for index, order in enumerate(reference_orders)
            if len(order) != expected
        ]
        if mismatches:
            preview = ", ".join(map(str, mismatches[:10]))
            raise ValueError(
                f"Reference records do not match --num-contigs {expected}: {preview}"
            )
        return expected

    if len(detected_counts) != 1:
        raise ValueError(
            "Auto-detection found mixed contig counts in one evaluation file: "
            f"{sorted(detected_counts)}. Evaluate each setting separately."
        )
    contig_count = detected_counts.pop()
    if contig_count not in SUPPORTED_CONTIG_COUNTS:
        raise ValueError(f"Unsupported detected contig count: {contig_count}")
    return contig_count


def evaluate_predictions(
    reference_orders: Sequence[Sequence[int]],
    prediction_texts: Sequence[str],
    contig_count: int,
    allow_reverse: bool,
) -> tuple[list[bool], list[bool], list[dict[str, Any]]]:
    if len(reference_orders) != len(prediction_texts):
        raise ValueError("Reference and prediction counts must be identical")

    correctness: list[bool] = []
    resolved_flags: list[bool] = []
    errors: list[dict[str, Any]] = []

    for index, (reference, prediction_text) in enumerate(
        zip(reference_orders, prediction_texts)
    ):
        prediction = extract_contig_order(prediction_text)
        resolved = is_valid_order(prediction, contig_count)
        correct = resolved and is_circular_match(
            reference, prediction, allow_reverse=allow_reverse
        )
        correctness.append(correct)
        resolved_flags.append(resolved)
        if not correct:
            errors.append(
                {
                    "record_index": index + 1,
                    "reference_order": "-".join(map(str, reference)),
                    "prediction_order": "-".join(map(str, prediction)),
                    "resolved": resolved,
                    "prediction_text": prediction_text,
                }
            )
    return correctness, resolved_flags, errors


def summarize_indices(
    correctness: Sequence[bool], resolved_flags: Sequence[bool], indices: Iterable[int]
) -> dict[str, float | int]:
    total = 0
    correct = 0
    unresolved = 0
    for index in indices:
        total += 1
        correct += int(correctness[index])
        unresolved += int(not resolved_flags[index])
    return {
        "total": total,
        "correct": correct,
        "unresolved": unresolved,
        "accuracy_percent": correct / total * 100.0 if total else float("nan"),
    }


def bootstrap_accuracy(
    correctness: Sequence[bool],
    resolved_flags: Sequence[bool],
    replicates: int,
    confidence_level: float,
    seed: int,
    show_progress: bool,
) -> dict[str, float | int]:
    if not correctness:
        raise ValueError("The evaluation dataset is empty")
    if len(correctness) != len(resolved_flags):
        raise ValueError("Correctness and resolution vectors must be identical in length")
    if replicates <= 0:
        raise ValueError("--bootstrap-replicates must be greater than zero")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("--confidence-level must be between zero and one")

    sample_count = len(correctness)
    rng = np.random.default_rng(seed)
    accuracies: list[float] = []
    unresolved_counts: list[int] = []

    for replicate_index in range(replicates):
        indices = rng.integers(0, sample_count, size=sample_count)
        summary = summarize_indices(correctness, resolved_flags, indices)
        accuracies.append(float(summary["accuracy_percent"]))
        unresolved_counts.append(int(summary["unresolved"]))
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


def build_result(
    contig_count: int,
    sample_count: int,
    original: dict[str, float | int],
    bootstrap: dict[str, float | int],
) -> dict[str, Any]:
    return {
        "contig_count": contig_count,
        "sample_count": sample_count,
        "correct": original["correct"],
        "total": original["total"],
        "unresolved": original["unresolved"],
        "original_accuracy_percent": original["accuracy_percent"],
        **bootstrap,
    }


def write_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)


def write_errors_csv(path: Path, errors: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "record_index",
        "reference_order",
        "prediction_order",
        "resolved",
        "prediction_text",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)


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
    reference_orders: Sequence[Sequence[int]],
    prediction_texts: Sequence[str],
    correctness: Sequence[bool],
    resolved_flags: Sequence[bool],
    count: int,
) -> None:
    for index in range(min(count, len(reference_orders))):
        print("=" * 72)
        print(f"Record: {index + 1}")
        print(f"Reference order: {list(reference_orders[index])}")
        print(f"Prediction text: {prediction_texts[index]}")
        print(f"Prediction order: {extract_contig_order(prediction_texts[index])}")
        print(f"Resolved: {resolved_flags[index]}")
        print(f"Correct: {correctness[index]}")


def print_results(result: dict[str, Any], confidence_level: float) -> None:
    confidence_percent = int(round(confidence_level * 100))
    interval = (
        f"{result['mean_percent']:.2f}% "
        f"[{result['ci_lower_percent']:.2f}, {result['ci_upper_percent']:.2f}]"
    )
    print("\nTask 3 circular contig-order accuracy")
    print("-" * 94)
    print(
        f"{'Contigs':<10} {'Original':>12} {'Correct/Total':>16} "
        f"{'Unresolved':>12} {f'Bootstrap mean ({confidence_percent}% CI)':>36}"
    )
    print(
        f"{result['contig_count']:<10} "
        f"{result['original_accuracy_percent']:>11.2f}% "
        f"{result['correct']:>7}/{result['total']:<8} "
        f"{result['unresolved']:>12} {interval:>36}"
    )


def main() -> None:
    args = parse_args()
    validate_arguments(args)
    records = load_references(args.references)
    predictions = load_predictions(args.predictions, len(records))
    reference_orders = parse_reference_orders(records)
    contig_count = resolve_contig_count(reference_orders, args.num_contigs)

    correctness, resolved_flags, errors = evaluate_predictions(
        reference_orders=reference_orders,
        prediction_texts=predictions,
        contig_count=contig_count,
        allow_reverse=args.allow_reverse,
    )
    original = summarize_indices(
        correctness, resolved_flags, range(len(reference_orders))
    )
    bootstrap = bootstrap_accuracy(
        correctness=correctness,
        resolved_flags=resolved_flags,
        replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
        show_progress=not args.quiet,
    )
    result = build_result(contig_count, len(records), original, bootstrap)
    payload = {
        "task": "Task 3: circular contig-order prediction",
        "settings": {
            "contig_count": contig_count,
            "allow_reverse": args.allow_reverse,
            "bootstrap_replicates": args.bootstrap_replicates,
            "confidence_level": args.confidence_level,
            "confidence_interval_method": "two-sided percentile bootstrap",
            "random_seed": args.seed,
            "references": str(args.references),
            "predictions": str(args.predictions),
        },
        "results": result,
    }

    if args.debug_samples > 0:
        print_debug_records(
            reference_orders,
            predictions,
            correctness,
            resolved_flags,
            args.debug_samples,
        )

    print(f"Reference records: {len(records)}")
    print(f"Prediction records: {len(predictions)}")
    print(f"Detected contig count: {contig_count}")
    print(f"Reverse-order matching: {args.allow_reverse}")
    print_results(result, args.confidence_level)

    write_csv(args.output_csv, result)
    write_json(args.output_json, payload)
    if args.errors_csv is not None:
        write_errors_csv(args.errors_csv, errors)
    print(f"CSV results: {args.output_csv}")
    print(f"JSON results: {args.output_json}")
    if args.errors_csv is not None:
        print(f"Error records: {args.errors_csv}")


if __name__ == "__main__":
    main()
