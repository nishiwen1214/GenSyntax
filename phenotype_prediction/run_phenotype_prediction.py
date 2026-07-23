#!/usr/bin/env python3
"""Reproduce the ten microbial-phenotype classification experiments."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


MODEL_NAMES = (
    "logistic_regression",
    "random_forest",
    "support_vector_machine",
    "gradient_boosting",
    "multilayer_perceptron",
)
DEFAULT_SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class MatchedDataset:
    """One phenotype dataset after deterministic species–embedding matching."""

    phenotype: str
    display_name: str
    features: np.ndarray
    labels: np.ndarray
    species: tuple[str, ...]
    sources: tuple[str, ...]
    diagnostics: dict[str, Any]


def normalize_name(value: Any) -> str:
    """Normalize organism names for deterministic matching."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = re.sub(r"[^A-Za-z0-9\s]", " ", str(value).strip().lower())
    return re.sub(r"\s+", " ", text).strip()


def normalize_category(value: Any) -> str | None:
    """Normalize a categorical BacDive label without merging biological classes."""
    if pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    return text or None


def parse_measurement(value: Any) -> float | None:
    """Parse a scalar or range and return the scalar or range midpoint."""
    if pd.isna(value):
        return None
    text = str(value).strip().replace("−", "-")
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    range_match = re.search(
        rf"({number})\s*(?:–|—|\bto\b|(?<=\d)-(?=\d))\s*({number})",
        text,
        flags=re.IGNORECASE,
    )
    try:
        if range_match:
            return (float(range_match.group(1)) + float(range_match.group(2))) / 2.0
        scalar = re.search(number, text)
        return float(scalar.group(0)) if scalar else None
    except ValueError:
        return None


def transform_label(value: Any, transform: str) -> str | None:
    """Apply the manuscript-defined phenotype encoding."""
    if transform == "categorical":
        return normalize_category(value)

    if transform == "cell_shape":
        label = normalize_category(value)
        if label is None:
            return None
        label = re.sub(r"[\s_-]*shaped$", "", label)
        aliases = {
            "bacillus": "rod",
            "spirillum": "spiral",
            "comma": "vibrio",
        }
        return aliases.get(label, label)

    if transform == "sign":
        if pd.isna(value):
            return None
        label = str(value).strip().lower()
        if label in {"+", "positive", "pos", "1", "true"}:
            return "positive"
        if label in {"-", "negative", "neg", "0", "false"}:
            return "negative"
        # Ambiguous values such as "+/-" are not assigned a class.
        return None

    if transform in {"motility", "presence"}:
        if pd.isna(value):
            return None
        label = str(value).strip().lower()
        positive = label in {"1", "1.0", "true", "yes", "positive", "present"}
        negative = label in {"0", "0.0", "false", "no", "negative", "absent"}
        if not positive and not negative:
            return None
        if transform == "motility":
            return "motile" if positive else "non-motile"
        return "present" if positive else "absent"

    measurement = parse_measurement(value)
    if measurement is None:
        return None
    if transform == "temperature":
        if measurement < 20.0:
            return "low (<20 C)"
        if measurement <= 40.0:
            return "medium (20-40 C)"
        return "high (>40 C)"
    if transform == "cell_length":
        return "short (<=2 um)" if measurement <= 2.0 else "long (>2 um)"
    if transform == "cell_width":
        return "narrow (<=0.5 um)" if measurement <= 0.5 else "wide (>0.5 um)"
    raise ValueError(f"Unknown label transform: {transform}")


def load_specs(path: Path) -> dict[str, dict[str, str]]:
    """Load and validate phenotype task definitions."""
    with path.open("r", encoding="utf-8") as handle:
        specs = json.load(handle)
    required = {"display_name", "csv_file", "label_column", "transform"}
    for name, spec in specs.items():
        missing = required.difference(spec)
        if missing:
            raise ValueError(f"Phenotype {name} is missing fields: {sorted(missing)}")
    return specs


def load_embeddings(
    path: Path,
    source_field: str,
    embedding_field: str,
) -> list[tuple[str, str, np.ndarray]]:
    """Load valid, finite, fixed-dimensional genome embeddings."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Embedding JSON must contain an array of objects.")

    embeddings: list[tuple[str, str, np.ndarray]] = []
    seen_sources: set[str] = set()
    expected_dimension: int | None = None
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        source_raw = item.get(source_field)
        source = normalize_name(source_raw)
        vector_raw = item.get(embedding_field)
        if not source or source in seen_sources or not isinstance(vector_raw, list):
            continue
        try:
            vector = np.asarray(vector_raw, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            continue
        if expected_dimension is None:
            expected_dimension = int(vector.size)
        elif vector.size != expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch at record {index}: "
                f"{vector.size} != {expected_dimension}."
            )
        embeddings.append((source, str(source_raw).strip(), vector))
        seen_sources.add(source)

    if not embeddings:
        raise ValueError(f"No valid embeddings found in {path}.")
    return embeddings


def _resolve_species_labels(
    frame: pd.DataFrame,
    label_column: str,
    transform: str,
    conflict_policy: str = "first",
    allowed_labels: set[str] | None = None,
) -> tuple[dict[str, str], dict[str, int]]:
    """Collapse duplicate species under an explicit conflict policy."""
    labels_by_species: dict[str, list[str]] = defaultdict(list)
    invalid_rows = 0
    for species_raw, label_raw in zip(frame["species"], frame[label_column]):
        species = normalize_name(species_raw)
        label = transform_label(label_raw, transform)
        if not species or label is None:
            invalid_rows += 1
            continue
        if allowed_labels is not None and label not in allowed_labels:
            continue
        if label not in labels_by_species[species]:
            labels_by_species[species].append(label)

    conflicts = {
        species: labels
        for species, labels in labels_by_species.items()
        if len(labels) > 1
    }
    if conflict_policy == "error" and conflicts:
        raise ValueError(
            f"Conflicting phenotype labels detected: {list(conflicts.items())[:5]}"
        )
    if conflict_policy == "exclude":
        resolved = {
            species: labels[0]
            for species, labels in labels_by_species.items()
            if len(labels) == 1
        }
    elif conflict_policy == "first":
        resolved = {
            species: labels[0] for species, labels in labels_by_species.items()
        }
    else:
        raise ValueError(f"Unknown conflict policy: {conflict_policy}")
    return resolved, {
        "csv_rows": int(len(frame)),
        "invalid_rows": invalid_rows,
        "unique_labeled_species": len(labels_by_species),
        "conflicting_species": len(conflicts),
        "conflicting_species_excluded": (
            len(conflicts) if conflict_policy == "exclude" else 0
        ),
        "conflicting_species_retained_first": (
            len(conflicts) if conflict_policy == "first" else 0
        ),
    }


def _source_matches_species(source: str, species: str, policy: str) -> bool:
    """Match species using the reported or a stricter audit policy."""
    if policy == "contains":
        return species in source
    if policy == "prefix":
        return source == species or source.startswith(f"{species} ")
    raise ValueError(f"Unknown species matching policy: {policy}")


def prepare_dataset(
    phenotype: str,
    spec: dict[str, str],
    data_dir: Path,
    embeddings: list[tuple[str, str, np.ndarray]],
    min_class_size: int,
    conflict_policy: str = "first",
    match_policy: str = "contains",
) -> MatchedDataset:
    """Prepare one task using the supplied-code filtering and matching rules."""
    csv_path = data_dir / spec["csv_file"]
    frame = pd.read_csv(csv_path)
    required = {"species", spec["label_column"]}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    transformed_labels = [
        transform_label(value, spec["transform"])
        for value in frame[spec["label_column"]]
    ]
    row_class_counts = Counter(
        label for label in transformed_labels if label is not None
    )
    valid_classes = {
        label for label, count in row_class_counts.items() if count >= min_class_size
    }
    labels_by_species, diagnostics = _resolve_species_labels(
        frame,
        spec["label_column"],
        spec["transform"],
        conflict_policy=conflict_policy,
        allowed_labels=valid_classes,
    )
    features: list[np.ndarray] = []
    labels: list[str] = []
    species_values: list[str] = []
    sources: list[str] = []
    used_sources: set[str] = set()
    species_with_multiple_source_matches = 0

    # Preserve embedding-file order, matching at most one genome per species.
    for species, label in labels_by_species.items():
        candidates = [
            (source, source_raw, vector)
            for source, source_raw, vector in embeddings
            if source not in used_sources
            and _source_matches_species(source, species, match_policy)
        ]
        if len(candidates) > 1:
            species_with_multiple_source_matches += 1
        if candidates:
            source, source_raw, vector = candidates[0]
            features.append(vector)
            labels.append(label)
            species_values.append(species)
            sources.append(source_raw)
            used_sources.add(source)

    counts_before = Counter(labels)
    if len(counts_before) < 2:
        raise ValueError(
            f"{phenotype}: fewer than two classes remain after matching."
        )
    insufficient_for_stratification = {
        label: count for label, count in counts_before.items() if count < 2
    }
    if insufficient_for_stratification:
        raise ValueError(
            f"{phenotype}: matched classes cannot be stratified: "
            f"{insufficient_for_stratification}."
        )

    features_array = np.asarray(features, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=object)
    counts_after = Counter(labels_array.tolist())
    diagnostics.update(
        {
            "row_class_counts_before_filter": dict(sorted(row_class_counts.items())),
            "classes_retained_at_minimum_size": sorted(valid_classes),
            "matched_species": len(labels),
            "unmatched_species": len(labels_by_species) - len(labels),
            "matched_class_counts": dict(sorted(counts_after.items())),
            "min_class_size": min_class_size,
            "conflict_policy": conflict_policy,
            "match_policy": match_policy,
            "species_with_multiple_source_matches": (
                species_with_multiple_source_matches
            ),
            "embedding_dimension": int(features_array.shape[1]),
        }
    )
    return MatchedDataset(
        phenotype=phenotype,
        display_name=spec["display_name"],
        features=features_array,
        labels=labels_array,
        species=tuple(species_values),
        sources=tuple(sources),
        diagnostics=diagnostics,
    )


def build_models(seed: int, n_jobs: int) -> dict[str, Any]:
    """Instantiate the five classifiers used in the manuscript."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=seed,
            class_weight="balanced",
            n_jobs=n_jobs,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            random_state=seed,
            class_weight="balanced",
            n_jobs=n_jobs,
        ),
        "support_vector_machine": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            random_state=seed,
            class_weight="balanced",
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=seed,
            subsample=0.8,
        ),
        "multilayer_perceptron": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            max_iter=1000,
            batch_size=64,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        ),
    }


def metric_row(
    true: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate the manuscript-reported metrics."""
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "weighted_precision": float(
            precision_score(true, predicted, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(true, predicted, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(true, predicted, average="weighted", zero_division=0)
        ),
    }


def evaluate_dataset(
    dataset: MatchedDataset,
    embedding_name: str,
    model_names: tuple[str, ...],
    seeds: tuple[int, ...],
    test_size: float,
    n_jobs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate all selected models over identical seed-specific splits."""
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(dataset.labels)
    indices = np.arange(len(encoded_labels))
    metrics: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    for seed in seeds:
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=encoded_labels,
        )
        scaler = StandardScaler()
        train_features = scaler.fit_transform(dataset.features[train_indices])
        test_features = scaler.transform(dataset.features[test_indices])
        train_labels = encoded_labels[train_indices]
        test_labels = encoded_labels[test_indices]

        for partition, partition_indices in (
            ("train", train_indices),
            ("test", test_indices),
        ):
            for index in partition_indices:
                splits.append(
                    {
                        "embedding": embedding_name,
                        "phenotype": dataset.phenotype,
                        "seed": seed,
                        "partition": partition,
                        "species": dataset.species[index],
                        "source": dataset.sources[index],
                        "label": dataset.labels[index],
                    }
                )

        available_models = build_models(seed, n_jobs)
        for model_name in model_names:
            model = available_models[model_name]
            model.fit(train_features, train_labels)
            train_predicted = model.predict(train_features)
            test_predicted = model.predict(test_features)
            row = {
                "embedding": embedding_name,
                "phenotype": dataset.phenotype,
                "phenotype_display_name": dataset.display_name,
                "model": model_name,
                "seed": seed,
                "n_samples": len(indices),
                "n_classes": len(encoder.classes_),
                "n_train": len(train_indices),
                "n_test": len(test_indices),
            }
            row.update(
                {f"train_{key}": value for key, value in metric_row(
                    train_labels, train_predicted
                ).items()}
            )
            row.update(
                {f"test_{key}": value for key, value in metric_row(
                    test_labels, test_predicted
                ).items()}
            )
            metrics.append(row)

            decoded_predictions = encoder.inverse_transform(test_predicted)
            for index, predicted in zip(test_indices, decoded_predictions):
                predictions.append(
                    {
                        "embedding": embedding_name,
                        "phenotype": dataset.phenotype,
                        "model": model_name,
                        "seed": seed,
                        "species": dataset.species[index],
                        "source": dataset.sources[index],
                        "reference": dataset.labels[index],
                        "prediction": predicted,
                        "correct": bool(predicted == dataset.labels[index]),
                    }
                )
    return metrics, splits, predictions


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate independent seeds as mean and sample standard deviation."""
    group_columns = [
        "embedding",
        "phenotype",
        "phenotype_display_name",
        "model",
        "n_samples",
        "n_classes",
    ]
    metric_columns = [
        column
        for column in metrics.columns
        if column.startswith("train_") or column.startswith("test_")
    ]
    summary = (
        metrics.groupby(group_columns)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part)
        for column in summary.columns.to_flat_index()
    ]
    return summary


def save_plot(summary: pd.DataFrame, path: Path) -> None:
    """Plot mean test accuracy and weighted F1 for all tasks and models."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phenotypes = list(dict.fromkeys(summary["phenotype_display_name"]))
    models = list(dict.fromkeys(summary["model"]))
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(phenotypes) * 1.3), 9))
    width = 0.8 / len(models)
    x = np.arange(len(phenotypes))

    for axis, metric, ylabel in (
        (axes[0], "test_accuracy", "Accuracy"),
        (axes[1], "test_weighted_f1", "Weighted F1"),
    ):
        for model_index, model in enumerate(models):
            subset = summary.set_index(["phenotype_display_name", "model"])
            means = [
                subset.loc[(phenotype, model), f"{metric}_mean"]
                for phenotype in phenotypes
            ]
            errors = [
                subset.loc[(phenotype, model), f"{metric}_std"]
                for phenotype in phenotypes
            ]
            positions = x - 0.4 + width / 2 + model_index * width
            axis.bar(
                positions,
                means,
                width,
                yerr=errors,
                capsize=2,
                label=model.replace("_", " "),
            )
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, 1)
        axis.set_xticks(x, phenotypes, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=3, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).with_name("phenotypes.json")
    default_data = Path(__file__).resolve().parents[1] / "Data" / "BacDive"
    parser = argparse.ArgumentParser(
        description="Evaluate genome embeddings on ten reported BacDive phenotypes."
    )
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--embedding-name", required=True)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--phenotypes",
        nargs="+",
        default=["all"],
        help="Task keys from phenotypes.json, or all.",
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-class-size", type=int, default=5)
    parser.add_argument(
        "--conflict-policy",
        choices=("first", "exclude", "error"),
        default="first",
        help="Handling of multiple labels for one species (default: first).",
    )
    parser.add_argument(
        "--match-policy",
        choices=("contains", "prefix"),
        default="contains",
        help="Species-to-Source matching rule (default: contains).",
    )
    parser.add_argument("--source-field", default="Source")
    parser.add_argument("--embedding-field", default="products_embedding")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/phenotype_prediction")
    )
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.min_class_size < 2:
        raise ValueError("--min-class-size must be at least 2.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates.")

    specs = load_specs(args.config)
    selected = list(specs) if args.phenotypes == ["all"] else args.phenotypes
    unknown = sorted(set(selected).difference(specs))
    if unknown:
        raise ValueError(f"Unknown phenotype keys: {unknown}")
    embeddings = load_embeddings(
        args.embeddings, args.source_field, args.embedding_field
    )

    all_metrics: list[dict[str, Any]] = []
    all_splits: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "embedding_name": args.embedding_name,
        "embedding_file": str(args.embeddings),
        "valid_embedding_records": len(embeddings),
        "seeds": args.seeds,
        "test_size": args.test_size,
        "min_class_size": args.min_class_size,
        "phenotypes": {},
    }

    for phenotype in selected:
        dataset = prepare_dataset(
            phenotype,
            specs[phenotype],
            args.data_dir,
            embeddings,
            args.min_class_size,
            args.conflict_policy,
            args.match_policy,
        )
        diagnostics["phenotypes"][phenotype] = dataset.diagnostics
        metrics, splits, predictions = evaluate_dataset(
            dataset,
            args.embedding_name,
            tuple(args.models),
            tuple(args.seeds),
            args.test_size,
            args.n_jobs,
        )
        all_metrics.extend(metrics)
        all_splits.extend(splits)
        all_predictions.extend(predictions)
        print(
            f"[INFO] {phenotype}: n={len(dataset.labels)}, "
            f"classes={len(set(dataset.labels))}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(all_metrics)
    summary = summarize_metrics(metrics_frame)
    metrics_frame.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(all_splits).to_csv(
        args.output_dir / "split_assignments.csv", index=False
    )
    pd.DataFrame(all_predictions).to_csv(
        args.output_dir / "test_predictions.csv", index=False
    )
    with (args.output_dir / "data_diagnostics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(diagnostics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if args.plot:
        save_plot(summary, args.output_dir / "phenotype_performance.pdf")
    print(f"[INFO] Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
