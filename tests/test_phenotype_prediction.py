"""Tests for the consolidated microbial-phenotype workflow."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "phenotype_prediction"
    / "run_phenotype_prediction.py"
)
SPEC = importlib.util.spec_from_file_location("phenotype_prediction", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PHENOTYPE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PHENOTYPE
SPEC.loader.exec_module(PHENOTYPE)


class PhenotypePredictionTests(unittest.TestCase):
    def test_measurement_ranges_use_midpoints(self) -> None:
        self.assertEqual(PHENOTYPE.parse_measurement("1-3 µm"), 2.0)
        self.assertEqual(PHENOTYPE.parse_measurement("20 to 40 C"), 30.0)
        self.assertEqual(PHENOTYPE.parse_measurement("0.5"), 0.5)

    def test_manuscript_discretization_boundaries(self) -> None:
        self.assertEqual(
            PHENOTYPE.transform_label(19.9, "temperature"), "low (<20 C)"
        )
        self.assertEqual(
            PHENOTYPE.transform_label(20, "temperature"), "medium (20-40 C)"
        )
        self.assertEqual(
            PHENOTYPE.transform_label(40, "temperature"), "medium (20-40 C)"
        )
        self.assertEqual(
            PHENOTYPE.transform_label(40.1, "temperature"), "high (>40 C)"
        )
        self.assertEqual(
            PHENOTYPE.transform_label(2, "cell_length"), "short (<=2 um)"
        )
        self.assertEqual(
            PHENOTYPE.transform_label(0.5, "cell_width"), "narrow (<=0.5 um)"
        )

    def test_categorical_cleaning_and_ambiguous_sign(self) -> None:
        self.assertEqual(
            PHENOTYPE.transform_label("rod-shaped", "cell_shape"), "rod"
        )
        self.assertEqual(PHENOTYPE.transform_label("+", "sign"), "positive")
        self.assertEqual(PHENOTYPE.transform_label("-", "sign"), "negative")
        self.assertIsNone(PHENOTYPE.transform_label("+/-", "sign"))

    def test_conflicting_species_labels_are_excluded(self) -> None:
        frame = pd.DataFrame(
            {
                "species": ["Species one", "Species one", "Species two"],
                "label": ["positive", "negative", "positive"],
            }
        )
        labels, diagnostics = PHENOTYPE._resolve_species_labels(
            frame, "label", "categorical", conflict_policy="exclude"
        )
        self.assertEqual(labels, {"species two": "positive"})
        self.assertEqual(diagnostics["conflicting_species_excluded"], 1)

    def test_default_conflict_policy_preserves_first_label(self) -> None:
        frame = pd.DataFrame(
            {
                "species": ["Species one", "Species one"],
                "label": ["positive", "negative"],
            }
        )
        labels, diagnostics = PHENOTYPE._resolve_species_labels(
            frame, "label", "categorical"
        )
        self.assertEqual(labels, {"species one": "positive"})
        self.assertEqual(diagnostics["conflicting_species_retained_first"], 1)

    def test_embedding_validation_rejects_dimension_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.json"
            path.write_text(
                json.dumps(
                    [
                        {"Source": "Species one", "products_embedding": [1, 2]},
                        {"Source": "Species two", "products_embedding": [1, 2, 3]},
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                PHENOTYPE.load_embeddings(path, "Source", "products_embedding")

    def test_three_seed_metrics_are_aggregated(self) -> None:
        rows = []
        for seed, value in zip((42, 43, 44), (0.7, 0.8, 0.9)):
            rows.append(
                {
                    "embedding": "model",
                    "phenotype": "trait",
                    "phenotype_display_name": "Trait",
                    "model": "logistic_regression",
                    "seed": seed,
                    "n_samples": 30,
                    "n_classes": 2,
                    "train_accuracy": value,
                    "test_accuracy": value,
                }
            )
        summary = PHENOTYPE.summarize_metrics(pd.DataFrame(rows))
        self.assertAlmostEqual(summary.loc[0, "test_accuracy_mean"], 0.8)
        self.assertAlmostEqual(summary.loc[0, "test_accuracy_std"], 0.1)


if __name__ == "__main__":
    unittest.main()
