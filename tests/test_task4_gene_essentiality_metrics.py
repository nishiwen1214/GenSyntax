"""Unit tests for the Task 4 gene-essentiality evaluator."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "task4_gene_essentiality_metrics.py"
)
SPEC = importlib.util.spec_from_file_location("task4_evaluator", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


class Task4EvaluatorTests(unittest.TestCase):
    def test_nonessential_is_parsed_before_essential(self) -> None:
        self.assertEqual(EVALUATOR.parse_label("non-essential"), "non-essential")
        self.assertEqual(EVALUATOR.parse_label("non essential"), "non-essential")
        self.assertEqual(EVALUATOR.parse_label("not essential"), "non-essential")
        self.assertEqual(EVALUATOR.parse_label("essential"), "essential")

    def test_empty_and_invalid_predictions_are_unresolved(self) -> None:
        self.assertIsNone(EVALUATOR.parse_label(""))
        self.assertIsNone(EVALUATOR.parse_label("unknown"))

    def test_internal_blank_prediction_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.txt"
            path.write_text("essential\n\nnon-essential\n", encoding="utf-8")
            self.assertEqual(
                EVALUATOR.load_predictions(path, 3),
                ["essential", "", "non-essential"],
            )

    def test_confusion_counts_and_metrics(self) -> None:
        references = ["essential", "essential", "non-essential", "non-essential"]
        predictions = ["essential", None, "essential", "non-essential"]
        result = EVALUATOR.calculate_metrics(references, predictions)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["unresolved"], 1)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(
            result["confusion_counts"]["essential"], {"tp": 1, "fp": 1, "fn": 1}
        )
        self.assertEqual(
            result["confusion_counts"]["non-essential"],
            {"tp": 1, "fp": 0, "fn": 1},
        )

    def test_bootstrap_accuracy_matches_direct_resampling(self) -> None:
        references = ["essential", "essential", "non-essential", "non-essential"]
        predictions = ["essential", None, "essential", "non-essential"]
        result = EVALUATOR.bootstrap_metrics(
            reference_labels=references,
            prediction_labels=predictions,
            replicates=20,
            confidence_level=0.90,
            seed=42,
            show_progress=False,
        )["accuracy"]

        hit = np.asarray([1.0, 0.0, 0.0, 1.0])
        rng = np.random.default_rng(42)
        direct = np.asarray(
            [
                hit[rng.integers(0, len(hit), size=len(hit))].mean() * 100.0
                for _ in range(20)
            ]
        )
        self.assertAlmostEqual(result["mean_percent"], float(np.mean(direct)))
        self.assertAlmostEqual(
            result["ci_lower_percent"], float(np.percentile(direct, 5.0))
        )
        self.assertAlmostEqual(
            result["ci_upper_percent"], float(np.percentile(direct, 95.0))
        )


if __name__ == "__main__":
    unittest.main()
