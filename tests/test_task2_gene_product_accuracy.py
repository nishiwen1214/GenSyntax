"""Unit tests for the Task 2 gene-product accuracy evaluator."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "task2_gene_product_accuracy.py"
)
SPEC = importlib.util.spec_from_file_location("task2_evaluator", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


class Task2EvaluatorTests(unittest.TestCase):
    def test_answer_parsing_priority(self) -> None:
        self.assertEqual(EVALUATOR.extract_answer_candidates("[D]"), ["D"])
        self.assertEqual(EVALUATOR.extract_answer_candidates("1. C"), ["C"])
        self.assertEqual(EVALUATOR.extract_answer_candidates("Answer: B"), ["B"])
        self.assertEqual(EVALUATOR.extract_answer_candidates("A"), ["A"])

    def test_out_of_range_answer_is_unresolved(self) -> None:
        allowed = set("ABCD")
        self.assertIsNone(EVALUATOR.extract_single_answer("[E]", allowed))
        self.assertEqual(EVALUATOR.extract_single_answer("[D]", allowed), "D")

    def test_four_option_auto_detection(self) -> None:
        record = {
            "Input": "Choose one. A: one; B: two; C: three; D: four",
            "Output": "[A]",
        }
        self.assertEqual(EVALUATOR.detect_option_count(record), 4)

    def test_eight_option_auto_detection(self) -> None:
        record = {
            "Input": (
                "Choose one. A: one; B: two; C: three; D: four; "
                "E: five; F: six; G: seven; H: eight"
            ),
            "Output": "[H]",
        }
        self.assertEqual(EVALUATOR.detect_option_count(record), 8)

    def test_internal_blank_prediction_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.txt"
            path.write_text("[A]\n\n[B]\n", encoding="utf-8")
            self.assertEqual(
                EVALUATOR.load_predictions(path, 3), ["[A]", "", "[B]"]
            )

    def test_accuracy_counts_unresolved_as_incorrect(self) -> None:
        result = EVALUATOR.evaluate_indices(
            reference_answers=["A", "B", "C", "D"],
            prediction_answers=["A", None, "D", "D"],
            indices=range(4),
        )
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["unresolved"], 1)
        self.assertEqual(result["accuracy_percent"], 50.0)

    def test_bootstrap_matches_direct_resampling(self) -> None:
        references = ["A", "B", "C", "D"]
        predictions = ["A", "B", None, "A"]
        result = EVALUATOR.bootstrap_accuracy(
            reference_answers=references,
            prediction_answers=predictions,
            replicates=20,
            confidence_level=0.90,
            seed=42,
            show_progress=False,
        )

        hit = np.asarray([1.0, 1.0, 0.0, 0.0])
        rng = np.random.default_rng(42)
        direct = np.asarray(
            [
                hit[rng.integers(0, len(hit), size=len(hit))].mean() * 100.0
                for _ in range(20)
            ]
        )
        self.assertAlmostEqual(result["mean_percent"], float(np.mean(direct)))
        self.assertAlmostEqual(
            result["standard_deviation_percent"], float(np.std(direct, ddof=1))
        )
        self.assertAlmostEqual(
            result["ci_lower_percent"], float(np.percentile(direct, 5.0))
        )
        self.assertAlmostEqual(
            result["ci_upper_percent"], float(np.percentile(direct, 95.0))
        )


if __name__ == "__main__":
    unittest.main()
