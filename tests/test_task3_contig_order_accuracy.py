"""Unit tests for the Task 3 contig-order evaluator."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "task3_contig_order_accuracy.py"
)
SPEC = importlib.util.spec_from_file_location("task3_evaluator", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


class Task3EvaluatorTests(unittest.TestCase):
    def test_extract_contig_order_variants(self) -> None:
        text = "Contig1-CONTIG 2-contig_3-contig-4"
        self.assertEqual(EVALUATOR.extract_contig_order(text), [1, 2, 3, 4])

    def test_cyclic_rotations_are_correct(self) -> None:
        reference = [3, 4, 5, 1, 2]
        self.assertTrue(EVALUATOR.is_circular_match(reference, [1, 2, 3, 4, 5]))
        self.assertTrue(EVALUATOR.is_circular_match(reference, [4, 5, 1, 2, 3]))

    def test_reverse_order_is_disabled_by_default(self) -> None:
        reference = [1, 2, 3, 4, 5]
        prediction = [5, 4, 3, 2, 1]
        self.assertFalse(EVALUATOR.is_circular_match(reference, prediction))
        self.assertTrue(
            EVALUATOR.is_circular_match(reference, prediction, allow_reverse=True)
        )

    def test_order_must_be_complete_permutation(self) -> None:
        self.assertTrue(EVALUATOR.is_valid_order([1, 3, 2], 3))
        self.assertFalse(EVALUATOR.is_valid_order([1, 2], 3))
        self.assertFalse(EVALUATOR.is_valid_order([1, 2, 2], 3))
        self.assertFalse(EVALUATOR.is_valid_order([1, 2, 3, 4], 3))

    def test_additional_contig_is_unresolved(self) -> None:
        correctness, resolved, _ = EVALUATOR.evaluate_predictions(
            reference_orders=[[1, 2, 3]],
            prediction_texts=["Contig 1-Contig 2-Contig 3-Contig 4"],
            contig_count=3,
            allow_reverse=False,
        )
        self.assertEqual(correctness, [False])
        self.assertEqual(resolved, [False])

    def test_internal_blank_prediction_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.txt"
            path.write_text("Contig 1\n\nContig 2\n", encoding="utf-8")
            self.assertEqual(
                EVALUATOR.load_predictions(path, 3), ["Contig 1", "", "Contig 2"]
            )

    def test_bootstrap_matches_direct_resampling(self) -> None:
        correctness = [True, True, False, False]
        resolved = [True, True, True, False]
        result = EVALUATOR.bootstrap_accuracy(
            correctness=correctness,
            resolved_flags=resolved,
            replicates=20,
            confidence_level=0.90,
            seed=42,
            show_progress=False,
        )

        hit = np.asarray(correctness, dtype=float)
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
