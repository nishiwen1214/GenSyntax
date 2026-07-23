"""Unit tests for iterative minimal-genome inference."""

from __future__ import annotations

import importlib.util
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "minimal_genome_inference.py"
SPEC = importlib.util.spec_from_file_location("minimal_genome", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MINIMAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MINIMAL)


def logprob(token: str, probability: float) -> SimpleNamespace:
    return SimpleNamespace(decoded_token=token, logprob=math.log(probability))


class MinimalGenomeTests(unittest.TestCase):
    def test_probability_is_normalized_over_allowed_labels(self) -> None:
        alternatives = {
            1: logprob(" essential", 0.3),
            2: logprob(" non", 0.1),
            3: logprob(" other", 0.6),
        }
        self.assertAlmostEqual(
            MINIMAL.essential_probability_from_logprobs(alternatives), 0.75
        )

    def test_missing_label_probability_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            MINIMAL.essential_probability_from_logprobs(
                {1: logprob(" essential", 0.9)}
            )

    def test_prompt_uses_organism_and_current_context(self) -> None:
        prompt = MINIMAL.build_instruction(
            "Test organism",
            "protein B",
            ["protein A", "protein B", "protein C"],
            1,
        )
        self.assertIn("Test organism chromosome", prompt)
        self.assertIn("$$protein B$$", prompt)
        self.assertTrue(prompt.endswith("[protein A][protein C]"))

    def test_iterative_reduction_deletes_and_restarts(self) -> None:
        record = {
            "Source": "Test organism",
            "Protein_products": [
                ["g0", "x", "A"],
                ["g1", "x", "B"],
                ["g2", "x", "C"],
            ],
        }
        probabilities = iter([0.1, 0.9, 0.8])
        original = MINIMAL.predict_essential_probability
        MINIMAL.predict_essential_probability = lambda *_: next(probabilities)
        try:
            result = MINIMAL.iterative_reduction(
                None, record, 0.5, random.Random(1), None
            )
        finally:
            MINIMAL.predict_essential_probability = original

        metadata = result["minimal_genome_metadata"]
        self.assertEqual(metadata["deleted_gene_count"], 1)
        self.assertEqual(metadata["retained_gene_count"], 2)
        self.assertEqual(metadata["model_evaluations"], 3)
        self.assertNotIn("predicted_necessary_products", record)

    def test_threshold_replicates_have_independent_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = MINIMAL.write_results(
                Path(directory), "model", 0.5, 1, 42, [{"result": "first"}]
            )
            second = MINIMAL.write_results(
                Path(directory), "model", 0.2, 2, 43, [{"result": "second"}]
            )
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())


if __name__ == "__main__":
    unittest.main()
