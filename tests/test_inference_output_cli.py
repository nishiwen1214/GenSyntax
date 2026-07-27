"""Tests for the shared output-file behavior of Task 1-4 inference scripts."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "Plasmid_host_identification.py",
    "Gene_function_prediction.py",
    "Contig_order_prediction.py",
    "Gene_essentiality_prediction.py",
)


class _GeneratedText:
    def __init__(self, text: str) -> None:
        self.text = text


class _RequestOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [_GeneratedText(text)]


class _FakeLLM:
    def __init__(self, **_: object) -> None:
        pass

    def chat(self, prompts: list[object], _: object) -> list[_RequestOutput]:
        return [_RequestOutput(f"prediction {index}") for index, _ in enumerate(prompts)]


def _load_script(script_name: str):
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = _FakeLLM
    fake_vllm.SamplingParams = object
    module_name = f"test_{Path(script_name).stem}"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / script_name)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {script_name}")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"vllm": fake_vllm}):
        spec.loader.exec_module(module)
    return module


class InferenceOutputFileTests(unittest.TestCase):
    def test_explicit_output_file_is_used_by_all_tasks(self) -> None:
        for script_name in SCRIPT_NAMES:
            with self.subTest(script=script_name), tempfile.TemporaryDirectory() as directory:
                module = _load_script(script_name)
                root = Path(directory)
                input_path = root / "input.json"
                output_path = root / "nested" / "predictions.txt"
                input_path.write_text(
                    json.dumps([{"instruction": "first"}, {"instruction": "second"}]),
                    encoding="utf-8",
                )

                with redirect_stdout(io.StringIO()):
                    module.run_inference(
                        model_path="example/model",
                        input_json_paths=[str(input_path)],
                        output_dir=str(root / "automatic"),
                        tensor_parallel_size=1,
                        sampling_params=object(),
                        output_file=str(output_path),
                    )

                self.assertEqual(
                    output_path.read_text(encoding="utf-8"),
                    "prediction 0\nprediction 1",
                )
                self.assertFalse((root / "automatic").exists())

    def test_output_file_rejects_multiple_inputs(self) -> None:
        for script_name in SCRIPT_NAMES:
            with self.subTest(script=script_name):
                module = _load_script(script_name)
                argv = [
                    script_name,
                    "--model-paths",
                    "model",
                    "--input-json-paths",
                    "one.json",
                    "two.json",
                    "--output-file",
                    "predictions.txt",
                ]
                with (
                    patch.object(sys, "argv", argv),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    module.parse_args()

    def test_readme_evaluates_the_files_created_by_inference(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        generated = re.findall(r"--output-file\s+(\S+)", readme)
        evaluated = re.findall(r"--predictions\s+(\S+)", readme)
        self.assertEqual(generated, evaluated)
        self.assertNotIn("DATASET_DIR/", readme)
        self.assertNotRegex(readme, r"--predictions\s+/path/to/")


if __name__ == "__main__":
    unittest.main()
