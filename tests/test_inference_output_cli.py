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
    last_options: dict[str, object] = {}

    def __init__(self, **options: object) -> None:
        type(self).last_options = options

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
                        max_model_len=131072,
                        rope_scaling={
                            "rope_type": "yarn",
                            "factor": 4.0,
                            "original_max_position_embeddings": 32768,
                        },
                    )

                self.assertEqual(
                    output_path.read_text(encoding="utf-8"),
                    "prediction 0\nprediction 1",
                )
                self.assertFalse((root / "automatic").exists())
                self.assertEqual(_FakeLLM.last_options["max_model_len"], 131072)
                self.assertEqual(
                    _FakeLLM.last_options["rope_scaling"]["rope_type"], "yarn"
                )

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

    def test_cli_accepts_128k_yarn_configuration(self) -> None:
        rope_scaling = (
            '{"rope_type":"yarn","factor":4.0,'
            '"original_max_position_embeddings":32768}'
        )
        for script_name in SCRIPT_NAMES:
            with self.subTest(script=script_name):
                module = _load_script(script_name)
                argv = [
                    script_name,
                    "--model-paths",
                    "model",
                    "--input-json-paths",
                    "input.json",
                    "--output-file",
                    "predictions.txt",
                    "--max-model-len",
                    "131072",
                    "--rope-scaling",
                    rope_scaling,
                ]
                with patch.object(sys, "argv", argv):
                    args = module.parse_args()
                self.assertEqual(args.max_model_len, 131072)
                self.assertEqual(args.rope_scaling["rope_type"], "yarn")

    def test_readme_evaluates_the_files_created_by_inference(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        task_workflows = readme.split("## Model compatibility", maxsplit=1)[0]
        generated = re.findall(r"--output-file\s+(\S+)", task_workflows)
        evaluated = re.findall(r"--predictions\s+(\S+)", task_workflows)
        self.assertEqual(generated, evaluated)
        self.assertNotIn("DATASET_DIR/", readme)
        self.assertNotRegex(readme, r"--predictions\s+/path/to/")

    def test_readme_preserves_reviewer_first_workflow(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        headings = (
            "## Web quick start",
            "## Choose how to run GenSyntax",
            "## Local CLI quick start",
            "## Task map",
            "## Complete task workflows",
        )
        positions = [readme.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("four primary tasks are **independent**", readme)
        normalized = re.sub(r"\s+", " ", readme)
        self.assertIn(
            "Both GenSyntax 8B and GenSyntax-Tiny support Tasks 1–4", normalized
        )
        self.assertNotIn("Task 1 only", readme)
        self.assertIn("[`docs/EVALUATION.md`](docs/EVALUATION.md)", readme)
        self.assertTrue((ROOT / "docs" / "EVALUATION.md").is_file())

    def test_web_launcher_uses_checkpoint_context_by_default(self) -> None:
        launcher = (ROOT / "web" / "start.sh").read_text(encoding="utf-8")
        self.assertIn('MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"', launcher)
        self.assertNotIn('MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"', launcher)
        self.assertIn('if [ -n "$MAX_MODEL_LEN" ]', launcher)
        self.assertIn('if [ -n "$ROPE_SCALING" ]', launcher)

    def test_reported_hardware_is_documented(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        training = (ROOT / "training" / "README.md").read_text(encoding="utf-8")
        for gpu_name in ("RTX 4090", "RTX A6000", "A100", "A800"):
            self.assertIn(gpu_name, readme)
        self.assertIn("single NVIDIA GPU", readme)
        self.assertIn("five compute nodes", training)
        self.assertIn("eight NVIDIA H100", training)
        self.assertIn("40 GPUs in total", training)
        self.assertIn("NVLink", training)


if __name__ == "__main__":
    unittest.main()
