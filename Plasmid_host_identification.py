"""
LLM Batch Inference Script (Open-source Friendly)
-------------------------------------------------
This script demonstrates batch inference using a local LLaMA-based LLM with vLLM,
applying sampling parameters and saving the generated outputs.

Author: SIAT_NLPer
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict
import json

from vllm import LLM, SamplingParams


def load_questions(path: str) -> List[str]:
    """Load JSON questions and extract prompts."""
    with open(path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    prompts = []
    for index, question in enumerate(questions):
        prompt = question.get('instruction') or question.get('Input') or question.get('input')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Record {index} in {path} has no non-empty instruction/Input field")
        prompts.append(prompt.strip())
    return prompts


def format_prompt(prompt: str) -> Dict:
    """Format a prompt for LLM input in chat format."""
    return {
        "messages": [
            {"role": "user", "content": prompt},
        ]
    }


def setup_environment(gpu_ids: str, tensor_parallel_size: int):
    """Set CUDA visible devices and check consistency with tensor parallel size."""
    if gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        gpu_ids_list = gpu_ids.split(',')
        if len(gpu_ids_list) != tensor_parallel_size:
            raise ValueError(
                f"Number of GPUs ({len(gpu_ids_list)}) does not match tensor_parallel_size ({tensor_parallel_size})."
            )


def run_inference(
        model_path: str,
        input_json_paths: List[str],
        output_dir: str,
        tensor_parallel_size: int,
        sampling_params: SamplingParams,
        output_file: str | None = None,
):
    """Run batch inference for a model on multiple input JSON files."""
    model_name = Path(model_path).name
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size
    )

    for json_path in input_json_paths:
        prompts = load_questions(json_path)
        formatted_prompts = [format_prompt(p)["messages"] for p in prompts]

        print(f"[INFO] Processing {len(prompts)} prompts from {json_path}...")

        outputs = llm.chat(formatted_prompts, sampling_params)
        all_predictions = []

        for output in outputs:
            text = output.outputs[0].text
            text_clean = " ".join([line.strip() for line in text.split('\n')])
            all_predictions.append(text_clean)

        destination = (
            Path(output_file)
            if output_file
            else Path(output_dir) / f"{Path(json_path).stem}_{model_name}.txt"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_predictions))

        print(f"[INFO] Saved {len(all_predictions)} predictions to {destination}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference using vLLM LLaMA models.")

    # Model and input/output
    parser.add_argument("--model-paths", type=str, nargs='+', required=True,
                        help="Paths to local LLaMA models. Example: "
                             "'your_path/checkpoint-xxx'")
    parser.add_argument("--input-json-paths", type=str, nargs='+', required=True,
                        help="Paths to input JSON files. Example: "
                             "'your_path/gene_task1_test.json'")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Directory to save generated outputs. Default: ./outputs")
    parser.add_argument(
        "--output-file",
        type=str,
        help=(
            "Exact prediction file path. Valid only with one model and one "
            "input file. When omitted, the filename is generated under --output-dir."
        ),
    )

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature for LLM.")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of generated tokens.")
    parser.add_argument("--top-logprobs", type=int, default=1, help="Number of top logprobs to output.")

    # Hardware
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="Number of GPUs to use.")
    parser.add_argument("--gpu-ids", type=str, default="0,1",
                        help="Comma-separated GPU IDs to use, e.g., '0,1'. Default: 0,1")

    args = parser.parse_args()
    if args.output_file and (
        len(args.model_paths) != 1 or len(args.input_json_paths) != 1
    ):
        parser.error(
            "--output-file requires exactly one --model-paths value and one "
            "--input-json-paths value."
        )
    return args


if __name__ == "__main__":
    args = parse_args()
    setup_environment(args.gpu_ids, args.tensor_parallel_size)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        logprobs=args.top_logprobs,
        min_p=0
    )

    for model_path in args.model_paths:
        run_inference(
            model_path=model_path,
            input_json_paths=args.input_json_paths,
            output_dir=args.output_dir,
            output_file=args.output_file,
            tensor_parallel_size=args.tensor_parallel_size,
            sampling_params=sampling_params
        )
