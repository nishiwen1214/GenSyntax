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
    return [q['instruction'] for q in questions]


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
        sampling_params: SamplingParams
):
    """Run batch inference for a model on multiple input JSON files."""
    model_name = Path(model_path).name
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 40960}
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

        output_file = Path(output_dir) / f"{Path(json_path).stem}_{model_name}.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_predictions))

        print(f"[INFO] Saved {len(all_predictions)} predictions to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference using vLLM LLaMA models.")

    # Model and input/output
    parser.add_argument("--model-paths", type=str, nargs='+', required=True,
                        help="Paths to local LLaMA models. Example: "
                             "'your_path/.../checkpoint'")
    parser.add_argument("--input-json-paths", type=str, nargs='+', required=True,
                        help="Paths to input JSON files. Example: "
                             "'your_path/gene_task2_test.json'")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Directory to save generated outputs. Default: ./outputs")

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature for LLM.")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of generated tokens.")
    parser.add_argument("--top-logprobs", type=int, default=1, help="Number of top logprobs to output.")

    # Hardware
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="Number of GPUs to use.")
    parser.add_argument("--gpu-ids", type=str, default="0,1",
                        help="Comma-separated GPU IDs to use, e.g., '0,1'. Default: 0,1")

    return parser.parse_args()


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
            tensor_parallel_size=args.tensor_parallel_size,
            sampling_params=sampling_params
        )
