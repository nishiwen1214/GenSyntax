"""
Minimal Genome Predictor
------------------------
This script predicts essential protein products in bacterial genomes using a language model (LLM).
The algorithm randomly selects protein products and uses the model to predict if they are essential.
When a non-essential product is found, previous assumptions are re-evaluated to ensure consistency.

Author: SIAT_NLPer
"""

import argparse
import os
import json
import random
from pathlib import Path
from vllm import LLM, SamplingParams


def format_prompt(prompt: str):
    """Format a prompt for LLM chat input."""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    }


def setup_environment(gpu_ids: str, tensor_parallel_size: int):
    """Set CUDA devices and check consistency with tensor parallelism."""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    gpu_ids_list = gpu_ids.split(',')
    if len(gpu_ids_list) != tensor_parallel_size:
        raise ValueError(
            f"Number of GPUs ({len(gpu_ids_list)}) does not match tensor_parallel_size ({tensor_parallel_size})")


def load_json_file(file_path: str):
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_prediction(llm, data_entry, sampling_params, thresholds):
    """
    Run essential protein product prediction on a single genome entry.

    Args:
        llm: vLLM instance
        data_entry: Single JSON entry with protein products
        sampling_params: SamplingParams for LLM
        thresholds: List of thresholds for essential probability
    Returns:
        dict: Updated entry with predicted necessary products
    """
    all_products_list = ['[' + elem[2].strip() + ']' for elem in data_entry['Protein_products']]
    total_indices = list(range(len(all_products_list)))

    for threshold in thresholds:
        candidate_indices = total_indices.copy()
        non_probs_list = []
        essential_probs_list = []

        while True:
            necessary_indices = set()
            pending_indices = candidate_indices.copy()
            all_essential = True

            while pending_indices:
                current_index = random.choice(pending_indices)
                current_product = all_products_list[current_index]

                # Build context excluding the current product
                context_elements = [all_products_list[i] for i in candidate_indices if i != current_index]

                instruction_text = (
                    f"The following list presents the protein products encoded by Mycoplasma mycoides subsp. capri str. GM12 chromosome. "
                    f"Please predict whether the gene corresponding to the protein product $$"
                    f"{current_product.strip('[]')}$$ is essential for this organism? "
                    f"Answer strictly in the following format: non-essential or essential\n"
                    f"{''.join(context_elements)}"
                )

                # Generate prediction
                formatted_instruction = [format_prompt(instruction_text)["messages"]]
                outputs = llm.chat(formatted_instruction, sampling_params)

                non_prob = 0.0
                essential_prob = 0.0

                # Parse token logprobs for normalized probabilities
                for output in outputs:
                    if output.outputs[0].logprobs:
                        for token_logprobs in output.outputs[0].logprobs:
                            sorted_tokens = sorted(token_logprobs.items(),
                                                   key=lambda x: x[1].logprob, reverse=True)
                            for rank, (token_id, logprob_info) in enumerate(sorted_tokens[:10]):
                                token_str = logprob_info.decoded_token
                                prob = float(os.environ.get('DUMMY_PROB', 0)) if 'torch' not in dir() else float(
                                    prob)  # Placeholder
                                if token_str == 'non':
                                    non_prob += prob
                                elif token_str == 'essential':
                                    essential_prob += prob

                # Normalize
                total_prob = non_prob + essential_prob
                normalized_non_prob = non_prob / total_prob if total_prob > 0 else 0
                normalized_essential_prob = essential_prob / total_prob if total_prob > 0 else 0

                if normalized_essential_prob > threshold:
                    necessary_indices.add(current_index)
                    pending_indices.remove(current_index)
                else:
                    candidate_indices.remove(current_index)
                    all_essential = False
                    break  # Restart process

            if all_essential:
                break  # Finished threshold loop

        # Save predicted essential products
        necessary_products_with_org_elements = [data_entry['Protein_products'][i] for i in total_indices if
                                                i in necessary_indices]
        data_entry['predicted_necessary_products'] = necessary_products_with_org_elements

    return data_entry


def main():
    parser = argparse.ArgumentParser(description="Minimal Genome Predictor using LLM")
    parser.add_argument("--model-path", type=str, required=True, help="Path to LLM model checkpoint")
    parser.add_argument("--input-json", type=str, required=True, help="Input JSON file with protein products")
    parser.add_argument("--gpu-ids", type=str, default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--top-logprobs", type=int, default=10, help="Top-N logprobs to return")

    args = parser.parse_args()

    setup_environment(args.gpu_ids, args.tensor_parallel_size)

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 40960}
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        logprobs=args.top_logprobs,
        min_p=0
    )

    data = load_json_file(args.input_json)
    thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]

    for data_index, entry in enumerate(data):
        # Example: Only process a subset if needed
        if data_index in [5]:
            updated_entry = run_prediction(llm, entry, sampling_params, thresholds)

            # Save outputs
            model_name = Path(args.model_path).name
            for threshold in thresholds:
                threshold_str = str(threshold).replace('.', '_')
                output_dir = Path(f'./results/minimal_genome_{model_name}')
                output_dir.mkdir(parents=True, exist_ok=True)

                with open(output_dir / f'entry_{data_index}_threshold_{threshold_str}.json', 'w',
                          encoding='utf-8') as f:
                    json.dump([updated_entry], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
