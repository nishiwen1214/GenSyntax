# GenSyntax training recipes

This directory contains the released recipes for continued pre-training (CPT) and supervised fine-tuning (SFT) of the LLaMA 3.1 8B GenSyntax model. The recipes use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

## Reported training hardware

The manuscript training runs used five compute nodes with eight NVIDIA H100
80 GB GPUs per node, for 40 GPUs in total. GPUs within each node were connected
through NVLink. The released configurations use BF16 and DeepSpeed ZeRO-3 with
CPU parameter and optimizer offload.

| Item | Reported configuration |
|---|---|
| Compute nodes | 5 |
| GPUs per node | 8 |
| Total GPUs | 40 |
| GPU model | NVIDIA H100 80 GB |
| Intra-node GPU interconnect | NVLink |
| Numerical precision | BF16 |
| Distributed memory strategy | DeepSpeed ZeRO-3 with CPU offload |

## Included recipes

The training directory provides:

- an LLaMA-Factory source snapshot reporting version `0.9.6.dev0`;
- one CPT configuration;
- one SFT configuration;
- data schemas for each stage.

Install LLaMA-Factory separately and retain its Apache-2.0 license and
citation. Record the LLaMA-Factory release or commit used for each experiment.
Prepare the CPT and SFT corpora in `training/data/` using the schemas below.

## Layout

```text
training/
├── README.md
├── configs/
│   ├── cpt.yaml
│   ├── sft.yaml
│   └── ds_z3_offload.json
└── data/
    ├── dataset_info.json
    ├── gensyntax_cpt.jsonl     # user-supplied complete CPT corpus
    └── gensyntax_sft.jsonl     # user-supplied complete SFT corpus
```

## Data schemas

CPT uses JSON Lines with one gene-product sequence per record:

```json
{"text": "[chromosomal replication initiator protein DnaA][DNA polymerase III subunit beta]..."}
```

SFT uses Alpaca-style JSON Lines:

```json
{"instruction": "", "input": "Complete task prompt", "output": "[B]"}
```

## Installation

The supplied source snapshot reports Python 3.11 or later and LLaMA-Factory `0.9.6.dev0`. The training configuration additionally requires DeepSpeed, FlashAttention 2, Liger Kernel and Adam-mini. Install a separately cloned, pinned LLaMA-Factory checkout:

```bash
pip install -e .
pip install -r requirements/deepspeed.txt \
  -r requirements/liger-kernel.txt \
  -r requirements/adam-mini.txt
pip install flash-attn --no-build-isolation
```

## Running the two stages

After installation, return to the GenSyntax repository root so that the relative configuration and data paths resolve consistently:

```bash
cd /path/to/GenSyntax
llamafactory-cli train training/configs/cpt.yaml
```

Merge the CPT LoRA adapter into the base model before SFT:

```bash
llamafactory-cli export \
  --model_name_or_path meta-llama/Llama-3.1-8B \
  --adapter_name_or_path saves/llama3.1-8b/lora/gensyntax-cpt \
  --template default \
  --finetuning_type lora \
  --export_dir saves/llama3.1-8b/gensyntax-cpt-merged
```

Set `model_name_or_path` in `configs/sft.yaml` to the merged CPT model path and then run:

```bash
llamafactory-cli train training/configs/sft.yaml
```

For the reported five-node, 40-GPU setup, launch the same command on each node:

```bash
FORCE_TORCHRUN=1 \
NNODES=5 \
NODE_RANK=<0..4> \
MASTER_ADDR=<node-0-address> \
MASTER_PORT=29500 \
llamafactory-cli train training/configs/cpt.yaml
```

Use the SFT configuration path for the second stage. Record the node rank, master address, random seed, CUDA/PyTorch versions, LLaMA-Factory commit, dataset checksums and output checkpoint checksums for every archival run.
