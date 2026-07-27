# GenSyntax

<img width="1888" height="542" alt="GenSyntax" src="https://github.com/user-attachments/assets/4737952d-dbcc-4941-961b-53619db945bd" />

GenSyntax is a post-annotation, function-level framework for representing
prokaryotic replicons as ordered sequences of gene-product descriptors. It is
designed for annotated chromosomes, plasmids, draft genomes,
metagenome-assembled genomes and annotated contigs. It does **not** operate
directly on raw sequencing reads or unannotated nucleotide sequences.

This repository accompanies the manuscript **“Decoding Prokaryotic Whole
Genomes with a Product-Contextualized Large Language Model.”**

## Web quick start

The fastest way to try GenSyntax is the public interface:
[http://111.2.199.31:2103/](http://111.2.199.31:2103/).

1. Open the interface and select **Single Prediction**.
2. Choose one task.
3. Enter the requested organism and ordered gene-product information using the
   examples shown on the page.
4. Select **Run Prediction** and inspect or copy the result.

For multiple records, select **Batch Prediction**, upload a UTF-8 JSON array
with an `Input`, `input` or `instruction` field in each record, run the job and
download the predictions. The public Web service has been tested as the
recommended five-minute introduction to the model.

## Choose how to run GenSyntax

| Route | Best for | Start here |
|---|---|---|
| Public Web | Immediate interactive testing without installation | [Open the interface](http://111.2.199.31:2103/) |
| Local command line | Versioned inference and quantitative evaluation | [Local CLI quick start](#local-cli-quick-start) |
| Local Web | Private interactive use with a local vLLM backend | [Local Web deployment](#local-web-deployment) |
| Training | Continued pre-training or supervised fine-tuning | [`training/README.md`](training/README.md) |

The four primary tasks are **independent**. Run only the task you need; no task
is a prerequisite for another. Both GenSyntax 8B and GenSyntax-Tiny support
Tasks 1–4. For long Tiny inputs, enable the documented 128K YaRN configuration.

## Local CLI quick start

The released scripts were developed with Python 3.10, CUDA 12.6, PyTorch 2.7.1
and vLLM 0.10.1.1. A Linux host with an NVIDIA GPU is recommended.

```bash
git clone https://github.com/nishiwen1214/GenSyntax.git
cd GenSyntax

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Task 1 data and its taxonomy table are included, so this is the smallest local
end-to-end example:

```bash
python Plasmid_host_identification.py \
  --model-paths MoonTideF/Llama-GenSyntax \
  --input-json-paths Data/gene_task1_test_1000_format.json \
  --output-file outputs/task1/predictions.txt \
  --gpu-ids 0 \
  --tensor-parallel-size 1

python evaluation/task1_plasmid_host_accuracy.py \
  --references Data/gene_task1_test_1000_format.json \
  --predictions outputs/task1/predictions.txt \
  --taxonomy Data/genus_taxonomy.csv \
  --output-csv outputs/task1/accuracy.csv \
  --output-json outputs/task1/accuracy.json
```

## Task map

Choose any row independently. Task 2 and Task 3 have separate released test
sets for each experimental setting.

| Task | Inference script | Test data | Prediction file used below | Evaluation script |
|---|---|---|---|---|
| Task 1: plasmid host identification | `Plasmid_host_identification.py` | `gene_task1_test_1000_format.json` | `outputs/task1/predictions.txt` | `evaluation/task1_plasmid_host_accuracy.py` |
| Task 2: gene-product disambiguation | `Gene_function_prediction.py` | `gene_task2_test_500_opts.json` (4 options)<br>`gene_task2_test_500_opts_8.json` (8 options) | `outputs/task2/opt4_predictions.txt`<br>`outputs/task2/opt8_predictions.txt` | `evaluation/task2_gene_product_accuracy.py` |
| Task 3: circular contig ordering | `Contig_order_prediction.py` | `gene_task3_test_500_contig3_format.json`<br>`gene_task3_test_500_contig4_format.json`<br>`gene_task3_test_500_contig5_format.json` | `outputs/task3/contig3_predictions.txt`<br>`outputs/task3/contig4_predictions.txt`<br>`outputs/task3/contig5_predictions.txt` | `evaluation/task3_contig_order_accuracy.py` |
| Task 4: gene essentiality | `Gene_essentiality_prediction.py` | `gene_task4_test_1000_format.json` | `outputs/task4/predictions.txt` | `evaluation/task4_gene_essentiality_metrics.py` |

## Evaluation data

The complete Task 1–4 test sets are available from the
[GenSyntax-data repository](https://huggingface.co/datasets/ShiwenNi/GenSyntax-data/tree/main/Data).
Download only the files required for the task you selected:

```bash
git clone https://huggingface.co/datasets/ShiwenNi/GenSyntax-data
cp GenSyntax-data/Data/gene_task{1,2,3,4}_*.json Data/
```

All commands below use repository-relative `Data/...` paths. The inference
scripts accept `instruction`, `Input` or `input` as the prompt field. An
optional `Output` field is a reference label and is not sent to the model.

## Complete task workflows

All inference commands below use one GPU. `--output-file` names the exact file
consumed by the following evaluation command and is valid when one model and
one input file are supplied. Each prediction file contains one line per input
record.

Evaluation definitions, bootstrap settings, denominator rules and invalid
prediction handling are documented separately in
[`docs/EVALUATION.md`](docs/EVALUATION.md).

### Task 1: plasmid host identification

The complete Task 1 command is the [local quick-start example](#local-cli-quick-start)
above. It reports accuracy from class through strain.

### Task 2: gene-product disambiguation

Four-option evaluation:

```bash
python Gene_function_prediction.py \
  --model-paths MoonTideF/Llama-GenSyntax \
  --input-json-paths Data/gene_task2_test_500_opts.json \
  --output-file outputs/task2/opt4_predictions.txt \
  --gpu-ids 0 \
  --tensor-parallel-size 1

python evaluation/task2_gene_product_accuracy.py \
  --references Data/gene_task2_test_500_opts.json \
  --predictions outputs/task2/opt4_predictions.txt \
  --num-options 4 \
  --output-csv outputs/task2/opt4_accuracy.csv \
  --output-json outputs/task2/opt4_accuracy.json
```

Eight-option evaluation:

```bash
python Gene_function_prediction.py \
  --model-paths MoonTideF/Llama-GenSyntax \
  --input-json-paths Data/gene_task2_test_500_opts_8.json \
  --output-file outputs/task2/opt8_predictions.txt \
  --gpu-ids 0 \
  --tensor-parallel-size 1

python evaluation/task2_gene_product_accuracy.py \
  --references Data/gene_task2_test_500_opts_8.json \
  --predictions outputs/task2/opt8_predictions.txt \
  --num-options 8 \
  --output-csv outputs/task2/opt8_accuracy.csv \
  --output-json outputs/task2/opt8_accuracy.json
```

The official Task 2 files expose detectable option markers, so
`--num-options auto` is also supported. The explicit values above make each
published setting immediately visible.

### Task 3: circular contig ordering

Three-contig evaluation:

```bash
python Contig_order_prediction.py \
  --model-paths MoonTideF/Llama-GenSyntax \
  --input-json-paths Data/gene_task3_test_500_contig3_format.json \
  --output-file outputs/task3/contig3_predictions.txt \
  --gpu-ids 0 \
  --tensor-parallel-size 1

python evaluation/task3_contig_order_accuracy.py \
  --references Data/gene_task3_test_500_contig3_format.json \
  --predictions outputs/task3/contig3_predictions.txt \
  --num-contigs 3 \
  --output-csv outputs/task3/contig3_accuracy.csv \
  --output-json outputs/task3/contig3_accuracy.json \
  --errors-csv outputs/task3/contig3_errors.csv
```

Four-contig evaluation:

```bash
python Contig_order_prediction.py \
  --model-paths MoonTideF/Llama-GenSyntax \
  --input-json-paths Data/gene_task3_test_500_contig4_format.json \
  --output-file outputs/task3/contig4_predictions.txt \
  --gpu-ids 0 \
  --tensor-parallel-size 1

python evaluation/task3_contig_order_accuracy.py \
  --references Data/gene_task3_test_500_contig4_format.json \
  --predictions outputs/task3/contig4_predictions.txt \
  --num-contigs 4 \
  --output-csv outputs/task3/contig4_accuracy.csv \
  --output-json outputs/task3/contig4_accuracy.json \
  --errors-csv outputs/task3/contig4_errors.csv
```

Five-contig evaluation:

```bash
python Contig_order_prediction.py \
  --model-paths MoonTideF/Llama-GenSyntax \
  --input-json-paths Data/gene_task3_test_500_contig5_format.json \
  --output-file outputs/task3/contig5_predictions.txt \
  --gpu-ids 0 \
  --tensor-parallel-size 1

python evaluation/task3_contig_order_accuracy.py \
  --references Data/gene_task3_test_500_contig5_format.json \
  --predictions outputs/task3/contig5_predictions.txt \
  --num-contigs 5 \
  --output-csv outputs/task3/contig5_accuracy.csv \
  --output-json outputs/task3/contig5_accuracy.json \
  --errors-csv outputs/task3/contig5_errors.csv
```

### Task 4: gene essentiality

```bash
python Gene_essentiality_prediction.py \
  --model-paths MoonTideF/Llama-GenSyntax \
  --input-json-paths Data/gene_task4_test_1000_format.json \
  --output-file outputs/task4/predictions.txt \
  --gpu-ids 0 \
  --tensor-parallel-size 1

python evaluation/task4_gene_essentiality_metrics.py \
  --references Data/gene_task4_test_1000_format.json \
  --predictions outputs/task4/predictions.txt \
  --output-csv outputs/task4/metrics.csv \
  --output-json outputs/task4/metrics.json \
  --errors-csv outputs/task4/errors.csv
```

Task 4 reports accuracy, class-specific precision/recall/F1 and macro
precision/recall/F1.

## Model compatibility

| Model | Supported released tasks | Context configuration |
|---|---|---|
| GenSyntax 8B: `MoonTideF/Llama-GenSyntax` | Tasks 1–4 | Use the checkpoint configuration |
| GenSyntax-Tiny: `ShijianW01/qwen3_0.6b_20250702_data` | Tasks 1–4 | 40,960 tokens by default; extendable to 131,072 with YaRN |

The Tiny checkpoint can run the same four tasks as the 8B checkpoint. Its
default Qwen3 configuration uses 40,960 tokens; this is a default runtime
setting, not a hard capability limit. For inputs requiring up to 128K tokens,
pass YaRN explicitly:

```bash
python Gene_essentiality_prediction.py \
  --model-paths ShijianW01/qwen3_0.6b_20250702_data \
  --input-json-paths Data/gene_task4_test_1000_format.json \
  --output-file outputs/task4/tiny_predictions.txt \
  --max-model-len 131072 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --gpu-ids 0 \
  --tensor-parallel-size 1
```

The same two context arguments can be used with any Task 1–4 inference script.
Static YaRN is intended for long inputs; omit it when the default context is
sufficient.

GenSyntax-Tiny requires a revision whose tokenizer artifacts were exported
with `tokenizer.save_pretrained(...)`. `extra_special_tokens` in
`tokenizer_config.json` must be a JSON object, and runtime-only metadata such as
`backend` and `is_local` should not be included. Validate the corrected
revision from an empty Hugging Face cache with both
`AutoTokenizer.from_pretrained(...)` and vLLM. Editing a cached JSON file is
not a supported procedure.

## Local Web deployment

The local Web application uses an OpenAI-compatible vLLM server:

1. Change to the Web directory: `cd web`.
2. Install its dependencies: `pip install -r requirements.txt`.
3. Start the model and Web server:

```bash
MODEL_PATH=MoonTideF/Llama-GenSyntax \
CUDA_VISIBLE_DEVICES=0,1 \
TENSOR_PARALLEL_SIZE=2 \
bash start.sh
```

4. Open `http://localhost:8101/`.

By default, `start.sh` lets vLLM read the model's context length from the
checkpoint. Set `MAX_MODEL_LEN` only when an explicit lower limit is needed.
To run Tiny with a 128K context after its tokenizer revision is corrected:

```bash
MODEL_PATH=ShijianW01/qwen3_0.6b_20250702_data \
CUDA_VISIBLE_DEVICES=0 \
TENSOR_PARALLEL_SIZE=1 \
MAX_MODEL_LEN=131072 \
ROPE_SCALING='{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
bash start.sh
```

## Additional workflows

### Minimal-genome inference

`minimal_genome_inference.py` implements the iterative reduction workflow:

```bash
python minimal_genome_inference.py \
  --model-path MoonTideF/Llama-GenSyntax \
  --input-json /path/to/genome.json \
  --output-dir outputs/minimal_genome \
  --thresholds 0.5 0.4 0.3 0.2 0.05 \
  --replicates 10 \
  --seed 42 \
  --gpu-ids 0 \
  --tensor-parallel-size 1
```

### Microbial phenotype prediction

The phenotype workflow requires the model-specific precomputed genome-embedding
JSON used for the reported experiment. The BacDive tables alone do not contain
these vectors. Reviewers should use the exact embedding file and checksum
provided in the associated review data package.

```bash
python phenotype_prediction/run_phenotype_prediction.py \
  --embeddings review_data/gensyntax_genome_embeddings.json \
  --embedding-name GenSyntax \
  --data-dir Data/BacDive \
  --phenotypes all \
  --seeds 42 43 44 \
  --output-dir outputs/phenotype_prediction/gensyntax \
  --plot
```

See [`phenotype_prediction/README.md`](phenotype_prediction/README.md) for the
task definitions and output inventory.

### Training

Continued pre-training and supervised fine-tuning recipes are provided under
[`training/`](training/). See [`training/README.md`](training/README.md) for
data schemas, configuration details and the multi-node launch command.

## Documentation and resources

- Evaluation definitions: [`docs/EVALUATION.md`](docs/EVALUATION.md)
- Data files and provenance: [`docs/DATA.md`](docs/DATA.md)
- GenSyntax 8B checkpoint:
  [MoonTideF/Llama-GenSyntax](https://huggingface.co/MoonTideF/Llama-GenSyntax)
- GenSyntax-Tiny checkpoint:
  [ShijianW01/qwen3_0.6b_20250702_data](https://huggingface.co/ShijianW01/qwen3_0.6b_20250702_data)
- Released evaluation data:
  [ShiwenNi/GenSyntax-data](https://huggingface.co/datasets/ShiwenNi/GenSyntax-data/tree/main/Data)

## Repository layout

```text
GenSyntax/
├── Data/                              # evaluation and phenotype data
├── evaluation/                        # deterministic Task 1–4 evaluators
├── phenotype_prediction/              # ten-phenotype evaluation
├── training/                          # CPT and SFT recipes
├── web/                               # FastAPI Web interface
├── Plasmid_host_identification.py     # Task 1 inference
├── Gene_function_prediction.py        # Task 2 inference
├── Contig_order_prediction.py         # Task 3 inference
├── Gene_essentiality_prediction.py    # Task 4 inference
└── minimal_genome_inference.py
```

## License

Code in this repository is released under the [MIT License](LICENSE). Dataset
and model licenses are documented separately in their respective repository
cards.
