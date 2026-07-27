# GenSyntax

GenSyntax is a post-annotation, function-level framework for representing prokaryotic replicons as ordered sequences of gene-product descriptors. It is designed for annotated chromosomes, plasmids, draft genomes, metagenome-assembled genomes and annotated contigs. It does **not** operate directly on raw sequencing reads or unannotated nucleotide sequences.

This repository accompanies the manuscript **“Decoding Prokaryotic Whole Genomes with a Product-Contextualized Large Language Model.”**

## Resources

| Component | Current status | Location |
|---|---|---|
| GenSyntax 8B inference checkpoint | Public model checkpoint | [Hugging Face](https://huggingface.co/MoonTideF/Llama-GenSyntax) |
| GenSyntax-Tiny checkpoint | Public, ungated Qwen3 checkpoint | [Hugging Face](https://huggingface.co/ShijianW01/qwen3_0.6b_20250702_data) |
| Evaluation/inference entry points | Available | repository root |
| Task 1 test data | Included locally | `Data/gene_task1_test_1000_format.json` |
| Task 2–4 test data | Public; download separately | [Hugging Face `Data/` directory](https://huggingface.co/datasets/ShiwenNi/GenSyntax-data/tree/main/Data) |
| Web interface | Available; requires a separately running vLLM server | `web/` |
| Continuous pre-training configuration | Available | `training/configs/cpt.yaml` |
| Supervised fine-tuning configuration | Available | `training/configs/sft.yaml` |
| Microbial phenotype prediction workflow and cleaned BacDive tables | Available | [`phenotype_prediction/`](phenotype_prediction/) and [`Data/BacDive/`](https://github.com/nishiwen1214/GenSyntax/tree/main/Data/BacDive) |

## Repository layout

```text
GenSyntax/
├── Data/                              # example inputs and selected phenotype tables
├── training/                          # CPT and SFT recipes and data schemas
├── phenotype_prediction/              # unified ten-phenotype evaluation
├── web/                               # FastAPI interface for an OpenAI-compatible vLLM server
├── Plasmid_host_identification.py     # task 1 batch inference
├── Gene_function_prediction.py        # task 2 batch inference
├── Contig_order_prediction.py         # task 3 batch inference
├── Gene_essentiality_prediction.py    # task 4 batch inference
├── minimal_genome_inference.py         # iterative minimal-genome inference
├── requirements.txt
└── docs/
    └── DATA.md                         # data schemas and provenance
```

## Installation

The released inference scripts were developed with Python 3.10, CUDA 12.6, PyTorch 2.7.1 and vLLM 0.10.1.1. A Linux host with an NVIDIA GPU is recommended.

```bash
git clone https://github.com/nishiwen1214/GenSyntax.git
cd GenSyntax

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the CUDA 12.6 PyTorch wheel explicitly.
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

For other CUDA versions, select the matching PyTorch wheel and record the exact environment used in any reported reproduction.

## Model checkpoints

| Manuscript name | Base architecture | Hugging Face identifier |
|---|---|---|
| GenSyntax | LLaMA 3.1 8B | `MoonTideF/Llama-GenSyntax` |
| GenSyntax-Tiny | Qwen3-0.6B | `ShijianW01/qwen3_0.6b_20250702_data` |

The GenSyntax-Tiny repository contains a merged root checkpoint and
intermediate checkpoints at steps 123,731, 247,462 and 371,193.

## Checkpoint loading

The task scripts load checkpoints through vLLM. Downloading or caching the model in advance makes failures easier to diagnose:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "MoonTideF/Llama-GenSyntax"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
print(model.config.model_type)
```

If this check fails, report the full exception, operating system, Python,
CUDA, PyTorch and Transformers versions.

### GenSyntax-Tiny tokenizer compatibility

GenSyntax-Tiny requires a checkpoint revision whose tokenizer artifacts were
exported with `tokenizer.save_pretrained(...)`. In particular,
`extra_special_tokens` in `tokenizer_config.json` must be a JSON object rather
than a list, and runtime-only metadata such as `backend` and `is_local` should
not be present. Earlier malformed tokenizer metadata can cause Transformers or
vLLM to stop during tokenizer initialization with
`AttributeError: 'list' object has no attribute 'keys'`.

The corrected revision should be verified from an empty Hugging Face cache with
both `AutoTokenizer.from_pretrained(...)` and vLLM before use. Editing a cached
JSON file is not a supported reproduction procedure. The model repository
maintainer should publish and identify the corrected revision in the model
card; until then, use the validated GenSyntax 8B checkpoint in the commands
below.

## Input format

The four task scripts accept a JSON array. Each object must contain an `instruction` string:

```json
[
  {
    "instruction": "This is the list of protein products encoded by a plasmid. Which bacterial host is this plasmid most likely to come from? Answer strictly in the following format: [order, family, genus, species, strain]\n[replication protein][hypothetical protein][transposase]"
  }
]
```

The command-line scripts accept `instruction`, `Input` or `input` as the prompt field. The optional `Output` field is treated as a reference label and is not sent to the model.

## Downloading the evaluation data

The GitHub repository includes the Task 1 test file as a small example. The complete released Task 1–4 evaluation files are hosted in the [`Data/` directory of the GenSyntax-data repository](https://huggingface.co/datasets/ShiwenNi/GenSyntax-data/tree/main/Data):

```text
gene_task1_test_1000_format.json
gene_task2_test_500_opts.json
gene_task2_test_500_opts_8.json
gene_task3_test_500_contig3_format.json
gene_task3_test_500_contig4_format.json
gene_task3_test_500_contig5_format.json
gene_task4_test_1000_format.json
```

Download the dataset and copy the released evaluation files into this
repository's `Data/` directory:

```bash
git clone https://huggingface.co/datasets/ShiwenNi/GenSyntax-data
cp GenSyntax-data/Data/gene_task{1,2,3,4}_*.json Data/
```

All commands below therefore use repository-relative `Data/...` paths. Pin the
dataset revision used for final manuscript results and report that revision
with the code commit.

## Batch inference

All four scripts share the same core arguments:

- `--model-paths`: one or more local checkpoint directories or Hugging Face model identifiers;
- `--input-json-paths`: one or more JSON input files;
- `--output-file`: exact output path when one model and one input are supplied;
- `--output-dir`: automatic-output directory when `--output-file` is omitted;
- `--gpu-ids`: comma-separated visible CUDA device indices;
- `--tensor-parallel-size`: must equal the number of values in `--gpu-ids`.

`--output-file` is restricted to one model and one input so that a prediction
file cannot be overwritten by multiple runs. Generated files contain exactly
one prediction line per input record.

### Task 1: plasmid host identification

Run inference and then evaluate the exact file produced:

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

The evaluator reports class-, order-, family-, genus-, species- and
strain-level accuracy with bootstrap confidence intervals. The taxonomy CSV
must contain `genus,class,order,family` columns.

### Task 2: gene-product disambiguation

Run the four-option experiment:

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

Run the eight-option experiment:

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

Four-option and eight-option records are evaluated separately. The default
statistical settings are 100 sample-level bootstrap replicates, a two-sided
90% percentile confidence interval and random seed 42.

### Task 3: circular contig ordering

Run the three-contig experiment:

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

Run the four-contig experiment:

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

Run the five-contig experiment:

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

Cyclic rotations of the reference order are accepted; reversed orders are not
accepted for manuscript evaluation.

### Task 4: gene essentiality

Run inference and then evaluate the exact file produced:

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

The evaluator reports accuracy, per-class precision/recall/F1 and macro
precision/recall/F1.

## Minimal-genome workflow

`minimal_genome_inference.py` implements the iterative reduction algorithm described in the manuscript. During each randomized traversal, a gene is removed when its normalized essentiality probability does not exceed the selected essentiality confidence threshold. Traversal restarts after every deletion and terminates when all genes in one complete traversal are retained. Higher thresholds therefore permit more aggressive reduction.

The following command evaluates five thresholds using ten independent randomized runs. Replicate `r` uses seed `42 + r - 1`; every threshold–replicate combination is written to a separate JSON file with its deletion trace and retained original indices.

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

Input records require a non-empty `Protein_products` list and should include `Source` (or `Organism`) for organism-specific prompts. Each product may be represented by the manuscript tuple/list schema, in which the third element is the product name, by a plain product-name string, or by an object containing `product`. The probability calculation uses the first-token log probabilities of the two constrained answers, `essential` and `non-essential`; execution stops with an explicit error if both alternatives are not returned. For archival reproduction, record the exact model revision, input checksum, vLLM version, thresholds and base seed.

## Continued pre-training and supervised fine-tuning

The released LLaMA 3.1 8B training recipes are under [`training/`](training/). They describe sequential LoRA-based continued pre-training and supervised fine-tuning with a 128,000-token cutoff, DeepSpeed ZeRO-3 CPU offload, BF16, FlashAttention 2, Liger Kernel and Adam-mini.

See [`training/README.md`](training/README.md) for the corpus schemas,
installation instructions, configuration details and multi-node launch
command.

## Microbial phenotype prediction

The ten microbial phenotype experiments reported in the manuscript are
implemented by a single configurable workflow under
[`phenotype_prediction/`](phenotype_prediction/). It evaluates five
classifiers on genome embeddings using identical stratified 80:20 splits and
three random seeds, and reports accuracy and weighted F1 as mean ± s.d. The
workflow also records class filtering, species matching, split assignments and
sample-level predictions. The ten cleaned phenotype tables are distributed in
the repository's [`Data/BacDive/`](https://github.com/nishiwen1214/GenSyntax/tree/main/Data/BacDive)
directory.

The phenotype workflow requires a model-specific, precomputed genome-embedding
JSON supplied through `--embeddings`. The BacDive phenotype tables do not
contain these vectors and cannot be used as a substitute. Reviewers reproducing
the reported phenotype results should obtain the exact embedding file and its
checksum from the authors or the associated review data package.

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
ten task definitions, exact discretization thresholds, input schema, model
settings and output inventory.

## Web interface

The public GenSyntax demonstration interface is available at
[http://111.2.199.31:2103/](http://111.2.199.31:2103/). It provides single-
record and batch prediction for the four primary tasks, together with
gene-level essentiality screening from a genome JSON file. The service is
intended for interactive exploration; manuscript-scale experiments should use
the versioned command-line workflows so that model revisions, parameters,
random seeds and outputs can be recorded.

### Quick web tutorial

#### Single prediction

1. Open the web interface and select **Single Prediction**.
2. Choose a task from the **Task** menu.
3. Complete the task-specific fields:
   - **Plasmid Host Identification:** paste the ordered products as
     `[product 1][product 2]...`;
   - **Gene Function Prediction:** enter the organism, replicon type, at least
     options A and B, and a gene list containing exactly one
     `[unknown product]`;
   - **Contig Order Prediction:** enter the number of contigs, organism name
     and numbered contig product lists;
   - **Gene Essentiality Prediction:** enter the organism, target product and
     the ordered genomic product context;
   - **Minimal Genome Inference:** upload a `.json` file using the schema shown
     on the page.
4. Select **Run Prediction**. For the first four tasks, the response appears
   under **Prediction Result** and can be copied with the copy button. A
   minimal-genome file is processed as a batch job; the interface switches to
   the batch progress and results view automatically.

For the first four tasks, preserve the order of the protein products and use
the bracketed product representation shown in the input fields. Product names
should come from a consistent annotation workflow, such as NCBI PGAP, because
differences in annotation terminology can affect predictions.

#### Batch prediction

1. Select **Batch Prediction**.
2. Choose the corresponding task.
3. Upload a UTF-8 JSON file containing an array of records:

```json
[
  {
    "Input": "complete task prompt"
  }
]
```

The keys `input` and `instruction` are also accepted. An optional `Output`
field may contain a reference answer and is not sent to the model. Select
**Run Batch Prediction**, wait for the progress indicator to finish, inspect
any item-level errors, and use **Download** to save the predictions.

For the web **Minimal Genome Inference** option, upload a JSON object or array
with `Source` and `Protein_products` fields:

```json
[
  {
    "Source": "synthetic bacterium JCVI-Syn3A",
    "Protein_products": [
      ["CDS", "dnaA", "chromosomal replication initiator protein", "[0:1353](+)"],
      ["CDS", "dnaN", "DNA polymerase III subunit beta", "[1510:2638](+)"]
    ]
  }
]
```

The current web endpoint converts this file into one gene-essentiality query
per protein product. It does **not** execute the randomized iterative reduction
algorithm or produce the threshold-specific minimal genomes reported in the
manuscript. Use [`minimal_genome_inference.py`](minimal_genome_inference.py)
for the complete IRA workflow.

If a request fails, first confirm that all required fields are filled and that
the uploaded file is valid UTF-8 JSON. A connection or inference error may
also indicate that the demonstration server or its vLLM backend is unavailable.
For reproducible analysis, retain the original input file and use the
command-line release rather than relying only on copied web output.

### Local deployment

The web application expects an OpenAI-compatible vLLM server. From `web/`:

```bash
pip install -r requirements.txt
MODEL_PATH=/absolute/path/to/model CUDA_VISIBLE_DEVICES=0,1 bash start.sh
```

The launcher defaults to two GPUs and a maximum model length of 131,072
tokens. Adjust these settings only when supported by the selected checkpoint
and hardware. The startup script checks backend readiness before launching the
web application and terminates both processes when either service exits.

## Reproducing the manuscript

The repository provides the GenSyntax inference and evaluation entry points,
LLaMA 3.1 8B CPT/SFT configurations, GenSyntax-Tiny weights, microbial
phenotype workflows and the associated data documentation.

See [`docs/DATA.md`](docs/DATA.md) for data schemas and provenance requirements.

## Data and code availability

The public code, model checkpoints and datasets are linked above. Versioned
release metadata, checksums and environment information should accompany the
final publication archive.

## License

Code in this repository is released under the [MIT License](LICENSE). Dataset and model licenses must be documented separately in their respective repository cards, including any upstream RefSeq/PGAP, BacDive and third-party benchmark terms.
