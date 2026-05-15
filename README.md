<img width="1504" height="426" alt="image" src="https://github.com/user-attachments/assets/0655cd14-3abc-4463-935a-7adb489e05e9" />


## Model
We release our fine-tuned genome inference model on Hugging Face:

👉 [GenSyntax on HuggingFace](https://huggingface.co/shuaimin4588/GenSyntax)

You can load it in Python as follows:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "shuaimin4588/GenoVerse"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
```
## Hardware
A single 4090 GPU is sufficient for model deployment and inference.

## Data
We release our data on HuggingFace:

👉 [Datasets on HuggingFace](https://huggingface.co/datasets/ShiwenNi/GenSyntax-data)
(This includes the complete test sets for each task, as well as the training data and test data for cell phenotype prediction.)


## Installation
Clone this repository and install the required dependencies:

git clone https://github.com/your-repo/GenSyntax.git

cd GenoVerse

pip install -r requirements.txt

## Inference Tasks
##### Plasmid Host Prediction

```bash
python Plasmid_host_identification.py \
    --model checkpoint \
    --input-json-paths test_data/gene_task1_test_1000_format.json
```
##### Gene Function Prediction

```bash
python Gene_function_prediction.py \
    --model checkpoint \
    --input-json-paths test_data/gene_task2_test_500_opts.json
```

##### Genome Assembly

```bash
python Genome_assembly.py \
    --model checkpoint \
    --input-json-paths test_data/gene_task3_test_500_contig3_format.json
```
##### Gene Essentiality Prediction

```bash
python Gene_essentiality_prediction.py \
    --model checkpoint \
    --input-json-paths test_data/gene_task4_test_1000_format.json
```
##### Derivation of minimal genomes

```bash
python minimal_genome_inference.py \
    --model checkpoint \
    --input-json-paths test_data/bacteria_chromosomes_9-mini.json
```
