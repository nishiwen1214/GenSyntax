# GenSyntax 🧬
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
## Data
We release our data on HuggingFace:

👉 [Datasets on HuggingFace](https://huggingface.co/datasets/ShiwenNi/GenSyntax-data)


## Installation
Clone this repository and install the required dependencies:

git clone https://github.com/your-repo/GenSyntax.git

cd GenoVerse

pip install -r requirements.txt

## Inference Tasks
##### Task 1: Plasmid Host Prediction

```bash
python inference_task1.py \
    --model checkpoint \
    --input-json-paths test_data/gene_task1_test_1000_format.json
```
##### Task 2: Unknown Product Prediction

```bash
python inference_task2.py \
    --model checkpoint \
    --input-json-paths test_data/gene_task2_test_500_opts.json
```

##### Task 3: Genome Assembly

```bash
python inference_task3.py \
    --model checkpoint \
    --input-json-paths test_data/gene_task3_test_500_contig3_format.json
```
##### Task 4: Gene Essentiality Prediction

```bash
python inference_task4.py \
    --model checkpoint \
    --input-json-paths test_data/gene_task4_test_1000_format.json
```
##### Task 5: Minimal Genome Prediction

```bash
python minimal_genome_inference.py \
    --model checkpoint \
    --input-json-paths test_data/bacteria_chromosomes_9-mini.json
```
