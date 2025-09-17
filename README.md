# GenoVerse
## Model
We release our fine-tuned genome inference model on Hugging Face:

👉 [GenoVerse on HuggingFace](https://huggingface.co/shuaimin4588/GenoVerse)

You can load it in Python as follows:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "shuaimin4588/GenoVerse"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
```
## Installation
Clone this repository and install the required dependencies:

git clone https://github.com/your-repo/GenoVerse.git

cd GenoVerse

pip install -r requirements.txt

## Inference Tasks
Task 1: Plasmid Host Prediction

python inference_plasmid_host_predict_task1.py --model your_path/checkpoint --input-json-paths your_path/test_data/gene_task1_test_1000_format_alpaca.json
