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

## Installation

Clone this repository and install the required dependencies:
git clone https://github.com/your-repo/GenoVerse.git
cd GenoVerse
pip install -r requirements.txt

## Inference Tasks

We provide five inference tasks that evaluate the genome reasoning abilities of LLMs.
Each task has a dedicated Python script under the tasks/ folder.

Task 1: Plasmid Host Prediction

Goal: Predict the bacterial host of a plasmid given the list of encoded protein products.

Input Example:
