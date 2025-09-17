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
