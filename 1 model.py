from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re

class TextGenerator:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        if torch.cuda.is_available():
            self.model.to('cuda')
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        input_len = len(input_ids[0])
        sequences = self.pipeline([prompt], max_length=input_len+40, do_sample=True, top_k=10, num_return_sequences=1)
        for seq in sequences:
            generated_text = seq['generated_text']
            # Extract text after the "classification label" indicator
            match = re.search(r'classification label:(.*)$', generated_text, re.MULTILINE)
            classification_text = match.group(1).strip() if match else 'No classification label found'
            return classification_text
