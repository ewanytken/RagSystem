from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from app.respondent.local_model.abstract_local_model import AbstractLocalRespondent

class TransformerWrapper(AbstractLocalRespondent):

    def __init__(self, model_name: str, use_cpu_only: bool = False, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        super().__init__(model, tokenizer, use_cpu_only)


    def generate(self, prompt: str, **kwargs) -> str:
        template = {"role": "user", "content": prompt}
        formatted_chat = self.tokenizer.apply_chat_template([template],
                                                            tokenize=False,
                                                            add_generation_prompt=True)
        inputs = self.tokenizer(formatted_chat,
                                return_tensors="pt",
                                add_special_tokens=False).to(self.device)
        output = self.model.generate(**inputs)

        return self.tokenizer.decode(output[0][inputs['input_ids'].size(1):],
                                     skip_special_tokens=True)