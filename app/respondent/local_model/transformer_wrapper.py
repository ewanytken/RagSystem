import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from app.respondent.local_model.abstract_local_model import AbstractLocalRespondent
from app.utils import Utils

class TransformerWrapper(AbstractLocalRespondent):
    def __init__(self, model_name: str = None, use_cpu_only: bool = False, **kwargs) -> None:

        self.config = Utils.get_config_file()
        self.model_name = model_name

        if self.model_name is None and self.config is not None:
            self.model_name = self.config['llm_local']['model']

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        super().__init__(model, tokenizer, use_cpu_only)

    def generate(self, prompt: str, **kwargs) -> str:
        template = {"role": "user", "content": prompt}
        formatted_chat = self.tokenizer.apply_chat_template([template],
                                                            tokenize=False,
                                                            add_generation_prompt=True)
        inputs = self.tokenizer(formatted_chat,
                                return_tensors="pt",
                                add_special_tokens=False).to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.config['llm_local']['max_new_tokens'])

        return self.tokenizer.decode(output[0][inputs['input_ids'].size(1):], skip_special_tokens=True)

    def set_model_name(self, model_name) -> None:
        self.model_name = model_name