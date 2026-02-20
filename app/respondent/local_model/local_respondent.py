from transformers import AutoTokenizer

from app.respondent.local_model.transformer_wrapper import TransformerWrapper
from app.respondent.interface_respondent import Respondent

class LocalRespondent(Respondent):

    def __init__(self, parameters_model=None):

        if parameters_model is None:
            parameters_model = {'name': "Felladrin/TinyMistral-248M-Chat-v3",
                                'max_new_tokens': 555}

        tokenizer = AutoTokenizer.from_pretrained(parameters_model['name'])

        parameter_to_generate = {
            "max_length": 555,
            "do_sample": True,
            "temperature": 0.9,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }

        if parameters_model['max_new_tokens'] is None:
            parameters_model['max_new_tokens'] = 555

        self.model = TransformerWrapper(model_name=parameters_model['name'],
                                        max_new_tokens=parameters_model['max_new_tokens'],
                                        **parameter_to_generate)

    def __call__(self, kwargs):
        return self.model.generate(kwargs)