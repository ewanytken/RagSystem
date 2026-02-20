from pathlib import Path
from typing import Optional, Dict

from app.logger import LoggerWrapper

logger = LoggerWrapper

"""
Input: str - user_query, str - context, str - entities, document in dir - template 
Output: str - prompt for respondent model 
"""

class PromptObject:

    def __init__(self):

        self.user_query: Optional[str] = "Query don't specify"
        self.context: Optional[str] = "Context don't specify"
        self.entities: Optional[str] = "Entities doesn't specify"
        self.template: Optional[str] = "Template don't load"

        self.prompt: Optional[str] = "Empty prompt"

    def set_prompt(self):

        #TODO bad implementation
        entities_context = ""
        if self.entities is not None:
            entities_context = "\nСвязанные сущности из графа знаний:\n"
            for i, entity in enumerate(self.entities):
                entities_context += f"{i}. {entity['label']}: {entity['entity']} (связь: {entity['score']:.2f})\n"

        self.prompt = self.template.format(
            context=self.context,
            entities_context=entities_context,
            query=self.user_query,
        )
        logger(f"CONTENT OF PROMPT: {self.prompt}")

    def get_prompt(self):
        return self.prompt

class PromptObjectBuilder:

    prompt_object: Optional[PromptObject] = None

    def __init__(self):
        self.prompt_object = PromptObject()

    def set_query(self, query: str):
       self.prompt_object.user_query = query
       return self

    def set_context(self, context: str):
        self.prompt_object.context = context
        return self

    def set_entities(self, entities: str):
        self.prompt_object.entities = entities
        return self

    def set_path_to_template(self, path_to_template: str):
        self.prompt_object.template = PromptObjectBuilder.load_prompt_template(path_to_template)
        return self

    def build(self):
        return self.prompt_object

    @staticmethod
    def load_prompt_template(path_to_template: str) -> str:
        path = Path(__file__).parent.parent / path_to_template
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            logger(f"prompt_template File not Found (without format) 41 {path}.")
        except Exception as e:
            logger(f"An error occurred: {e}")