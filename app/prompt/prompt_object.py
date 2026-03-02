from typing import Optional, List, Dict

from app.logger import LoggerWrapper
from app.utils import Utils

logger = LoggerWrapper()

"""
Input: str - user_query, str - context, str - entities, document in dir - template 
Output: str - prompt for respondent model 
"""

class PromptObject:

    def __init__(self):
        self.config: Optional[Dict] = None
        self.user_query: Optional[str] = "Query don't specify"
        self.context: Optional[str] = "Context don't specify"
        self.entities: Optional[List] = []
        self.template: Optional[str] = "Empty template"
        self.triplet: Optional[List] = []

        self.final_prompt: Optional[str] = "Empty prompt"

    def make_final_prompt(self):

        entities_context = ""
        if self.entities is not None:
            entities_context = "\nRelated entities from knowledge graph. Use it for more deeper answer to query:\n"
            for i, entity in enumerate(self.entities):
                entities_context += f"{i}. {entity['label']}: {entity['entity']} (relation: {entity['score']:.2f})\n"

        if self.triplet is not None:
            pass

        self.template = Utils.load_template(self.config["templates"]["prompt_template"])

        self.final_prompt = self.template.format(
            context=self.context,
            entities_context=entities_context,
            query=self.user_query,
        )
        logger(f"CONTENT OF FINAL PROMPT: {self.final_prompt}")

    def get_final_prompt(self) -> str:
        return self.final_prompt

    def set_config(self, config: Dict):
        self.config = config

    def set_context(self, context: str):
        self.context = context

    def set_triplet(self, triplet: List):
        self.triplet = triplet

    def set_entities(self, entities: List):
        self.entities = entities

    def set_query(self, query: str):
        self.user_query = query