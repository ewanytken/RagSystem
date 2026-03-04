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
        self.chunk: Optional[List] = []
        self.entities: Optional[List] = []
        self.template: Optional[str] = "Empty template"
        self.triplets: Optional[List] = []
        self.triplet_context: Optional[str] = "Context don't specify"
        self.final_prompt: Optional[str] = "Empty prompt"

    def make_final_prompt(self):

        entities_context = ""
        if self.entities is not None:
            entities_context = "\nRelated entities from documents. Use it for more deeper answer to query:\n"
            for i, entity in enumerate(self.entities):
                entities_context += f"{i}. Entity: {entity['entity']} is label: {entity['label']} \n"

        if self.triplets is not None:
            documents_extracted_from_triplet = set()
            triplet_context = "Triplet extracted from documents:\n"
            for i, triplet in enumerate(self.triplets):
                triplet_context += f"{i}. {triplet['subject']} --> [{triplet['predicate']}]--> {triplet['object']} \n"
                documents_extracted_from_triplet.add(triplet['document'])
            self.triplet_context = triplet_context + "\n".join(documents_extracted_from_triplet)

        context = ""
        if self.chunk is not None:
            context = "=== RETRIEVED DOCUMENT PASSAGES ===\n"
            for i, chunk in enumerate(self.chunk, 1):
                relevance = chunk.get('score', 'N/A')
                if isinstance(relevance, float):
                    relevance = f"{relevance:.2f}"
                context += f"\n[Passage {i}] (Relevance: {relevance})\n"
                context += f"{chunk.get('text', '')}\n"
                context += "-" * 50

        self.template = Utils.load_template(self.config["templates"]["prompt_template"])
        #TODO chunk processing to context
        self.final_prompt = self.template.format(
            context=context,
            entities_context=entities_context,
            query=self.user_query,
        )
        logger(f"CONTENT OF FINAL PROMPT: {self.final_prompt}")

    def get_final_prompt(self) -> str:
        return self.final_prompt

    def set_config(self, config: Dict):
        self.config = config

    def set_context(self, chunk: List):
        self.chunk = chunk

    def set_triplet(self, triplet: List):
        self.triplets = triplet

    def set_entities(self, entities: List):
        self.entities = entities

    def set_query(self, query: str):
        self.user_query = query