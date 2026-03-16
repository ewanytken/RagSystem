from typing import Optional, List, Dict

from app.logger import LoggerAuxiliary
from app.prompt.abstract_prompt import AbstractPrompt
from app.utils import Utils

logger = LoggerAuxiliary()

"""
Input: str - user_query, str - context, str - entities, document in dir - template 
Output: str - prompt for respondent model 
"""

class PromptObject(AbstractPrompt):

    def __init__(self):
        super().__init__()
        self.config: Optional[Dict] = None
        self.user_query: Optional[str] = "Query don't specify"
        self.chunks: Optional[List] = []
        self.entities: Optional[List] = []
        self.template: Optional[str] = "Empty template"
        self.triplets: Optional[List] = []
        self.final_prompt: Optional[str] = "Empty prompt"

    def make_final_prompt(self):

        entities_context = ""
        if self.entities is not None:
            entities_context = "\nRelated entities from documents. Use it for more deeper answer to query:\n"
            for i, entity in enumerate(self.entities):
                entities_context += f"{i}. Entity: {entity['entity']} is label: {entity['label']} \n"

        logger(f"Entities: {entities_context}")

        triplet_context = ""
        if self.triplets is not None:
            documents_extracted_from_triplet = set()
            triplet_context = "Triplet extracted from documents:\n"
            for i, triplet in enumerate(self.triplets):
                triplet_context += f"{i}. {triplet['subject']} --> [{triplet['predicate']}]--> {triplet['object']} \n"
                documents_extracted_from_triplet.add(triplet['document'])
            triplet_context = triplet_context + "\n".join(documents_extracted_from_triplet)

        logger(f"Triplets: {triplet_context}")

        chunks_context = ""
        if self.chunks is not None:
            chunks_context = "=== RETRIEVED DOCUMENT PASSAGES ===\n"
            for i, chunk in enumerate(self.chunks, 1):
                relevance = chunk.get('score', 'N/A')
                if isinstance(relevance, float):
                    relevance = f"{relevance:.2f}"
                chunks_context += f"\n[Passage {i}] (Relevance: {relevance})\n"
                chunks_context += f"{chunk.get('text', '')}\n"
                chunks_context += "-" * 50

        logger(f"Chunks: {chunks_context}")

        self.template = Utils.load_template(self.config["templates"]["prompt_template"])
        logger(f"Template: {self.template}")

        self.final_prompt = self.template.format(
            context=chunks_context,
            entities_context=entities_context,
            query=self.user_query,
            triplets=triplet_context,
        )
        logger(f"FINAL PROMPT: {self.final_prompt}")

    def get_final_prompt(self) -> str:
        return self.final_prompt

    def set_config(self, config: Dict):
        self.config = config

    def set_chunks(self, chunks: List):
        self.chunks = chunks

    def set_triplet(self, triplet: List):
        self.triplets = triplet

    def set_entities(self, entities: List):
        self.entities = entities

    def set_query(self, query: str):
        self.user_query = query