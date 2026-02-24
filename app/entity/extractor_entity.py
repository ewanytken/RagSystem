
from typing import List, Optional, Dict
from app.entity.abstract_entity import AbstractEntity
from app.logger import LoggerWrapper

logger = LoggerWrapper

"""
Aggregation entities extractors
Input: List[AbstractEntity]  
Output: set of extracted entities Optional[set]
        'score' is 0.9 always
"""

class EntityExtractor:

    def __init__(self, extractors: List[AbstractEntity]):

        self.extractors: Optional[List[AbstractEntity]] = extractors
        self.documents: Optional[List[str]] = []
        self.entities: Optional[set] = set()

        if not self.documents:
            logger(f"Entities extracting...")
            for extractor in self.extractors:
                extractor.set_text_extraction(self.documents)
                extractor.extractor_entity()
                self.entities.add(*extractor.get_extract_entities())

        logger(f"(All methods) Entities extracted: {len(self.get_entity())} ")

    def set_documents(self, documents: List[str]):
        self.documents = documents

    def get_entity(self):
        return self.entities
