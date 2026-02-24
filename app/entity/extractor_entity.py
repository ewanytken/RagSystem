
from typing import List, Optional, Dict
from app.entity.abstract_entity import AbstractEntity
from app.graph.graph_object import GraphObject
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
        self.chunked_documents: Optional[List[List[str]]] = []

        self.entities: Optional[set] = set()
        self.graph: Optional[GraphObject] = None

        logger(f"(All methods) Entities extracted: {len(self.get_entity())} ")

    def document_extractor(self):
        if not self.documents:
            logger(f"Entities from documents extracting...")
            for document in self.documents:
                for extractor in self.extractors:
                    extractor.set_text_extraction(document)
                    extractor.extractor_entity()
                    self.entities.add(*extractor.get_extract_entities())

                    self.graph.set_entities(extractor.get_extract_entities())
                    self.graph.add_to_knowledge_graph(document)

    def chunk_extractor(self):
        if not self.chunked_documents:
            logger(f"Entities from chunks extracting...")
            for document in self.chunked_documents: # document not str -> List[str] need changed
                for extractor in self.extractors:
                    extractor.set_text_extraction(document)
                    extractor.extractor_entity()
                    self.entities.add(*extractor.get_extract_entities())

                    self.graph.set_entities(extractor.get_extract_entities())
                    self.graph.add_to_knowledge_graph(document)

    def set_documents(self, documents: List[str]):
        self.documents = documents

    def set_chunked_documents(self, chunked_documents: List[List[str]]):
        self.chunked_documents = chunked_documents

    def set_graph(self, graph: GraphObject):
        self.graph = graph

    def get_entity(self):
        return self.entities
