
from typing import List, Optional, Set

from app.entity.abstract_entity import AbstractEntity
from app.graph.graph_entity import GraphEntity
from app.logger import LoggerWrapper

logger = LoggerWrapper()

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
        self.graph: Optional[GraphEntity] = None

        logger(f"Number of Entity Extractor downloaded: {len(self.extractors)} ")

    def document_extractor(self) -> None:
        if not self.documents:
            logger(f"Entities from documents extracting...")
            for document in self.documents:
                for extractor in self.extractors:
                    extractor.set_text_extraction(document)
                    extractor.extractor_entity()
                    self.entities.add(*extractor.get_extract_entities())

                    if self.graph is not None:
                        self.graph.set_entities(extractor.get_extract_entities())
                        self.graph.add_to_knowledge_graph(document)

        logger(f"Graph-entity status: {self.graph.get_knowledge_graph_stats()}")

    def set_documents(self, documents: List[str]) -> None:
        self.documents = documents

    def set_graph(self, graph: GraphEntity) -> None:
        self.graph = graph

    def get_entity(self) -> Set:
        return self.entities
