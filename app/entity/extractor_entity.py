
from typing import List, Optional, Set

from app.entity.abstract_entity import AbstractEntity
from app.graph.graph_entity import GraphEntity
from app.graph.triplet_extractor import TripletExtractor
from app.logger import LoggerWrapper

logger = LoggerWrapper()

"""
Aggregation entities extractors
Input: List[AbstractEntity]  
Output: set of extracted entities Optional[set]
        'score' is 0.9 always
"""

class EntityExtractor:

    def __init__(self):

        self.extractors: Optional[List[AbstractEntity]] = []
        self.documents: Optional[List[str]] = []

        self.entities: Optional[set] = set()
        self.query_entities: Optional[set] = set()

        self.graph: Optional[GraphEntity] = None
        self.triplet: Optional[TripletExtractor] = None

        logger(f"Number of Entity Extractor downloaded: {len(self.extractors)} ")

    def entities_and_graphs_extractor(self) -> None:
        if self.documents and self.extractors:
            logger(f"Document's Entities extracting...")
            for i, document in enumerate(self.documents):
                for extractor in self.extractors:
                    extractor.set_text_extraction(document)
                    extractor.extractor_entity()
                    self.entities.update(*extractor.get_extract_entities())

                    if self.graph is not None:
                        self.graph.set_entities(extractor.get_extract_entities())
                        self.graph.add_to_knowledge_graph(document)

            if self.graph is not None:
                logger(f"Graph-entity status: {self.graph.get_knowledge_graph_stats()}")

            if self.triplet is not None:
                self.triplet.set_documents(self.documents)
                self.triplet.extract_triplets()

    # maybe its don't need
    def query_extractor(self, query: str) -> None:
        logger(f"Query extracting...")
        for extractor in self.extractors:
            extractor.set_text_extraction(query)
            extractor.extractor_entity()
            self.query_entities.add(*extractor.get_extract_entities())

    def set_documents(self, documents: List[str]) -> None:
        self.documents = documents

    def set_graph(self, graph: GraphEntity) -> None:
        self.graph = graph

    def set_triple_graph(self, triplet: TripletExtractor) -> None:
        self.triplet = triplet

    def set_extractors(self, extractors: List[AbstractEntity]) -> None:
        self.extractors = extractors

    def get_entities(self) -> Set:
        return self.entities

    def get_query_entities(self) -> Set:
        return self.query_entities