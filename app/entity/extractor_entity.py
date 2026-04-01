
from typing import List, Optional, Set

from app.entity.abstract_entity import AbstractEntity
from app.graph.graph_entity import GraphEntity
from app.graph.triplet_extractor import TripletExtractor
from app.logger import LoggerWrapper, LoggerAuxiliary

logger = LoggerWrapper()
logger_aux = LoggerAuxiliary()

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

        self.entities: Optional[List] = []
        self.query_entities: Optional[set] = set()

        self.graph: Optional[GraphEntity] = None
        self.triplet: Optional[TripletExtractor] = None

        logger(f"Number of Entity Extractor downloaded: {len(self.extractors)} ")

    def entities_and_graphs_extractor(self) -> None:
        if self.documents and self.extractors:
            logger(f"Document's Entities extracting...")
            for document in self.documents:
                for extractor in self.extractors:
                    extractor.set_text_extraction(document)
                    extractor.extractor_entity()
                    unique_extraction = self.unique_by_entity(extractor.get_extract_entities())
                    self.entities.extend(unique_extraction)

                    if self.graph is not None:
                        self.graph.set_entities(unique_extraction)
                        self.graph.add_to_knowledge_graph(document)

            if self.graph is not None:
                logger_aux(f"Graph-entity status: {self.graph.summary_graph_entities()}")

            if self.triplet is not None:
                self.triplet.set_documents(self.documents)
                self.triplet.extract_triplets()

    # maybe it don't need
    def query_extractor(self, query: str) -> None:
        logger(f"Query extracting...")
        for extractor in self.extractors:
            extractor.set_text_extraction(query)
            extractor.extractor_entity()
            self.query_entities.add(*extractor.get_extract_entities())

    def unique_by_entity(self, entities: Optional[List]):
        seen_ids = set()
        unique_data = []

        for entity in entities:
            if entity['entity'] not in seen_ids:
                unique_data.append(entity)
                seen_ids.add(entity['entity'])

        return unique_data

    def set_documents(self, documents: List[str]) -> None:
        self.documents = documents

    def set_graph(self, graph: GraphEntity) -> None:
        self.graph = graph

    def set_triple_graph(self, triplet: TripletExtractor) -> None:
        self.triplet = triplet

    def set_extractors(self, extractors: List[AbstractEntity]) -> None:
        self.extractors = extractors

    def get_entities(self) -> List:
        return self.entities

    def get_query_entities(self) -> Set:
        return self.query_entities