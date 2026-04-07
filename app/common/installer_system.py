from typing import List, Optional, Union

from app.documents_processor.abstract_document_handler import DocumentHandler
from app.entity.abstract_entity import AbstractEntity
from app.entity.extractor_entity import EntityExtractor
from app.entity.gliner2_entity import GlinerTwoEntity
from app.entity.gliner_entity import GlinerEntity
from app.graph.graph_entity import GraphEntity
from app.graph.triplet_extractor import TripletExtractor
from app.indexer.indexer_object import Indexer
from app.logger import LoggerWrapper
from app.prompt.abstract_prompt import AbstractPrompt
from app.respondent.abstract_respondent import Respondent
from app.utils import Utils

logger = LoggerWrapper()

class InstallerSystem:

    def __init__(self):

        self.extractors: Optional[List[AbstractEntity]] = []
        self.word_handler: Optional[DocumentHandler] = None
        self.prompt_object: Optional[AbstractPrompt] = None
        self.indexer: Optional[Indexer] = None
        self.graph_entity: Optional[GraphEntity] = None
        self.llm_responder: Optional[Respondent] = None
        self.triplet_graph: Optional[TripletExtractor] = None

        self.config = Utils.get_config_file()
        self.extractor = EntityExtractor()

    def documents_processor(self) -> tuple[list[str], list[str]]:
        self.word_handler.set_config(self.config)
        self.word_handler.handle_documents()
        return self.word_handler.get_chunked_documents(), self.word_handler.get_handled_documents()

    def indexer_installer_processor(self, documents) -> None:
        try:
            self.indexer.set_config(self.config)
            self.indexer.set_embedding_model()
            self.indexer.documents_indexing(documents)
        except Exception as e:
            logger(f"Documents indexing failed [[63]]: {e}")

    def indexer_query(self, query) -> tuple[list[dict], list[str]]:
        try:
            self.indexer.documents_retriever(query)
            return self.indexer.get_retrieval_documents(), self.indexer.get_retrieved_text_only()
        except Exception as e:
            logger(f"Indexer query failed [[64]]: {e}")

    def prompt_processor(self, query: str, retrieved_docs: List, entities: List = None, triplets: List = None) -> str:
        self.prompt_object.set_config(self.config)
        self.prompt_object.set_entities(entities)
        self.prompt_object.set_triplet(triplets)
        self.prompt_object.set_chunks(retrieved_docs)
        self.prompt_object.set_query(query)
        self.prompt_object.make_final_prompt()
        return self.prompt_object.get_final_prompt()

    def extractor_processor(self, documents: List[str]) -> List:
        label_for_gliner = Utils.load_label_description()

        if self.extractors:
            for ext in self.extractors:
                if isinstance(ext, GlinerEntity) or isinstance(ext, GlinerTwoEntity):
                    ext.set_config(self.config)
                    ext.set_gliner_model()
                    ext.set_gliner_label(label_for_gliner)

            self.extractor.set_extractors(self.extractors)
        else:
            logger(f"Entities Extractors doesn't install {len(self.extractors)}")

        self.extractor.set_documents(documents)

        if self.graph_entity:
            self.extractor.set_graph(self.graph_entity)

        if self.triplet_graph:
            self.triplet_graph.set_config(self.config)
            self.triplet_graph.set_documents(documents)
            self.extractor.set_triple_graph(self.triplet_graph)

        self.extractor.entities_and_graphs_extractor()

        return self.extractor.get_entities()

    def find_entities_from_graph(self, indexer_doc: Union[str, List]) -> list[dict]:
        if self.graph_entity is not None:
            return self.graph_entity.find_related_entities_from_doc(indexer_doc)
        else:
            logger(f"Graph entity don't install. Return empty list.")
            return []

    def find_docs_from_graph(self, entity: str) -> set:
        if self.graph_entity is not None:
            return self.graph_entity.find_doc_by_entity(entity)
        else:
            logger(f"Graph entity don't install. Return empty set.")
            return set()

    def find_triplets(self, query: str) -> list[dict]:
        triplets_list = []
        if self.triplet_graph:
            self.triplet_graph.extract_triplets(query)
            triplet = self.triplet_graph.get_triplets_from_query()
            for t in triplet:
                self.triplet_graph.search_relation_by_subject(t[0])
                triplets_list.extend(self.triplet_graph.get_triplets_from_graph())

                self.triplet_graph.search_relation_from_graph(t[0], t[1], t[2])
                triplets_list.extend(self.triplet_graph.get_triplets_from_graph())
            return triplets_list
        else:
            logger(f"Triplets Graph don't install. Return empty list.")
            return []

    def llm_model_processor(self, prompt: str) -> str:
        if self.llm_responder is not None:
            return self.llm_responder.generate(prompt)
        else:
            logger(f"LLM Model not installed. Return empty string.")
            return ""

    def get_extractors(self) -> list[AbstractEntity]:
        return self.extractors

    def get_triplet_graph(self) -> TripletExtractor:
        return self.triplet_graph

    def get_entities_graph(self) -> GraphEntity:
        return self.graph_entity

class Builder:

    def __init__(self):
        self.installer = InstallerSystem()

    def set_entities_extractors(self, extractor: AbstractEntity):
        self.installer.extractors.append(extractor)
        return self

    def set_indexer(self, indexer: Indexer):
        self.installer.indexer = indexer
        return self

    def set_graph_entity(self, graph_entity: GraphEntity):
        self.installer.graph_entity = graph_entity
        return self

    def set_triplet_graph(self, triplet_graph: TripletExtractor):
        self.installer.triplet_graph = triplet_graph
        return self

    def set_llm_responder(self, llm_responder: Respondent):
        self.installer.llm_responder = llm_responder
        return self

    def set_document_handler(self, word_handler: DocumentHandler):
        self.installer.word_handler = word_handler
        return self

    def set_prompt_object(self, prompt_object: AbstractPrompt):
        self.installer.prompt_object = prompt_object
        return self

    def build(self) -> InstallerSystem:
        return self.installer
