from typing import List, Optional

from app.documents_processor.word_handler import WordHandler
from app.entity.extractor_entity import EntityExtractor
from app.entity.gliner_entity import GlinerEntity
from app.entity.regex_entity import RegexEntity
from app.graph.graph_entity import GraphEntity
from app.graph.triplet_extractor import TripletExtractor
from app.indexer.indexer_object import Indexer
from app.logger import LoggerWrapper
from app.prompt.prompt_object import PromptObjectBuilder
from app.respondent.abstract_respondent import Respondent
from app.utils import Utils

logger = LoggerWrapper()

class InstallerSystem:

    regex: Optional[RegexEntity] = None
    gliner: Optional[GlinerEntity] = None
    word_handler: Optional[WordHandler] = None
    indexer: Optional[Indexer] = None
    graph_entity: Optional[GraphEntity] = None
    llm_responder: Optional[Respondent] = None
    triplet_graph: Optional[TripletExtractor] = None

    def __init__(self):
        self.config = Utils.get_config_file()
        self.extractor = EntityExtractor()

    def documents_processor(self, *args, **kwargs) -> tuple[list[str], list[str]]:
        self.word_handler.set_config(self.config)
        self.word_handler.handle_documents()
        return self.word_handler.get_chunked_documents(), self.word_handler.get_handled_documents()

    def indexer_installer_processor(self, documents, *args, **kwargs) -> None:
        try:
            self.indexer.set_config(self.config)
            self.indexer.set_embedding_model()
            self.indexer.documents_indexing(documents)
        except Exception as e:
            logger(f"Documents indexing failed [[63]]: {e}")

    def indexer_query(self, query, *args, **kwargs) -> list[str]:
        try:
            self.indexer.documents_retriever(query)
            return self.indexer.get_retrieval_documents()
        except Exception as e:
            logger(f"Indexer query failed [[64]]: {e}")

    def prompt_processor(self, query, context, entities, *args, **kwargs) -> str:
        builder_prompts = PromptObjectBuilder()
        prompt_object = (builder_prompts
                         .set_query(query)
                         .set_context(context)
                         .set_entities(entities) # entities = [{'label': 111, 'entity': 222, 'score': 333}]
                         .set_path_to_template(self.config["templates"]["prompt_template"]).build())
        prompt_object.set_prompt()
        return prompt_object.get_prompt()

    def extractor_processor(self, documents: List[str]) -> set:
        label_for_gliner = list(set(Utils.load_dictionary().values()))

        self.gliner.set_config(self.config)
        self.gliner.set_gliner_model()
        self.gliner.set_gliner_label(label_for_gliner)

        entities_extractors = [self.regex, self.gliner]

        self.extractor.set_extractors(entities_extractors)
        self.extractor.set_documents(documents)

        self.extractor.set_graph(self.graph_entity)
        self.extractor.set_triple_graph(self.triplet_graph)

        self.extractor.entities_and_graphs_extractor()

        return self.extractor.get_entities()

    def find_entities_from_graph(self, query: str) -> list[dict]:
        return self.graph_entity.find_related_entities(query)

    def find_triplets(self, query: str) -> list[dict]:
        self.triplet_graph.extract_triplets(query)
        triplet = self.triplet_graph.get_extracted_query()
        for t in triplet:
            self.triplet_graph.search_relation_from_graph(t[0], t[1], t[2])
        return self.triplet_graph.get_extracted_relation()

    def find_triplets_by_subject(self, query: str) -> list[dict]:
        self.triplet_graph.extract_triplets(query)
        triplet = self.triplet_graph.get_extracted_query()
        for t in triplet:
            self.triplet_graph.search_relation_by_subject(t[0])
        return self.triplet_graph.get_extracted_relation()

    def llm_model_processor(self, prompt: str) -> str:
        self.llm_responder.set_config(self.config)
        return self.llm_responder.generate(prompt)

    def set_llm_model(self, llm_responder):
        self.llm_responder = llm_responder