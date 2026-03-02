from typing import List, Dict
from app.documents_processor.word_handler import WordHandler
from app.entity.extractor_entity import EntityExtractor
from app.entity.gliner_entity import GlinerEntity
from app.entity.regex_entity import RegexEntity
from app.graph.graph_entity import GraphEntity
from app.indexer.indexer_object import Indexer
from app.prompt.prompt_object import PromptObject, PromptObjectBuilder
from app.respondent.external_model.respondent_ollama import OllamaModel
from app.respondent.external_model.respondent_other_service import ExternalModel
from app.respondent.local_model.transformer_wrapper import TransformerWrapper
from app.utils import Utils


class DocumentProcessor:
    def __init__(self):
        self.word_handler = WordHandler()
        self.word_handler.set_config(Utils.get_config_file())
        self.word_handler.handle_documents()

        chunks = self.word_handler.get_chunked_documents()
        documents = self.word_handler.get_handled_documents()

class IndexerProcessor:
    def __init__(self, documents: List[str], query_retriever: str) -> None:
        self.indexer = Indexer()
        self.indexer.set_config(Utils.get_config_file())
        self.indexer.set_embedding_model()

        self.indexer.documents_indexing(documents)
        self.indexer.documents_retriever(query_retriever)

        retrieved_document = self.indexer.get_retrieval_documents()

class PromptProcessor:
    def __init__(self, entities: List[Dict], query: str, context: str, template: str) -> None:
        self.prompt_object = PromptObject()
        self.builder = PromptObjectBuilder()
        config = Utils.get_config_file()

        # entities = [{'label': 111, 'entity': 222, 'score': 333}]
        prompt_object = (self.builder
                         .set_query(query)
                         .set_context(context)
                         .set_entities(entities)
                         .set_path_to_template(config).build())
        prompt_object.set_prompt()

        prompt = prompt_object.get_prompt()

class ExtractorProcessor:
    def __init__(self, documents: List[str], find_entity_from_graph: str) -> None:
        config = Utils.get_config_file()
        label_for_gliner = list(set(Utils.load_dictionary().values()))

        self.regex = RegexEntity()

        self.gliner = GlinerEntity()
        self.gliner.set_config(config)
        self.gliner.set_gliner_model()
        self.gliner.set_gliner_label(label_for_gliner)

        entities_extractors = [self.regex, self.gliner]

        extractor = EntityExtractor()
        extractor.set_extractors(entities_extractors)

        graph_entity = GraphEntity()
        extractor.set_graph(graph_entity)

        extractor.set_documents(documents)
        extractor.entities_and_graphs_extractor()

        related = graph_entity.find_related_entities(find_entity_from_graph)
        entity = extractor.get_entities()

class LocalModelProcessor:
    def __init__(self, prompt: str):
        config = Utils.get_config_file()
        model = TransformerWrapper()
        model.set_config(config)

        output = model.generate(prompt)


class OllamaModelProcessor:
    def __init__(self, prompt: str):
        config = Utils.get_config_file()
        model = OllamaModel()
        model.set_config(config)

        output = model.generate(prompt)

class ExternalModelProcessor:
    def __init__(self, prompt: str):
        config = Utils.get_config_file()
        model = ExternalModel()
        model.set_config(config)

        output = model.generate(prompt)