import unittest

from app.documents_processor.word_handler import WordHandler
from app.entity.extractor_entity import EntityExtractor
from app.entity.gliner_entity import GlinerEntity
from app.entity.regex_entity import RegexEntity
from app.graph.graph_entity import GraphEntity
from app.logger import LoggerWrapper
from app.prompt.prompt_object import PromptObject
from app.utils import Utils

logger = LoggerWrapper()

class TestRAGSystem(unittest.TestCase):

    def setUp(self):
        self.prompt_object = PromptObject()
        self.word_handler = WordHandler()

    def test_document_processing(self):
        config = Utils.get_config_file()
        graph = GraphEntity()
        gliner = GlinerEntity()
        gliner.set_config(config)
        regex = RegexEntity()
        extractor = EntityExtractor()
        extractor.set_extractors([gliner, regex])
        extractor.set_graph(graph=graph)

        dic = Utils.load_dictionary()
        dic.values()

        self.word_handler.set_config(config)

        self.word_handler.handle_documents()
        documents = self.word_handler.get_handled_documents()
        logger(documents)

        extractor.set_documents(documents=documents)
        extractor.entities_and_graphs_extractor()

        logger(graph.get_knowledge_graph_stats())
        logger(extractor.get_entities())


if __name__ == '__main__':
    unittest.main()