import unittest

from app.documents_processor.word_handler import WordPdfHandler
from app.entity.extractor_entity import EntityExtractor
from app.entity.gliner2_entity import GlinerTwoEntity
from app.entity.gliner_entity import GlinerEntity
from app.entity.regex_entity import RegexEntity
from app.graph.graph_entity import GraphEntity
from app.graph.triplet_extractor import TripletExtractor
from app.logger import LoggerWrapper
from app.respondent.external_model.respondent_other_service import ExternalModel
from app.utils import Utils

logger = LoggerWrapper()

class TestRAGSystem(unittest.TestCase):

    def setUp(self):
        self.word_handler = WordPdfHandler()

        self.config = Utils.get_config_file()

        self.graph = GraphEntity()

        self.gliner = GlinerEntity()
        self.gliner_two = GlinerTwoEntity()
        self.regex = RegexEntity()
        self.triplet = TripletExtractor()

    def test_document_processing(self):
        self.word_handler.set_config(self.config)
        self.word_handler.handle_documents()
        chunks = self.word_handler.get_chunked_documents()

        chunks = chunks[:1]
        dic = Utils.load_label_description()

        self.gliner.set_gliner_label(dic)
        self.gliner.set_config(self.config)
        self.gliner.set_gliner_model()

        self.gliner_two.set_gliner_label(dic)
        self.gliner_two.set_config(self.config)
        self.gliner_two.set_gliner_model()

        extractor = EntityExtractor()
        extractor.set_extractors([self.gliner, self.gliner_two, self.regex])
        extractor.set_graph(graph=self.graph)
        extractor.set_documents(documents=chunks)

        self.triplet.set_llm_model(ExternalModel())
        self.triplet.set_config(self.config)

        extractor.set_triple_graph(self.triplet)
        extractor.entities_and_graphs_extractor()

        logger(extractor.get_entities())

if __name__ == '__main__':
    unittest.main()