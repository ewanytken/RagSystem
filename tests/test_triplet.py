import unittest

from app.documents_processor.word_handler import WordHandler
from app.graph.triplet_extractor import TripletExtractor
from app.logger import LoggerWrapper
from app.respondent.local_model.transformer_wrapper import TransformerWrapper
from app.utils import Utils

logger = LoggerWrapper()

class TestRAGSystem(unittest.TestCase):

    def setUp(self):
        self.triplet = TripletExtractor()
        self.word_handler = WordHandler()

    def test_document_processing(self):
        config = Utils.get_config_file()

        self.word_handler.set_config(config)
        self.word_handler.handle_documents()
        chunk = self.word_handler.get_chunked_documents()[:2]

        self.triplet.set_config(config)
        self.triplet.set_documents(chunk)
        model = TransformerWrapper()
        self.triplet.set_llm_model(model)
        self.triplet.extract_triplets()
        self.triplet.search_relation_by_subject("models")
        logger(self.triplet.get_extracted_relation())

        self.triplet.search_relation_from_graph("models",  "included", "management")
        logger(self.triplet.get_extracted_relation())

if __name__ == '__main__':
    unittest.main()