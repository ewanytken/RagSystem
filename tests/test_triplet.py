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
        documents = self.word_handler.get_handled_documents()
        logger(documents)
        logger(len(documents))

        self.triplet.set_config(config)
        self.triplet.set_documents(documents)
        model = TransformerWrapper()
        self.triplet.set_llm_model(model)
        self.triplet.extract_triplets()
        self.triplet.get_extracted_relation()

if __name__ == '__main__':
    unittest.main()