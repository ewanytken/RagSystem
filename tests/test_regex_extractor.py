import unittest

from app.documents_processor.word_handler import WordPdfHandler
from app.entity.regex_entity import RegexEntity
from app.logger import LoggerWrapper
from app.utils import Utils

logger = LoggerWrapper()

class TestRAGSystem(unittest.TestCase):

    def setUp(self):
        self.gliner = RegexEntity()
        self.word_handler = WordPdfHandler()

    def test_document_processing(self):
        config = Utils.get_config_file()

        self.word_handler.set_config(config)
        self.word_handler.handle_documents()
        documents = self.word_handler.get_handled_documents()
        logger(documents)
        logger(len(documents))

        self.gliner.set_text_extraction(documents[0])
        self.gliner.extractor_entity()
        logger(self.gliner.get_extract_entities())

if __name__ == '__main__':
    unittest.main()