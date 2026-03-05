import os
import unittest

from app.documents_processor.word_handler import WordHandler
from app.logger import LoggerWrapper
from app.utils import Utils

logger = LoggerWrapper()

class TestRAGSystem(unittest.TestCase):

    def setUp(self):
        self.word_handler = WordHandler()

    def test_document_processing(self):
        config = Utils.get_config_file()
        dic = Utils.load_label_description()
        temp = Utils.load_template("extraction_template_eng")
        logger(temp)
        logger(config)
        logger(dic)
        logger(config['paths']['documents_dir'])

        self.word_handler.set_config(config)

        self.word_handler.handle_documents()
        logger(self.word_handler.get_chunked_documents())
        logger(self.word_handler.get_handled_documents())


if __name__ == '__main__':
    unittest.main()