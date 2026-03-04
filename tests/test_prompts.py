import unittest

from app.documents_processor.word_handler import WordHandler
from app.logger import LoggerWrapper
from app.utils import Utils

logger = LoggerWrapper()

class TestRAGSystem(unittest.TestCase):

    def setUp(self):
        self.word_handler = WordHandler()
        config = Utils.get_config_file()

        self.word_handler.set_config(config)
        self.word_handler.handle_documents()

    def test_document_processing(self):
        temp = Utils.load_template("template_two")
        chunk = self.word_handler.get_chunked_documents()

        prompt = temp.format(document=chunk)

        print(prompt)




if __name__ == '__main__':
    unittest.main()