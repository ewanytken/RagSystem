import unittest

from app.documents_processor.word_handler import WordPdfHandler
from app.logger import LoggerWrapper
from app.respondent.local_model.transformer_wrapper import TransformerWrapper
from app.utils import Utils

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        self.config = Utils.get_config_file()
        self.doc_handler = WordPdfHandler()
        self.doc_handler.set_config(self.config)
        self.transformer_wrapper = TransformerWrapper()

    def test_document_processing(self):
        self.doc_handler.handle_documents()
        doc = [" ".join(d) for d in self.doc_handler.get_handled_documents()]
        print(doc)
        response = self.transformer_wrapper.generate(doc[0])
        logger(response)

if __name__ == '__main__':
    unittest.main()