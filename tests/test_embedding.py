import unittest

from app.documents_processor.word_handler import WordHandler
from app.indexer.indexer_object import Indexer
from app.logger import LoggerWrapper
from app.utils import Utils

logger = LoggerWrapper()

class TestRAGSystem(unittest.TestCase):

    def setUp(self):
        self.embedding = Indexer()
        self.word_handler = WordHandler()

    def test_document_processing(self):
        config = Utils.get_config_file()

        self.word_handler.set_config(config)
        self.word_handler.handle_documents()
        chunk = self.word_handler.get_chunked_documents()

        self.embedding.set_config(config)
        self.embedding.set_embedding_model()
        self.embedding.documents_indexing(chunk)
        self.embedding.documents_retriever("attack response rate")
        self.embedding.get_retrieval_documents()
        logger(self.embedding.get_retrieved_text_only())

if __name__ == '__main__':
    unittest.main()