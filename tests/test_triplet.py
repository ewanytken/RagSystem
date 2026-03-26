import unittest

from app.documents_processor.word_handler import WordPdfHandler
from app.graph.triplet_extractor import TripletExtractor
from app.logger import LoggerWrapper
from app.respondent.external_model.respondent_other_service import ExternalModel
from app.respondent.local_model.transformer_wrapper import TransformerWrapper
from app.utils import Utils

logger = LoggerWrapper()

class TestRAGSystem(unittest.TestCase):

    def setUp(self):
        self.triplet = TripletExtractor()
        self.word_handler = WordPdfHandler()

    def test_document_processing(self):
        config = Utils.get_config_file()

        self.word_handler.set_config(config)
        self.word_handler.handle_documents()
        doc_in_chunks = self.word_handler.get_chunked_documents()

        self.triplet.set_config(config)
        self.triplet.set_documents(doc_in_chunks)
        # model = TransformerWrapper()
        model = ExternalModel()

        self.triplet.set_llm_model(model)
        self.triplet.extract_triplets()

        query = "Кто является генеральным директором АО «Селектел»?"
        self.triplet.extract_triplets(query)
        triplet_query = self.triplet.get_extracted_query()
        logger(self.triplet.get_extracted_relation())
        logger(self.triplet.get_extracted_query())
        for triplet in triplet_query:
            self.triplet.search_relation_by_subject(triplet)
            logger(self.triplet.get_extracted_relation())

            self.triplet.search_relation_from_graph(triplet)
            logger(self.triplet.get_extracted_relation())

if __name__ == '__main__':
    unittest.main()