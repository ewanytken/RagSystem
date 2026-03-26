import unittest

from app.documents_processor.word_handler import WordPdfHandler
from app.logger import LoggerWrapper
from app.prompt.prompt_assembler import FinalAssembler

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        self.word_handler = WordPdfHandler()
        self.final_assembler = FinalAssembler()

    def test_document_processing(self):
        chunks = [{"text": "value11_chunks",
                   "score": "value22_chunks"},
                  {"text": "value2222_chunks",
                   "score": "value23_chunks"}]

        triplets = [{"subject": "triple11_chunks", "predicate": "triple22_chunks", "object": "triple2222_chunks", "document": "triple23_chunks"},
                    {"subject": "triple444_chunks", "predicate": "triple333chunks", "object": "triple2555552_chunks", "document": "triple2345345_chunks"}]

        entities = [{"entity": "value_entities", "label": "value2_entities"},
                    {"entity": "valu325235tities", "label": "value234324ities"}]

        query = "QUERYQUERYQUERYQUERYQUERYQUERYQUERYQUERYQUERY"

        self.final_assembler.set_query(query)
        self.final_assembler.set_triplet(triplets)
        self.final_assembler.set_entities(entities)
        self.final_assembler.set_chunks(chunks)
        self.final_assembler.make_final_prompt()
        print(self.final_assembler.get_final_prompt())

if __name__ == '__main__':
    unittest.main()