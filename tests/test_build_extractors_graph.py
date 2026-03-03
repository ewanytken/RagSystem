import unittest

from app.common.installer_system import Builder
from app.documents_processor.word_handler import WordHandler
from app.entity.gliner_entity import GlinerEntity
from app.entity.regex_entity import RegexEntity
from app.graph.graph_entity import GraphEntity
from app.indexer.indexer_object import Indexer
from app.logger import LoggerWrapper
from app.prompt.prompt_object import PromptObject
from app.respondent.local_model.transformer_wrapper import TransformerWrapper

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        word_handler = WordHandler()
        regex = RegexEntity()
        llm = TransformerWrapper()
        prompt = PromptObject()
        indexer = Indexer()
        gliner = GlinerEntity()
        graph = GraphEntity()
        self.installer_system = (Builder()
                                 .set_indexer(indexer)
                                 .set_document_handler(word_handler)
                                 .set_entities_extractors(regex)
                                 .set_entities_extractors(gliner)
                                 .set_llm_responder(llm)
                                 .set_graph_entity(graph)
                                 .set_prompt_object(prompt)
                                 .build())

    def test_document_processing(self):
        chunk, doc = self.installer_system.documents_processor()
        self.installer_system.indexer_installer_processor(chunk)
        entities = self.installer_system.extractor_processor(chunk)

        query = "What's ASR?"
        retrieved_doc = self.installer_system.indexer_query(query)
        entities_from_graph = self.installer_system.find_entities_from_graph(query)
        entities.update(entities_from_graph)
        final_prompt = self.installer_system.prompt_processor(query=query, context=retrieved_doc, entities=entities)
        logger(self.installer_system.llm_model_processor(final_prompt))

if __name__ == '__main__':
    unittest.main()