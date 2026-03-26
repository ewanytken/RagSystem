import unittest

from app.common.installer_system import Builder
from app.documents_processor.word_handler import WordPdfHandler
from app.entity.gliner2_entity import GlinerTwoEntity
from app.entity.gliner_entity import GlinerEntity
from app.entity.regex_entity import RegexEntity
from app.graph.graph_entity import GraphEntity
from app.indexer.indexer_object import Indexer
from app.logger import LoggerWrapper
from app.prompt.prompt_assembler import FinalAssembler
from app.respondent.local_model.transformer_wrapper import TransformerWrapper

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        word_handler = WordPdfHandler()
        regex = RegexEntity()
        llm = TransformerWrapper()
        self.final_assembler = FinalAssembler()
        indexer = Indexer()
        gliner_two = GlinerTwoEntity()
        gliner = GlinerEntity()
        graph = GraphEntity()

        self.installer_system = (Builder()
                                 .set_indexer(indexer)
                                 .set_document_handler(word_handler)
                                 .set_entities_extractors(regex)
                                 .set_entities_extractors(gliner)
                                 .set_entities_extractors(gliner_two)
                                 .set_llm_responder(llm)
                                 .set_graph_entity(graph)
                                 .set_prompt_object(self.final_assembler)
                                 .build())

    def test_document_processing(self):
        chunk, doc = self.installer_system.documents_processor()
        self.installer_system.indexer_installer_processor(chunk)
        self.installer_system.extractor_processor(chunk)

        query = "What's ASR?"
        retrieved_doc, txt = self.installer_system.indexer_query(query)
        logger(f"RETRIEVED DOC: {retrieved_doc}")
        entities_from_graph = self.installer_system.find_entities_from_graph(retrieved_doc)
        logger(f"ENTITIES FROM GRAPH: {entities_from_graph}")
        final_prompt = self.installer_system.prompt_processor(query=query, retrieved_docs=retrieved_doc, entities=entities_from_graph)
        logger(f"FINAL PROMPT: {final_prompt}")
        logger(f"OUTPUT FROM LLM: {self.installer_system.llm_model_processor(final_prompt)}")

if __name__ == '__main__':
    unittest.main()