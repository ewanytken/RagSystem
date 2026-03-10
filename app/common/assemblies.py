from typing import Optional

from app.common.installer_system import Builder
from app.documents_processor.word_handler import WordHandler
from app.entity.gliner2_entity import GlinerTwoEntity
from app.entity.gliner_entity import GlinerEntity
from app.entity.regex_entity import RegexEntity
from app.graph.graph_entity import GraphEntity
from app.graph.triplet_extractor import TripletExtractor
from app.indexer.indexer_object import Indexer
from app.prompt.abstract_prompt import AbstractPrompt
from app.prompt.prompt_assembler import FinalAssembler
from app.prompt.prompt_object import PromptObject
from app.respondent.abstract_respondent import Respondent
from app.respondent.local_model.transformer_wrapper import TransformerWrapper


class Assembler:
    def __init__(self):
        self.word_handler: Optional[WordHandler] = WordHandler()
        self.indexer: Optional[Indexer] = Indexer()

        self.llm_extractor: Optional[Respondent] = TransformerWrapper()
        self.responder: Optional[Respondent] = TransformerWrapper()


        self.regex: Optional[RegexEntity] = RegexEntity()
        self.gliner_two: Optional[GlinerTwoEntity] = GlinerTwoEntity()
        self.gliner: Optional[GlinerEntity] = GlinerEntity()

        self.graph: Optional[GraphEntity] = GraphEntity()
        self.triplet: Optional[TripletExtractor] = TripletExtractor()

        # prompt_object - simple, prompt_assembler - advanced
        self.prompt_assembler: Optional[FinalAssembler] = FinalAssembler()
        self.prompt_object: Optional[PromptObject] = PromptObject()


    def set_prompter(self, prompter: AbstractPrompt):
        self.prompt_object = prompter

    def set_responder(self, responder: Respondent):
        self.responder = responder

    def set_llm_extractor(self, llm_extractor: Respondent):
        self.llm_extractor = llm_extractor

