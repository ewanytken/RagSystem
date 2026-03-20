import unittest
from enum import Enum
from typing import Optional, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.common.installer_system import Builder, InstallerSystem
from app.documents_processor.word_handler import WordHandler
from app.entity.abstract_entity import AbstractEntity
from app.graph.graph_entity import GraphEntity
from app.graph.triplet_extractor import TripletExtractor
from app.indexer.indexer_object import Indexer
from app.logger import LoggerWrapper
from app.prompt.abstract_prompt import AbstractPrompt
from app.prompt.prompt_assembler import FinalAssembler
from app.respondent.abstract_respondent import Respondent
from app.respondent.local_model.transformer_wrapper import TransformerWrapper

console = Console()
logger = LoggerWrapper()

class ModelProvider(Enum):
    LOCAL = "local"
    REMOTE = "remote"

class ModelRemote(Enum):
    OLLAMA = "ollama"
    GIGA = "giga"
    EXTERNAL = "external_service"

class ModelLocalTicker:
    QWEN3 = "Qwen/Qwen3-4B-Instruct-2507"
    FELLADRIN = "Felladrin/TinyMistral-248M-Chat-v3"
    DEFAULT = ""

class PromptProvider(Enum):
    ADVANCED = "advanced"
    SIMPLE = "simple"

class Constructor:
    def __init__(self):
        self.word_handler: Optional[WordHandler] = WordHandler()
        self.indexer: Optional[Indexer] = Indexer()
        self.graph: Optional[GraphEntity] = GraphEntity()

        self.installer_system_builder = (Builder()
                                         .set_document_handler(self.word_handler)
                                         .set_indexer(self.indexer)
                                         .set_graph_entity(self.graph))

        self.respondent: Optional[Respondent] = None

        self.regex: Optional[AbstractEntity] = None
        self.gliner_two: Optional[AbstractEntity] = None
        self.gliner: Optional[AbstractEntity] = None

        self.triplets: Optional[TripletExtractor] = None

        self.prompter: Optional[AbstractPrompt] = None

        Constructor.show_banner()

    def get_installer_system(self) -> InstallerSystem:
        return self.installer_system_builder

    @staticmethod
    def show_banner():
        banner = """
        ╔═══════════════════════════════════════════════════════════╗
        ║     🚀 Advanced RAG Interactive System                    ║
        ║     Build your custom RAG pipeline component by component ║
        ╚═══════════════════════════════════════════════════════════╝
        """
        console.print(Panel(banner, style="bold cyan", border_style="bright_blue"))

    def configure_modules(self):
        console.print("\n[bold blue]📚 Modules Configurator[/bold blue]")


        console.print("\n[bold yellow]🎯 Choose Prompt Assembler[/bold yellow]")
        self.prompter = self.prompt_chooser()
        self.installer_system_builder.set_prompt_object(self.prompter)


        console.print("\n[bold yellow]🎯 Choose Model Respondent[/bold yellow]")
        self.respondent = self.model_chooser()
        self.installer_system_builder.set_llm_responder(self.respondent)

        self.installer_system_builder = self.installer_system_builder.build()
        self.show_summary()

    def prompt_chooser(self) -> AbstractPrompt:
        prompter: Optional[AbstractPrompt] = FinalAssembler()

        if prompter is None:
            raise Exception("Prompt assembler is None [[121]]")

        return prompter

    def model_chooser(self) -> Respondent:

        respondent: Optional[Respondent] = TransformerWrapper()

        return respondent

    def show_summary(self):

        table = Table(title="RAG System Configuration Summary", border_style="cyan")
        table.add_column("Number", style="cyan")
        table.add_column("Component", style="green")
        table.add_column("Status", style="yellow")

        components = [self.prompter, self.respondent, self.word_handler,
                      self.regex, self.gliner, self.gliner_two,
                      self.indexer,
                      self.graph, self.triplets]

        for num, component in enumerate(components):
            if component:
                table.add_row(
                    str(num),
                    component.__repr__(),
                    "✅ Configured"
                )
            else:
                table.add_row(
                    str(num),
                    component.__repr__(),
                    "❌ Not configured",
                )

        console.print(table)

class ApiCall(Constructor):

    def __init__(self):
        super().__init__()
        self.complete_installer: Optional[InstallerSystem] = None
        self.query: Optional[str] = None

    def set_query(self, query: str) -> None:
        self.query = query

    def get_query(self) -> str:
        return self.query

    def run_interactive(self):

        response: Optional[str] = None
        console.print("\n[bold cyan] Building RAG System[/bold cyan]")

        self.configure_modules()

        try:
            self.complete_installer = self.get_installer_system()
        except Exception as e:
            logger(f"Cannot get installer system or installer not complete [[122]] {e}")

        console.print("\n[bold cyan] Load Documents and chinking them[/bold cyan]")
        doc, chunk = self.complete_installer.documents_processor()

        console.print("\n[bold cyan] Indexing documents with embedding model[/bold cyan]")
        self.complete_installer.indexer_installer_processor(chunk)

        if self.complete_installer.get_extractors():
            console.print("\n[bold cyan] Extracting Entities and make Graphs[/bold cyan]")
            self.complete_installer.extractor_processor(chunk)

        doc_dict_by_query: Optional[List[Dict]] = None
        doc_str_only_by_query: Optional[List[str]] = None
        entities_by_document_search: Optional[List[Dict]] = None
        triplets_by_full_query: Optional[List[Dict]] = None
        triplets_by_subject: Optional[List[Dict]] = None
        triplets: Optional[List[Dict]] = None
        try:
            if self.get_query():
                console.print("\n[bold cyan] Query obtain[/bold cyan]")
                console.print("\n[bold cyan] Retrieving documents from Indexer[/bold cyan]")

                doc_dict_by_query, doc_str_only_by_query = self.complete_installer.indexer_query(self.query)

                if doc_str_only_by_query and self.complete_installer.get_extractors():
                    console.print("\n[bold cyan] Find entities by extracted documents in Entity Graph[/bold cyan]")
                    entities_by_document_search = self.complete_installer.find_entities_from_graph(doc_str_only_by_query)

                if self.complete_installer.get_triplet_graph():
                    console.print("\n[bold cyan] Find triplets by Obtain Query in Triplets Graph[/bold cyan]")

                    triplets_by_full_query = self.complete_installer.find_triplets(self.query)
                    triplets_by_subject = self.complete_installer.find_triplets_by_subject(self.query)

                    if triplets_by_full_query and triplets_by_subject:
                        triplets = triplets_by_full_query + triplets_by_subject
                    else:
                        triplets = triplets_by_subject if triplets_by_subject else triplets_by_full_query

                console.print("\n[bold cyan] Assembling final prompt from all available information[/bold cyan]")
                assembled_prompt: Optional[str] = self.complete_installer.prompt_processor(self.query, doc_dict_by_query, entities_by_document_search, triplets)

                console.print("\n[bold cyan] Request to LLM Respondent[/bold cyan]")
                response = self.complete_installer.llm_model_processor(assembled_prompt)

        except Exception as e:
            logger(f"Answer don't obtain. System don't work correctly [[123]] {e}")

        if response:
            raise Exception(f"Answer don't assign")

        return response
class Test(unittest.TestCase):

    def setUp(self):
        self.api = ApiCall()

    def test_document_processing(self):

        self.api.set_query("What is ASR?")
        response = self.api.run_interactive()
        print(response)

if __name__ == '__main__':
    unittest.main()