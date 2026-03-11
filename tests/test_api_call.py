import unittest
from enum import Enum
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.common.complete_interface import ApiCall
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

        respondent: Optional[Respondent] = TransformerWrapper("Felladrin/TinyMistral-248M-Chat-v3")

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

class Test(unittest.TestCase):

    def setUp(self):
        self.api = Constructor()

    def test_document_processing(self):
        self.api.configure_modules()
        installer = self.api.get_installer_system()
        d, c = installer.documents_processor()
        print(c)
if __name__ == '__main__':
    unittest.main()