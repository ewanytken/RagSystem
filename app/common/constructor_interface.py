from enum import Enum
from enum import Enum
from typing import Optional, Union

import questionary
from questionary import Choice
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.common.installer_system import Builder, InstallerSystem
from app.documents_processor.word_handler import WordHandler
from app.entity.abstract_entity import AbstractEntity
from app.entity.gliner2_entity import GlinerTwoEntity
from app.entity.gliner_entity import GlinerEntity
from app.entity.regex_entity import RegexEntity
from app.graph.graph_entity import GraphEntity
from app.graph.triplet_extractor import TripletExtractor
from app.indexer.indexer_object import Indexer
from app.logger import LoggerWrapper
from app.prompt.abstract_prompt import AbstractPrompt
from app.prompt.prompt_assembler import FinalAssembler
from app.prompt.prompt_object import PromptObject
from app.respondent.abstract_respondent import Respondent
from app.respondent.external_model.respondent_giga import TargetGiga
from app.respondent.external_model.respondent_ollama import OllamaModel
from app.respondent.external_model.respondent_other_service import ExternalModel
from app.respondent.local_model.transformer_wrapper import TransformerWrapper

console = Console()
logger = LoggerWrapper()

class GraphType(Enum):
    TRIPLETS = "triplets"
    GRAPH = "graph"

class Entities(Enum):
    REGEX = "regex"
    GLINER = "gliner"
    GLINER2 = "gliner2"

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

    def get_installer_system(self) -> Union[Builder, InstallerSystem]:
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

    async def configure_modules(self):
        console.print("\n[bold blue]📚 Modules Configurator[/bold blue]")

        console.print("\n[bold yellow]🎯 Regex Module[/bold yellow]")
        add_regex = questionary.confirm(
            "Do you need to add Regex extractor?",
            default=False
        ).ask()
        if add_regex:
            self.regex = RegexEntity()
            self.installer_system_builder.set_entities_extractors(self.regex)


        console.print("\n[bold yellow]🎯 Gliner Module[/bold yellow]")
        add_gliner = questionary.confirm(
            "Do you need to add Gliner extractor?",
            default=False
        ).ask()
        if add_gliner:
            self.gliner = GlinerEntity()
            self.installer_system_builder.set_entities_extractors(self.gliner)


        console.print("\n[bold yellow]🎯 Gliner2 Module[/bold yellow]")
        add_gliner_two = questionary.confirm(
            "Do you need to add Gliner2 extractor?",
            default=False
        ).ask()
        if add_gliner_two:
            self.gliner_two = GlinerTwoEntity()
            self.installer_system_builder.set_entities_extractors(self.gliner_two)


        console.print("\n[bold yellow]🎯 Triplets Module[/bold yellow]")
        add_triplets = questionary.confirm(
            "Do you need to add Triplets?",
            default=False
        ).ask()
        if add_triplets:
            llm_triplet_extractor = await self.model_chooser()
            self.triplets = TripletExtractor()
            self.triplets.set_llm_model(llm_triplet_extractor)
            self.installer_system_builder.set_triplet_graph(self.triplets)


        console.print("\n[bold yellow]🎯 Choose Prompt Assembler[/bold yellow]")
        self.prompter = await self.prompt_chooser()
        self.installer_system_builder.set_prompt_object(self.prompter)


        console.print("\n[bold yellow]🎯 Choose Model Respondent[/bold yellow]")
        self.respondent = await self.model_chooser()
        self.installer_system_builder.set_llm_responder(self.respondent)

        await self.construction()
        self.show_summary()

    async def construction(self) -> None:
        self.installer_system_builder.build()

    async def prompt_chooser(self) -> AbstractPrompt:
        prompter: Optional[AbstractPrompt] = None
        try:
            provider = questionary.select(
                "Select simple or advanced prompter:",
                choices=[
                    Choice(title="Advanced prompter", value=PromptProvider.ADVANCED),
                    Choice(title="Simple prompter", value=PromptProvider.SIMPLE),
                ]
            ).ask()

            if provider == PromptProvider.ADVANCED:
                prompter = FinalAssembler()
            elif provider == PromptProvider.SIMPLE:
                prompter = PromptObject()
            else:
                prompter = None

        except Exception as e:
            logger(f"Cannot assign prompt assembler [[120]] {e}")

        if prompter is None:
            raise Exception("Prompt assembler is None [[121]]")

        return prompter

    async def model_chooser(self) -> Respondent:

        respondent: Optional[Respondent] = None
        try:
            provider = questionary.select(
                "Select local or remote model:",
                choices=[
                    Choice(title="Local model", value=ModelProvider.LOCAL),
                    Choice(title="Remote model", value=ModelProvider.REMOTE),
                ]
            ).ask()

            if provider == ModelProvider.REMOTE.value:
                service_name = questionary.select(
                    "Select remote model:",
                    choices=[
                        Choice(title="Ollama local model", value = ModelRemote.OLLAMA),
                        Choice(title="External model from service", value = ModelRemote.EXTERNAL),
                        Choice(title="GigaChat", value = ModelRemote.GIGA),
                    ]
                ).ask()

                if service_name == ModelRemote.OLLAMA.value:
                    respondent = OllamaModel()
                elif service_name == ModelRemote.EXTERNAL.value:
                    respondent = ExternalModel()
                elif service_name == ModelRemote.GIGA.value:
                    respondent = TargetGiga()
                else:
                    respondent = None

            elif provider == ModelProvider.LOCAL.value:
                model_ticker = questionary.select(
                    "Select ticker for local model:",
                    choices=[
                        Choice(title="Qwen/Qwen3-4B-Instruct-2507", value = ModelLocalTicker.QWEN3),
                        Choice(title="Felladrin/TinyMistral-248M-Chat-v3", value = ModelLocalTicker.FELLADRIN),
                        Choice(title="default", value = ModelLocalTicker.DEFAULT),
                    ]
                ).ask()
                respondent = TransformerWrapper(model_ticker)
            else:
                respondent = None

        except Exception as e:
            logger(f"Cannot assign model [[110]] {e}")

        if respondent is None:
            raise Exception("Respondent model is None [[111]]")

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
