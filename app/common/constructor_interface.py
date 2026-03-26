from enum import Enum
from typing import Optional, Dict

import questionary
from questionary import Choice
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.common.installer_system import Builder, InstallerSystem
from app.documents_processor.word_handler import WordPdfHandler
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
from app.utils import Utils

console = Console()
logger = LoggerWrapper()

class ModelProvider(Enum):
    LOCAL = "local"
    REMOTE = "remote"

class ModelRemote(Enum):
    OLLAMA = "ollama"
    GIGA = "giga"
    EXTERNAL = "external_service"

class ModelLocalTicker(Enum):
    QWEN3 = "Qwen/Qwen3-4B-Instruct-2507"
    FELLADRIN = "Felladrin/TinyMistral-248M-Chat-v3"
    DEFAULT = "anticipate"

class PromptProvider(Enum):
    ADVANCED = "advanced"
    TEMPLATE = "templated"

class RemoteFreeModel(Enum):
    OPENAI = "openai/gpt-oss-120b"
    LLAMA = "meta-llama/llama-3.3-70b-instruct"
    QWEN = "qwen/qwen3-next-80b-a3b-instruct"
    GEMMA = "google/gemma-3-27b-it"
    STEPFUN = "stepfun/step-3.5-flash"
    HUNTER = "openrouter/hunter-alpha"
    DEFAULT = "default: load from config.yaml"


class Constructor:
    def __init__(self):
        self.word_handler: Optional[WordPdfHandler] = WordPdfHandler()
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

        self.metrics_config: Optional[Dict] = {"init_metrics": False}

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
            console.print("\n[bold yellow]🎯 Model for triplets construction[/bold yellow]")
            llm_triplet_extractor = self.model_chooser()
            self.triplets = TripletExtractor()
            self.triplets.set_llm_model(llm_triplet_extractor)
            self.installer_system_builder.set_triplet_graph(self.triplets)


        console.print("\n[bold yellow]🎯 Choose Prompt Assembler[/bold yellow]")
        self.prompter = self.prompt_chooser()
        self.installer_system_builder.set_prompt_object(self.prompter)


        console.print("\n[bold yellow]🎯 Choose Model Respondent[/bold yellow]")
        self.respondent = self.model_chooser()
        self.installer_system_builder.set_llm_responder(self.respondent)

        self.installer_system_builder = self.installer_system_builder.build()
        self.show_summary()

        self.metrics_config = self.metrics_initializer()

    def prompt_chooser(self) -> AbstractPrompt:
        prompter: Optional[AbstractPrompt] = None
        try:
            provider = questionary.select(
                "Select templated or advanced prompter:",
                choices=[
                    Choice(title="Advanced prompter", value=PromptProvider.ADVANCED),
                    Choice(title="Template prompter (for different language)", value=PromptProvider.TEMPLATE),
                ]
            ).ask()

            if provider == PromptProvider.ADVANCED:
                prompter = FinalAssembler()
            elif provider == PromptProvider.TEMPLATE:
                prompter = PromptObject()
            else:
                prompter = None

        except Exception as e:
            logger(f"Cannot assign prompt assembler [[120]] {e}")

        if prompter is None:
            raise Exception("Prompt assembler is None [[121]]")

        return prompter

    def model_chooser(self) -> Respondent:

        respondent: Optional[Respondent] = None
        try:
            provider = questionary.select(
                "Select local or remote model:",
                choices=[
                    Choice(title="Local model", value=ModelProvider.LOCAL),
                    Choice(title="Remote model", value=ModelProvider.REMOTE),
                ]
            ).ask()

            if provider == ModelProvider.REMOTE:
                service_name = questionary.select(
                    "Select type of remote model:",
                    choices=[
                        Choice(title="External model from service (Work with OpenAI Client)", value=ModelRemote.EXTERNAL),
                        Choice(title="Ollama local model", value = ModelRemote.OLLAMA),
                        Choice(title="GigaChat (specify parameters in config file)", value = ModelRemote.GIGA),
                    ]
                ).ask()

                if service_name == ModelRemote.OLLAMA:
                    respondent = OllamaModel()
                elif service_name == ModelRemote.EXTERNAL:
                    model_remote_ticket = questionary.select(
                        "Select OpenRouter's model or default remote model:",
                        choices=[
                            Choice(title="Open AI 120B OSS", value=RemoteFreeModel.OPENAI),
                            Choice(title="LLAMA 3.3 70B", value=RemoteFreeModel.LLAMA),
                            Choice(title="QWEN 3 80B", value=RemoteFreeModel.QWEN),
                            Choice(title="GEMMA 3 27B", value=RemoteFreeModel.GEMMA),
                            Choice(title="STEPFUN 3.5", value=RemoteFreeModel.STEPFUN),
                            Choice(title="OpRouter HUNTER", value=RemoteFreeModel.HUNTER),
                            Choice(title="default. Load from config.yaml", value=RemoteFreeModel.DEFAULT)
                        ]
                    ).ask()
                    if model_remote_ticket == RemoteFreeModel.DEFAULT:
                        respondent = ExternalModel()
                    else:
                        respondent = ExternalModel(model_remote_ticket.value)
                elif service_name == ModelRemote.GIGA:
                    respondent = TargetGiga()
                else:
                    respondent = None

            elif provider == ModelProvider.LOCAL:
                model_ticker = questionary.select(
                    "Select ticket for local model:",
                    choices=[
                        Choice(title="Qwen/Qwen3-4B-Instruct-2507", value = ModelLocalTicker.QWEN3),
                        Choice(title="Felladrin/TinyMistral-248M-Chat-v3", value = ModelLocalTicker.FELLADRIN),
                        Choice(title="default (load ticket from config file)", value = ModelLocalTicker.DEFAULT),
                    ]
                ).ask()
                if model_ticker == ModelLocalTicker.DEFAULT:
                    respondent = TransformerWrapper()
                else:
                    respondent = TransformerWrapper(model_ticker.value)
            else:
                respondent = None

        except Exception as e:
            logger(f"Cannot assign model [[110]] {e}")

        if respondent is None:
            raise Exception("Respondent model is None [[111]]")

        return respondent

    def metrics_initializer(self) -> Dict:

        is_init_metrics = questionary.confirm(
            "Do you need Metrics Calculation? (False by default)",
            default=False
        ).ask()

        metrics_config = {"init_metrics": is_init_metrics}
        if is_init_metrics:
            judge_metrics = questionary.confirm(
                "Do you need add Judge LLM Metrics? (False by default)",
                default=False
            ).ask()

            if judge_metrics:
                metrics_config.update({"judge_metrics": judge_metrics})
                try:
                    provider = questionary.select(
                        "Select LOCAL or REMOTE model:",
                        choices=[
                            Choice(title="Local metric model (default from config.yaml)", value=ModelProvider.LOCAL),
                            Choice(title="Remote metric model (default from config.yaml)", value=ModelProvider.REMOTE),
                        ]
                    ).ask()

                    config = Utils.get_config_file()
                    if provider == ModelProvider.LOCAL:
                        model = TransformerWrapper(config.get('metrics', {}).get("model_judge_local"))
                    elif provider == ModelProvider.REMOTE:
                        model = ExternalModel(config.get('metrics', {}).get("model_judge_remote"))
                    else:
                        model = None

                    if model is None:
                        raise Exception(f"Model for Metrics don't initialize")

                    metrics_config.update({"judge_model": model})

                except Exception as e:
                    logger(f"Model for Metrics don't install: {e}")
            else:
                metrics_config.update({"judge_metrics": False})

        return metrics_config

    def show_summary(self):

        table = Table(title="RAG System Configuration Summary", border_style="cyan")
        table.add_column("Number", style="cyan")
        table.add_column("Component", style="green")
        table.add_column("Status", style="yellow")

        components = {"Prompter component": self.prompter, "Respondent component": self.respondent, "Word(docx) component": self.word_handler,
                      "Regex component": self.regex, "Gliner component": self.gliner, "Gliner2 component": self.gliner_two,
                      "Indexer component": self.indexer,
                      "EntityGraph component": self.graph, "Triplets component": self.triplets}

        for i, (key, value) in enumerate(components.items(), 1):
            if value:
                table.add_row(
                    str(i),
                    value.__repr__(),
                    "✅ Configured"
                )
            else:
                table.add_row(
                    str(i),
                    key,
                    "❌ Not configured",
                )
        console.print(table)
