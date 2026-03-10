import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
import questionary
from accelerate.test_utils.scripts.external_deps.test_ds_alst_ulysses_sp import model_name
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from questionary import Choice
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from app.common.installer_system import Builder
from app.documents_processor.word_handler import WordHandler
from app.entity.abstract_entity import AbstractEntity
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

console = Console()

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

        self.llm_extractor: Optional[Respondent] = None
        self.responder: Optional[Respondent] = None

        self.regex: Optional[AbstractEntity] = None
        self.gliner_two: Optional[AbstractEntity] = None
        self.gliner: Optional[AbstractEntity] = None

        self.triplets: Optional[TripletExtractor] = None

        # prompt_object - simple, prompt_assembler - advanced
        self.prompt_advanced: Optional[AbstractPrompt] = None
        self.prompt_simple: Optional[AbstractPrompt] = None

        self.style = Style.from_dict({
            'prompt': 'ansicyan bold',
            'question': 'ansiyellow',
            'answer': 'ansigreen',
        })

    console.print("[bold green]✓ RAG System building [/bold green]")

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

        llm_triplet_extractor = None
        model_name = None

        add_triplets = questionary.confirm(
            "Do you need to add Triplets?",
            default=False
        ).ask()

        if add_triplets:
            provider = questionary.select(
                "Select local or remote model:",
                choices=[
                    Choice(title="local", value=ModelProvider.LOCAL),
                    Choice(title="remote", value=ModelProvider.REMOTE),
                ]
            ).ask()

            if provider == ModelProvider.LOCAL.value:
                model_name = questionary.select(
                        "Select model for triplet extraction:",
                        choices=[
                            Choice(title="Qwen/Qwen3-4B-Instruct-2507", value=ModelProvider.LOCAL),
                            Choice(title="Felladrin/TinyMistral-248M-Chat-v3", value=ModelProvider.LOCAL),
                        ]
                    ).ask()
            elif provider == ModelProvider.REMOTE.value:
                # TODO add remote model
                model_name = "Qwen/Qwen3-4B-Instruct-2507"
                llm_triplet_extractor = TransformerWrapper()
                llm_triplet_extractor.set_model_name(model_name)
            else:
                model_name = None

            self.triplets = TripletExtractor()
            if llm_triplet_extractor:
                self.triplets.set_llm_model(llm_triplet_extractor)
            else:
                raise Exception("Model for triplet extraction is not available")
            self.installer_system_builder.set_triplet_graph(self.triplets)

    async def configure_llm(self):
        console.print("\n[bold yellow]🎯 Respondent LLM Configurator[/bold yellow]")

        provider = questionary.select(
            "Select LLM provider:",
            choices=[
                Choice(title="🤖 OpenAI (GPT-4, GPT-3.5)", value=ModelProvider.OPENAI),
                Choice(title="🎨 Anthropic (Claude)", value=ModelProvider.ANTHROPIC),
                Choice(title="📚 Cohere", value=ModelProvider.COHERE),
                Choice(title="🏠 Ollama (Local)", value=ModelProvider.OLLAMA),
            ]
        ).ask()

        # Model selection based on provider
        if provider == ModelProvider.OPENAI.value:
            model_name = questionary.select(
                "Select model:",
                choices=[
                    "gpt-4-turbo-preview",
                    "gpt-4",
                    "gpt-3.5-turbo"
                ]
            ).ask()
        elif provider == ModelProvider.ANTHROPIC.value:
            model_name = questionary.select(
                "Select model:",
                choices=[
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-2.1"
                ]
            ).ask()
        elif provider == ModelProvider.OLLAMA.value:
            model_name = questionary.text(
                "Enter Ollama model name:",
                default="llama2"
            ).ask()
        else:
            model_name = questionary.text(
                "Enter model name:",
                default="command"
            ).ask()

    def show_summary(self):
        components = self.builder.components

        table = Table(title="RAG System Configuration Summary", border_style="cyan")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")

        for component_name, component in components.items():
            if component:
                if component_name == "retriever":
                    details = f"Type: {component['type'].value}"
                elif component_name in ["embedding_model", "llm"]:
                    details = f"Provider: {component['provider'].value}\nModel: {component['model_name']}"
                elif component_name == "vector_store":
                    details = f"Type: {component['type']}"
                elif component_name == "chunker":
                    details = f"Size: {component['chunk_size']}, Overlap: {component['chunk_overlap']}"
                else:
                    details = "Configured"

                table.add_row(
                    component_name.replace('_', ' ').title(),
                    "✅ Configured",
                    details
                )
            else:
                table.add_row(
                    component_name.replace('_', ' ').title(),
                    "❌ Not configured",
                    "Optional"
                )

        console.print(table)