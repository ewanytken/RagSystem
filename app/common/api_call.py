from typing import Optional, List, Dict

from rich.console import Console

from app.common.constructor_interface import Constructor
from app.common.installer_system import InstallerSystem
from app.logger import LoggerWrapper
from app.logger.logger_metrics import LoggerMetrics
from metrics.metrics_executor import MetricsExecutor

console = Console()
logger = LoggerWrapper()
logger_metrics = LoggerMetrics()

"""
# - def documents_processor(self)
# - def indexer_installer_processor(self, documents)

# - def extractor_processor(self, documents: List[str]) -> set

# - def indexer_query(self, query) -> tuple[list[dict], list[str]]
# - def find_entities_from_graph(self, indexer_doc: str) -> list[dict]
# def find_docs_from_graph(self, entity: str) -> set[dict[str, str | int]]

# - def find_triplets(self, query: str) -> list[dict]
# - def find_triplets_by_subject(self, query: str) -> list[dict]

# - def prompt_processor(self, query: str, chunks: List, entities: List, triplets: List = None) -> str
# - def llm_model_processor(self, prompt: str) -> str
"""

class ApiCall(Constructor):

    def __init__(self):
        super().__init__()
        self.complete_installer: Optional[InstallerSystem] = None

        self.query: Optional[str] = None
        self.doc_dict_retrieved: Optional[List[Dict]] = None
        self.doc_text_retrieved: Optional[List[str]] = None
        self.entities_by_document_search: Optional[List[Dict]] = None
        self.extracted_triplets: Optional[List[Dict]] = None
        self.response: Optional[str] = None
        self.all_metrics_scores: Optional[Dict] = None

        self.metrics_executor = MetricsExecutor()

        console.print("\n[bold cyan] Building RAG System[/bold cyan]")
        self.configure_modules()

        try:
            self.complete_installer = self.get_installer_system()
        except Exception as e:
            logger(f"Cannot get installer system or installer not complete [[122]] {e}")

    def run_interactive(self):

        console.print("\n[bold cyan] Load Documents and chinking them[/bold cyan]")
        chunk, doc  = self.complete_installer.documents_processor()

        console.print("\n[bold yellow] Indexing documents with embedding model[/bold yellow]")
        self.complete_installer.indexer_installer_processor(chunk)

        if self.complete_installer.get_extractors():
            console.print("\n[bold yellow] Extracting Entities and make Graphs[/bold yellow]")
            self.complete_installer.extractor_processor(chunk)

        try:
            if self.get_query():
                console.print("\n[bold red] Query obtain[/bold red]")
                console.print("\n[bold yellow] Retrieving documents from Indexer[/bold yellow]")

                self.doc_dict_retrieved, self.doc_text_retrieved = self.complete_installer.indexer_query(self.query)

                if self.doc_text_retrieved and self.complete_installer.get_extractors():
                    console.print("\n[bold green] Find entities by extracted documents in Entity Graph[/bold green]")
                    self.entities_by_document_search = self.complete_installer.find_entities_from_graph(self.doc_text_retrieved)

                if self.complete_installer.get_triplet_graph():
                    console.print("\n[bold cyan] Find triplets by Obtain Query in Triplets Graph[/bold cyan]")

                    triplets_by_full_query = self.complete_installer.find_triplets(self.query)
                    triplets_by_subject = self.complete_installer.find_triplets_by_subject(self.query)
                    logger(f"Number of retrieved triplets by full query: {len(triplets_by_full_query)}")
                    logger(f"Number of retrieved triplets by subject: {len(triplets_by_subject)}")

                    if triplets_by_full_query or triplets_by_subject:
                        all_triplets = triplets_by_full_query + triplets_by_subject
                        self.extracted_triplets = [dict(t) for t in set(frozenset(d.items()) for d in all_triplets)]

                console.print("\n[bold red] Assembling final prompt from all available information[/bold red]")
                assembled_prompt: Optional[str] = self.complete_installer.prompt_processor(self.query,
                                                                                           self.doc_dict_retrieved,
                                                                                           self.entities_by_document_search,
                                                                                           self.extracted_triplets)

                console.print("\n[bold red] Request to LLM Respondent[/bold red]")
                logger(f"First 300 symbols from Assembled prompt: {assembled_prompt[:300]}")

                self.response = self.complete_installer.llm_model_processor(assembled_prompt)
                logger(f"Length of output symbols: {len(self.response)}")
        except Exception as e:
            logger(f"Answer don't obtain. System don't work correctly [[123]] {e}")

        self.metric_processor()
        return self.response

    def metric_processor(self):
        try:
            if self.metrics_config['init_metrics']:
                console.print("\n[bold blue] Simple Metrics Calculation in processing ... [/bold blue]")

                config_eval = { "response": self.get_response(), #str
                                "query": self.get_query(), #str
                                "context": self.get_doc_text_retrieved(), #List[str]
                                "judge_model": self.metrics_config.get("judge_model", "empty_model")}

                self.metrics_executor.set_config_eval(config_eval)
                self.metrics_executor.generation_evaluator()
                self.metrics_executor.retriever_evaluator()

                if self.metrics_config['judge_metrics']:
                    console.print("\n[bold cyan] Metrics LLM Judge in processing ... [/bold cyan]")
                    self.metrics_executor.judge_evaluator()

                self.all_metrics_scores = self.metrics_executor.get_overall_scores()
                logger_metrics(f"All obtained metrics: {self.all_metrics_scores}")
            else:
                logger(f"Metrics doesn't calculate")
        except Exception as e:
            logger(f"Metrics processing ERROR: {e}")

    def get_response(self) -> str:
        return self.response

    def get_doc_text_retrieved(self) -> List[str]:
        return self.doc_text_retrieved

    def set_query(self, query: str) -> None:
        self.query = query

    def get_query(self) -> str:
        return self.query

    def get_metric_executor(self) -> MetricsExecutor | None:
        if self.metrics_executor:
            return self.metrics_executor
        else:
            logger(f"Metrics Executor doesn't initialize")
            return None