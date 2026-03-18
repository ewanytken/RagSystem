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
        self.triplets: Optional[List[Dict]] = None
        self.response: Optional[str] = None

        self.metrics_config: Optional[Dict] = None

        console.print("\n[bold cyan] Building RAG System[/bold cyan]")
        self.configure_modules()

        try:
            self.complete_installer = self.get_installer_system()
            self.metrics_config = self.get_metrics_config()
        except Exception as e:
            logger(f"Cannot get installer system or installer not complete [[122]] {e}")

    def run_interactive(self):

        console.print("\n[bold cyan] Load Documents and chinking them[/bold cyan]")
        doc, chunk = self.complete_installer.documents_processor()

        console.print("\n[bold cyan] Indexing documents with embedding model[/bold cyan]")
        self.complete_installer.indexer_installer_processor(chunk)

        if self.complete_installer.get_extractors():
            console.print("\n[bold cyan] Extracting Entities and make Graphs[/bold cyan]")
            self.complete_installer.extractor_processor(chunk)

        try:
            if self.get_query():
                console.print("\n[bold green] Query obtain[/bold green]")
                console.print("\n[bold green] Retrieving documents from Indexer[/bold green]")

                self.doc_dict_retrieved, self.doc_text_retrieved = self.complete_installer.indexer_query(self.query)

                if self.doc_text_retrieved and self.complete_installer.get_extractors():
                    console.print("\n[bold cyan] Find entities by extracted documents in Entity Graph[/bold cyan]")
                    self.entities_by_document_search = self.complete_installer.find_entities_from_graph(self.doc_text_retrieved)

                if self.complete_installer.get_triplet_graph():
                    console.print("\n[bold cyan] Find triplets by Obtain Query in Triplets Graph[/bold cyan]")

                    triplets_by_full_query = self.complete_installer.find_triplets(self.query)
                    triplets_by_subject = self.complete_installer.find_triplets_by_subject(self.query)
                    logger(f"Number of retrieved triplets by full query: {len(triplets_by_full_query)} \n"
                           f"Number of retrieved triplets by subject: {len(triplets_by_subject)}")

                    if triplets_by_full_query or triplets_by_subject:
                        all_triplets = triplets_by_full_query + triplets_by_subject
                        self.triplets = [dict(t) for t in set(frozenset(d.items()) for d in all_triplets)]

                console.print("\n[bold cyan] Assembling final prompt from all available information[/bold cyan]")
                assembled_prompt: Optional[str] = self.complete_installer.prompt_processor(self.query,
                                                                                           self.doc_dict_retrieved,
                                                                                           self.entities_by_document_search,
                                                                                           self.triplets)

                console.print("\n[bold cyan] Request to LLM Respondent[/bold cyan]")
                logger(f"First 300 symbols from Assembled prompt: {assembled_prompt[:300]}")

                self.response = self.complete_installer.llm_model_processor(assembled_prompt)
                logger(f"Length of output symbols: {len(self.response)}")
        except Exception as e:
            logger(f"Answer don't obtain. System don't work correctly [[123]] {e}")

        self.metric_processor()
        return self.response

    def metric_processor(self):
        try:
            if self.metrics_config["init_metrics"]:
                console.print("\n[bold cyan] Simple Metrics Calculation in processing ... [/bold cyan]")
                response_by_word = self.get_response().split()
                query_by_word = self.get_query().split()
                overall_context = self.get_context()
                retrieved_docs = self.doc_text_retrieved

                config = {"response": response_by_word,
                          "query": query_by_word,
                          "context": overall_context,
                          "retrieved_docs": retrieved_docs,
                          "judge_model": self.metrics_config.get("judge_model", "empty_model")}

                metrics_executor = MetricsExecutor(config)
                metrics_executor.generation_evaluator()
                metrics_executor.retriever_evaluator()

                if self.metrics_config["judge_metrics"]:
                    console.print("\n[bold cyan] Metrics LLM Judge in processing ... [/bold cyan]")
                    metrics_executor.judge_evaluator()

                metrics_executor.get_overall_scores()
            else:
                logger(f"Pass metrics evaluation")
        except Exception as e:
            logger_metrics(f"Metrics processing ERROR: {e}")

    def get_context(self) -> List[str]:
        entities_context = ""
        if self.entities_by_document_search is not None:
            entities_context = "\nRelated entities from documents. Use it for more deeper answer to query:\n"
            for i, entity in enumerate(self.entities_by_document_search):
                entities_context += f"{i}. Entity: {entity['entity']} is label: {entity['label']} \n"

        triplet_context = ""
        if self.triplets is not None:
            documents_extracted_from_triplet = set()
            triplet_context = "Triplet extracted from documents:\n"
            for i, triplet in enumerate(self.triplets):
                triplet_context += f"{i}. {triplet['subject']} --> [{triplet['predicate']}]--> {triplet['object']} \n"
                documents_extracted_from_triplet.add(triplet['document'])
            triplet_context = triplet_context + "\n".join(documents_extracted_from_triplet)

        chunks_context = ""
        if self.doc_dict_retrieved is not None:
            chunks_context = "=== RETRIEVED DOCUMENT PASSAGES ===\n"
            for i, chunk in enumerate(self.doc_dict_retrieved, 1):
                relevance = chunk.get('score', 'N/A')
                if isinstance(relevance, float):
                    relevance = f"{relevance:.2f}"
                chunks_context += f"\n[Passage {i}] (Relevance: {relevance})\n"
                chunks_context += f"{chunk.get('text', '')}\n"

        overall_context = chunks_context + entities_context + triplet_context
        logger_metrics(f"Overall context for generate_metrics method: {overall_context}\n")
        return overall_context

    def get_response(self) -> str:
        return self.response

    def get_doc_text_retrieved(self) -> List[str]:
        return self.doc_text_retrieved

    def set_query(self, query: str) -> None:
        self.query = query

    def get_query(self) -> str:
        return self.query
