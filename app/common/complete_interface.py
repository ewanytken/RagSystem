import asyncio
from typing import Optional, List, Dict
from rich import Console
from app.common.constructor_interface import Constructor
from app.common.installer_system import InstallerSystem
from app.logger import LoggerWrapper

console = Console()
logger = LoggerWrapper()

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

    def set_query(self, query: str) -> None:
        self.query = query

    def get_query(self) -> str:
        return self.query

    async def run_interactive(self):

        answer: Optional[str] = None

        await self.configure_modules()
        try:
            self.complete_installer = self.get_installer_system()
        except Exception as e:
            logger(f"Cannot get installer system or installer not complete [[122]] {e}")

        doc, chunk = self.complete_installer.documents_processor()
        self.complete_installer.indexer_installer_processor(chunk)

        if self.complete_installer.get_extractors():
            self.complete_installer.extractor_processor(chunk)

        doc_dict_by_query: Optional[List[Dict]] = None
        doc_str_only_by_query: Optional[List[str]] = None
        entities_by_document_search: Optional[List[Dict]] = None
        triplets_by_full_query: Optional[List[Dict]] = None
        triplets_by_subject: Optional[List[Dict]] = None
        triplets: Optional[List[Dict]] = None
        try:
            if self.get_query():
                doc_dict_by_query, doc_str_only_by_query = self.complete_installer.indexer_query(self.query)
                if doc_str_only_by_query and self.complete_installer.get_entities_graph():
                    entities_by_document_search = self.complete_installer.find_entities_from_graph(doc_str_only_by_query)

                if self.complete_installer.get_triplet_graph():
                    triplets_by_full_query = self.complete_installer.find_triplets(self.query)
                    triplets_by_subject = self.complete_installer.find_triplets_by_subject(self.query)

                    if triplets_by_full_query and triplets_by_subject:
                        triplets = triplets_by_full_query + triplets_by_subject
                    else:
                        triplets = triplets_by_subject if triplets_by_subject else triplets_by_full_query

                assembled_prompt: Optional[str] = self.complete_installer.prompt_processor(self.query, chunk, entities_by_document_search, triplets)
                answer = self.complete_installer.llm_model_processor(assembled_prompt)

        except Exception as e:
            logger(f"Answer don't obtain. System don't work correctly [[123]] {e}")

        if answer:
            raise Exception(f"Answer don't assign")

        return answer
