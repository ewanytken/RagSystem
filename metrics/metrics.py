from typing import Dict, List, Optional, Any

from rich.table import Table
from rich.console import Console
from sentence_transformers import SentenceTransformer

from app.respondent.abstract_respondent import Respondent
from app.utils import Utils
from metrics.dataset_handler.dataset_handler import DatasetHandler

console = Console()

class Metrics:

    def __init__(self):
        self.score: Optional[Dict] = {}

        self.candidate: str = ""
        self.relevant_doc: Optional[List | str] = []

        self.config: Optional[Dict] = Utils.get_config_file()

        self.model_sim: Optional[Any] = None
        self.model_judge: Optional[Respondent] = None

    def init_processing(self) -> None:
        if self.config["metrics"]["model_sim"]:
            ticket = self.config["metrics"]["model_sim"]
            self.model_sim = SentenceTransformer(ticket)

    def show_scores(self) -> None:
        table = Table(title="RAG Metrics Summary", border_style="cyan")
        table.add_column("Number", style="yellow")
        table.add_column("Metrics", style="blue")
        table.add_column("Scores", style="green")

        for i, (key, value) in enumerate(self.score.items(), 1):
            if value:
                table.add_row(
                    str(i),
                    key,
                    str(value)
                )
        console.print(table)

    def get_score(self) -> Dict[str, float | Dict[str, float]]:
        return self.score

    def set_candidates(self, candidate: str) -> None:
        self.candidate = candidate

    def set_relevant_docs(self, docs: List | str) -> None:
        self.relevant_doc = docs

    def get_relevant_doc(self) -> List[str]:
        return self.relevant_doc

    def get_candidate(self) -> List[str] | str:
        return self.candidate

    def set_model_sim(self, model) -> None:
        self.model_sim = model
