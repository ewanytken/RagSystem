from typing import Dict, List, Optional, Any

from rich.table import Table
from rich.console import Console
from sentence_transformers import SentenceTransformer

from app.respondent.abstract_respondent import Respondent
from app.utils import Utils

console = Console()

class Metrics:

    def __init__(self):
        self.score: Optional[Dict] = {}
        self.candidates: List = []
        self.relevant_docs: List = []
        self.config: Optional[Dict] = Utils.get_config_file()
        self.model_sim: Optional[Any] = None
        self.model_judge: Optional[Respondent] = None
        self.lang: Optional[str] = "en"

    def dataset_and_init_processing(self) -> None:
        if self.config["metrics"]["model_sim"]:
            ticket = self.config["metrics"]["model_sim"]
            self.model_sim = SentenceTransformer(ticket)

        # TODO dataset processor
        datasets = self.config["metrics"]["datasets"]
        self.candidates = self.config["metrics"]
        self.relevant_docs = self.config["metrics"]

    def show_scores(self) -> None:
        table = Table(title="RAG System Configuration Summary", border_style="cyan")
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

    def set_candidates(self, candidates) -> None:
        self.candidates = candidates

    def set_relevant_docs(self, docs: List) -> None:
        self.relevant_docs = docs

    def get_relevant_docs(self) -> List[str]:
        return self.relevant_docs

    def get_candidates(self) -> List[str]:
        return self.candidates

    def set_model_sim(self, model) -> None:
        self.model_sim = model
