from typing import Dict

from rich.table import Table
from rich.console import Console

console = Console()

class Metrics:
    def __init__(self):
        self.score = {}

    def show_scores(self) -> None:
        table = Table(title="RAG System Configuration Summary", border_style="cyan")
        table.add_column("Metrics", style="blue")
        table.add_column("Scores", style="green")

        for key, value in self.score.items():
            if value:
                table.add_row(
                    key,
                    str(value)
                )
        console.print(table)

    def get_score(self) -> Dict[str, float | Dict[str, float]]:
        return self.score