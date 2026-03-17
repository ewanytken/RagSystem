import questionary
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from app.common.api_call import ApiCall
from app.logger import LoggerWrapper

logger = LoggerWrapper()

console = Console()

def run_rag_system() -> None:
    api = ApiCall()

    is_init_metrics = questionary.confirm(
        "Do you need Metrics Calculation? (False by default)",
        default=False
    ).ask()

    if is_init_metrics:
        api.init_metrics(is_init_metrics)

    console.print("\n[bold blue] Enter your queries (type 'exit' to quit)[/bold blue]")

    while True:
        query = Prompt.ask("\n\t[bold blue] Send query to RAG System[/bold blue]")

        if query.lower() in ['exit', 'quit']:
            break

        api.set_query(query)
        response = api.run_interactive()

        console.print("\n[bold green] Response from RAG System[/bold green]")
        console.print(Panel(response, title="Response", border_style="green"))

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RAG System Interactive")
    parser.add_argument('--run', action='store_true', help='Run the CLI in current window')

    args = parser.parse_args()

    if args.run:
        run_rag_system()
    else:
        console.print("[bold]RAG System [/bold]")
        console.print("python main.py --run        # Run in current window")

if __name__ == "__main__":
    main()