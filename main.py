from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from app.common.complete_interface import ApiCall

console = Console()

def run_rag_system() -> None:
    response: Optional[str] = None
    cli = ApiCall()
    console.print("\n[bold green] RAG System Ready! Enter your queries (type 'exit' to quit)[/bold green]")

    while True:
        query = Prompt.ask("\n[bold cyan] Send query[/bold cyan]")

        if query.lower() in ['exit', 'quit']:
            break

        cli.set_query(query)
        # with console.status("[bold green]Processing..."):
        response = cli.run_interactive()

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
        # Default: show help
        console.print("[bold]RAG System [/bold]")
        console.print("python main.py --run        # Run in current window")


if __name__ == "__main__":
    main()