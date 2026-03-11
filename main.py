import asyncio
import os
import sys

from rich import Console
from rich.panel import Panel
from rich.prompt import Prompt

from app.common.complete_interface import ApiCall

console = Console()

def run_rag_system():
    cli = ApiCall()
    console.print("\n[bold green]RAG System Ready! Enter your queries (type 'exit' to quit)[/bold green]")

    while True:
        query = Prompt.ask("\n[bold cyan] query[/bold cyan]")

        if query.lower() in ['exit', 'quit']:
            break

        cli.set_query(query)
        with console.status("[bold green]Processing..."):
            response = asyncio.run(cli.run_interactive())

        console.print(Panel(response, title="Response", border_style="green"))

def main():
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