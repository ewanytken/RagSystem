from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from app.common.api_call import ApiCall
from app.logger import LoggerWrapper, LoggerAuxiliary
from app.utils import Utils
from metrics.dataset_handler.csv_handler import CSVHandler

logger = LoggerWrapper()
logger_auxiliary = LoggerAuxiliary()

console = Console()

def start_rag_system() -> None:
    api = ApiCall()

    console.print("\n[bold blue] Enter your queries (type 'exit' to quit)[/bold blue]")

    while True:
        query = Prompt.ask("\n\t[bold blue] Send query to RAG System[/bold blue]")

        if query.lower() in ['exit', 'quit']:
            break

        api.set_query(query)
        response = api.run_interactive()

        console.print("\n[bold green] Response from RAG System[/bold green]")
        console.print(Panel(response, title="Response", border_style="green"))

def auto_metrics() -> None:

    config = Utils.get_config_file()
    dataset_handler = CSVHandler()
    dataset_handler.set_config(config)
    dataset_handler.json_convert_to_csv()
    dataset_handler.show_dataset()
    dataset_handler.install_reference_values()

    console.print("\n[bold red] Be sure to select METRICS!!! [/bold red]")

    api = ApiCall()

    if api.get_metric_executor() is None:
        raise Exception("Mode with metrics don't select or MetricExecutor don't exist")

    metric_executor = api.get_metric_executor()

    golden_answers = dataset_handler.get_golden_answers()
    relevant_contexts = dataset_handler.get_contexts()
    query = dataset_handler.get_questions()

    if len(relevant_contexts) == len(query) == len(golden_answers):
        for query, answer, context in zip(query, golden_answers, relevant_contexts):
            metric_executor.set_candidate(answer)
            metric_executor.set_relevant_context(context)

            # must be last
            api.set_query(query)

            response = api.run_interactive()
            logger(f"Response from RAG: {response[:15]}")
            logger_auxiliary(f"Dataset Query: {query}")
            logger_auxiliary(f"Relevant Context: {context}")
            logger_auxiliary(f"Golden Answer: {answer}")
            logger_auxiliary(f"RAG Response: {context}")
    else:
        logger(f"Dataset loaded or made not correct: {len(relevant_contexts)}, {len(query)}, {len(golden_answers)}")

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RAG System Interactive")

    parser.add_argument('--run', action='store_true', help='Run Interactive mode (metrics with dataset don t work')
    parser.add_argument('--autometrics', action='store_true', help='Run Autometrics mode with dataset load')

    args = parser.parse_args()

    if args.run:
        start_rag_system()
    elif args.autometrics:
        auto_metrics()
    else:
        console.print("[bold]RAG System [/bold]")
        console.print("python main.py --run           # Run in interactive mode")
        console.print("python main.py --autometrics   # Run in metrics mode")


if __name__ == "__main__":
    main()