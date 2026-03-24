import unittest

from rich.console import Console

from app.logger import LoggerWrapper
from app.respondent.external_model.respondent_other_service import ExternalModel
from app.respondent.local_model.transformer_wrapper import TransformerWrapper

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_document_processing(self):
        console = Console()
        console.print("[bold]RAG System [/bold]")
        is_init_metrics_question = False

        metrics_config = {"init_metrics": is_init_metrics_question}
        try:
            provider = "Local"

            if provider == "Local":
                model = TransformerWrapper()
            elif provider == "Remote":
                model = ExternalModel()
            else:
                model = None

            metrics_config.update({"model": model})
        except Exception as e:
            logger(f"Model for Metrics don't install: {e}")

        logger(metrics_config["init_metrics"])
        logger(metrics_config["model"])

if __name__ == '__main__':
    unittest.main()