import unittest

from rich.console import Console

from app.logger import LoggerWrapper
from app.logger.logger_metrics import LoggerMetrics
from app.respondent.external_model.respondent_other_service import ExternalModel
from app.respondent.local_model.transformer_wrapper import TransformerWrapper
from app.utils import Utils
from metrics.judge_metrics import JudgeMetrics
from metrics.metrics_executor import MetricsExecutor

console = Console()
logger = LoggerWrapper()
logger_metrics = LoggerMetrics()

class Test(unittest.TestCase):

    def setUp(self):
        self.config = Utils.get_config_file()

    def test_document_processing(self):
        console.print("\n[bold blue] Simple Metrics Calculation in processing ... [/bold blue]")

        response = "$22,500.00"
        query = "What is the total amount of the invoice?"
        candidate = "220000 dollars and zero cent"

        response_by_word = "$22,500.00".split()
        query_by_word = "What is the total amount of the invoice?".split()

        overall_context = """Services Vendor Inc. 
                            100 Elm Street Pleasantville, NY 
                            TO Alpha Inc. 5900 1st Street Los Angeles, CA 
                            Description Front End Engineering Service $5000.00 
                             Back End Engineering Service $7500.00 
                             Quality Assurance Manager $10,000.00 
                             Total Amount $22,500.00 
                            Make all checks payable to Services Vendor Inc. Payment is due within 30 days.
                            If you have any questions concerning this invoice, contact Bia Hermes. 
                            THANK YOU FOR YOUR BUSINESS!  INVOICE INVOICE # 0001 DATE 01/01/2022 FOR Alpha Project P.O. # 1000"""

        retrieved_docs = """Services Vendor Inc. 
                            100 Elm Street Pleasantville, NY 
                            TO Alpha Inc. 5900 1st Street Los Angeles, CA 
                            Description Front End Engineering Service $5000.00 
                             Back End Engineering Service $7500.00 
                             Quality Assurance Manager $10,000.00 
                             Total Amount $22,500.00 
                            Make all checks payable to Services Vendor Inc. Payment is due within 30 days.
                            If you have any questions concerning this invoice, contact Bia Hermes. 
                            THANK YOU FOR YOUR BUSINESS!  INVOICE INVOICE # 0001 DATE 01/01/2022 FOR Alpha Project P.O. # 1000"""

        relevant_docs = """Services Vendor Inc. 
                             Total Amount $22,500.00 
                            Make all checks payable to Services Vendor Inc. Payment is due within 30 days.
                            If you have any questions concerning this invoice, contact Bia Hermes. 
                            THANK YOU FOR YOUR BUSINESS!  INVOICE INVOICE # 0001 DATE 01/01/2022 FOR Alpha Project P.O. # 1000"""

        self.judge_metric = JudgeMetrics()
        model = ExternalModel()
        self.judge_metric.set_model_judge(model)
        self.judge_metric.set_contexts(retrieved_docs)
        self.judge_metric.set_queries(query)
        self.judge_metric.set_answers(response)
        self.judge_metric.judge_calculation()
        self.judge_metric.show_scores()

if __name__ == '__main__':
    unittest.main()