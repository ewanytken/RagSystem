import unittest

import numpy as np
from bert_score import score
from huggingface_hub.cli.cache import verify
from nltk.translate.bleu_score import sentence_bleu
from app.logger import LoggerWrapper
from metrics.groundedness.ground_base import RuleBasedGroundedness

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_document_processing(self):

        response = "$22,500.00"
        query = "What is the total amount of the invoice?"
        response = ["$220000 dollars and zero cent".split()]
        candidate = "220000 dollars and zero cent".split()
        print(sentence_bleu(response,candidate))
        # P, R, F1 = score([candidate], [response], lang="en")
        # print(F1.mean().item())
        print("===========================================================")
        response = "$220000 dollars and zero cent. Description Front End Engineering Service $5000.00 "
        context = ["""Services Vendor Inc. 
                            100 Elm Street Pleasantville, NY 
                            TO Alpha Inc. 5900 1st Street Los Angeles, CA 
                            Description Front End Engineering Service $5000.00 
                             Back End Engineering Service $7500.00 
                             Quality Assurance Manager $10,000.00 
                             Total Amount $22,500.00 
                            Make all checks payable to Services Vendor Inc. Payment is due within 30 days.
                            If you have any questions concerning this invoice, contact Bia Hermes. 
                            THANK YOU FOR YOUR BUSINESS!  INVOICE INVOICE # 0001 DATE 01/01/2022 FOR Alpha Project P.O. # 1000"""]

        # Usage example
        rule_groundedness = RuleBasedGroundedness(threshold=0.3)
        result = rule_groundedness.evaluate(
            response=response,
            contexts=context
        )
        print(f"Groundedness score: {result.score:.2f} - {result.level.value}")
if __name__ == '__main__':
    unittest.main()