import unittest

import torch
from app.logger import LoggerWrapper
from app.respondent.local_model.transformer_wrapper import TransformerWrapper

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_document_processing(self):
        model = TransformerWrapper()
        logger(model.generate("Hello World"))

if __name__ == '__main__':
    unittest.main()