import unittest

import torch
from gliner import GLiNER
from gliner2 import GLiNER2

from app.logger import LoggerWrapper
from app.utils import Utils

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_document_processing(self):
        config = Utils.get_config_file()
        model_ticker = config['gliner2']['ticket']

        self.gliner2 = GLiNER2.from_pretrained(
            model_ticker
        )
        self.gliner2.to("cuda")
        print(f"Gliner 2: {print(next(self.gliner2.parameters())[-1].device)}")

        self.gliner = GLiNER.from_pretrained(
            config['gliner']['ticket'],
        )
        self.gliner.to("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Gliner : {print(next(self.gliner.parameters())[-1].device)}")

    if __name__ == '__main__':
        unittest.main()