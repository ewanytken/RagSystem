import unittest

import torch
from app.logger import LoggerWrapper

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_document_processing(self):
        logger(torch.cuda.is_available())

if __name__ == '__main__':
    unittest.main()