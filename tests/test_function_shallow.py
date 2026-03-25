import json
import os
import unittest
from pathlib import Path

import pandas as pd

from app.logger import LoggerWrapper
from app.utils import Utils

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_document_processing(self):
        Utils.convert_datasets_to_csv()

    if __name__ == '__main__':
        unittest.main()