import unittest

from transformers import AutoConfig

from app.logger import LoggerWrapper
from app.respondent.external_model.respondent_other_service import ExternalModel
from app.utils import Utils

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        self.model = ExternalModel()

    def test_document_processing(self):
        print(self.model.generate("Hello World"))

if __name__ == '__main__':
    unittest.main()