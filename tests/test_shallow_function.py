import unittest

from transformers import AutoConfig

from app.logger import LoggerWrapper
from app.respondent.external_model.respondent_other_service import ExternalModel
from app.utils import Utils

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_document_processing(self):
        pass

if __name__ == '__main__':
    unittest.main()