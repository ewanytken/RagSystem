import unittest

from app.logger import LoggerWrapper, LoggerAuxiliary
from app.logger.logger_metrics import LoggerMetrics

logger = LoggerWrapper()
loggerA = LoggerAuxiliary()
loggerB = LoggerMetrics()

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_document_processing(self):
        logger(f"Wrapper")
        loggerA(f"Auxiliary")
        loggerB(f"Metrics")

if __name__ == '__main__':
    unittest.main()