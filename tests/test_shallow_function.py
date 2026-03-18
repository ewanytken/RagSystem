import unittest

from app.logger import LoggerWrapper

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_document_processing(self):
        string = "Simple Metrics Calculation in processing..."
        j = string.split()
        print(j)
        s = " ".join(j)
        print(s)
if __name__ == '__main__':
    unittest.main()