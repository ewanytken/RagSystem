import unittest

from app.logger import LoggerWrapper
from app.utils import Utils

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        self.config = Utils.get_config_file()


    def test_document_processing(self):
        print(Utils.load_template(self.config['metrics']['prompts']['completeness']))
        print(Utils.load_template(self.config['graph']['extractor_prompt']))
if __name__ == '__main__':
    unittest.main()