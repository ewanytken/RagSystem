import unittest

from app.logger import LoggerWrapper
from app.utils import Utils

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_document_processing(self):
        config = Utils.get_config_file("config_rus.yaml")

        completeness = Utils.load_template(config['metrics']['prompts']['completeness'])
        correctness = Utils.load_template(config['metrics']['prompts']['correctness'])
        faithfulness = Utils.load_template(config['metrics']['prompts']['faithfulness'])
        relevance = Utils.load_template(config['metrics']['prompts']['relevance'])
        template = Utils.load_template(config['metrics']['prompts']['groundedness'])
        prompt = Utils.load_template(config['templates']['prompt_template'])
        extr = Utils.load_template(config['graph']['extractor_prompt'])

        print(extr)
        print(prompt)
        print(completeness)
        print(correctness)
        print(faithfulness)
        print(relevance)
        print(template)

    if __name__ == '__main__':
        unittest.main()