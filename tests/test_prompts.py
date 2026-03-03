import unittest

from app.logger import LoggerWrapper
from app.prompt.prompt_object import PromptObject, PromptObjectBuilder
from app.utils import Utils

logger = LoggerWrapper()

class TestRAGSystem(unittest.TestCase):

    def setUp(self):
        self.builder = PromptObjectBuilder()

    def test_document_processing(self):

        temp = Utils.load_template("extraction_template_eng")
        # print(temp)
        entities = [{'label': 111, 'entity': 222, 'score': 333}]
        prompt_object = self.builder.set_query("SOME QUERY").set_context("SOME CONTEXT").set_entities(entities).set_path_to_template("prompt_template").build()
        prompt_object.set_prompt()
        logger(prompt_object.get_prompt())




if __name__ == '__main__':
    unittest.main()