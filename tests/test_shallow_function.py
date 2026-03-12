import unittest

from transformers import AutoConfig

from app.logger import LoggerWrapper
from app.utils import Utils

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        self.config = Utils.get_config_file()

    def get_max_context_length(self, model_name):

        try:
            config = AutoConfig.from_pretrained(model_name)

            if hasattr(config, "max_position_embeddings"):
                return config.max_position_embeddings
            elif hasattr(config, "model_max_length"):
                if config.model_max_length > 1e9:
                    return "Very large (check max_position_embeddings if available)"
                return config.model_max_length
            else:
                return "Max context length attribute not found in config."

        except Exception as e:
            return f"Error loading config: {e}"

    def test_document_processing(self):
        print(self.get_max_context_length("Qwen/Qwen3-4B-Instruct-2507"))

if __name__ == '__main__':
    unittest.main()