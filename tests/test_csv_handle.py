import unittest

from app.logger import LoggerWrapper
from app.utils import Utils
from metrics.dataset_handler.csv_handler import CSVHandler

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_document_processing(self):
        config = Utils.get_config_file()
        handler = CSVHandler()
        handler.set_config(config)
        handler.json_convert_to_csv()
        handler.show_dataset()
        handler.install_reference_values()

        print(len(handler.get_contexts()))
        print(len(handler.get_questions()))
        print(len(handler.get_golden_answers()))

        print(handler.data_frame['question'].tolist())

    if __name__ == '__main__':
        unittest.main()