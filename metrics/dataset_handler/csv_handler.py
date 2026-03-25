from app.logger.logger_metrics import LoggerMetrics
from metrics.dataset_handler.dataset_handler import DatasetHandler

logger = LoggerMetrics()

class CSVHandler(DatasetHandler):
    def __init__(self):
        super().__init__()

        self.golden_answer = []
        self.context = []
        self.question = []

    def install_reference_values(self):
        if not self.data_frame.empty:
            self.question = self.data_frame['question'].tolist()
            self.golden_answer = self.data_frame['golden_answer'].tolist()
            self.context = self.data_frame['context'].tolist()

            logger(f"Extracted {len(self.question)} questions. Extracted Golden Answers {len(self.golden_answer)}. Extracted Contexts {len(self.context)} ")
        else:
            logger(f"DataFrame don't install: {self.data_frame}")

    def get_questions(self):
        if self.question:
            return self.question
        else:
            logger(f"Questions doesn't load")
            return []

    def get_golden_answers(self):
        if self.golden_answer:
            return self.golden_answer
        else:
            logger(f"Answers doesn't load")
            return []

    def get_contexts(self):
        if self.context:
            return self.context
        else:
            logger(f"Contexts doesn't load")
            return []

    def show_dataset(self):
        logger(f"Questions: {len(self.question)}")
        logger(f"Golden answers: {len(self.golden_answer)}")
        logger(f"Contexts: {len(self.context)}")
        logger(f"Path to dataset: {self.config["dataset"]["path_dataset"]}")
        logger(f"Name saved file: {self.config["dataset"]["file_save"]}")
        logger(f"Font format: {self.config["dataset"]["font_coding"]}")