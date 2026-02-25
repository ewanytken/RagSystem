from app.documents_processor.word_handler import WordHandler
from app.utils import Utils


class UnionFacade:
    def __init__(self):
        self.word_handler = WordHandler()
        self.word_handler.set_config(Utils.get_config_file())
        self.word_handler.handle_documents()
        self.word_handler.get_chunked_documents()
        self.word_handler.get_handled_documents()