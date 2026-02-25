from abc import ABC, abstractmethod
from typing import Dict, List

"""
Abstract class for different type of documents (Word, Pdf and other)
"""

class DocumentHandler(ABC):

    @abstractmethod
    def set_config(self, config: Dict):
        pass

    @abstractmethod
    def handle_documents(self):
        pass

    @abstractmethod
    def get_handled_documents(self) -> List[str]:
        pass