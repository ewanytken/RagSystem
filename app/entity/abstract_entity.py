from abc import ABC, abstractmethod
from typing import List, Dict

class AbstractEntity(ABC):

    @abstractmethod
    def set_text_extraction(self, text_extraction: List[str]):
        pass

    @abstractmethod
    def get_extract_entities(self) -> List[Dict]:
        pass

    @abstractmethod
    def extractor_entity(self):
        pass