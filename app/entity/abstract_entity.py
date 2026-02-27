from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class AbstractEntity(ABC):

    config: Optional[Dict] = None

    def set_config(self, config: Dict):
        self.config = config

    @abstractmethod
    def set_text_extraction(self, text_extraction: List[str]):
        pass

    @abstractmethod
    def get_extract_entities(self) -> List[Dict]:
        pass

    @abstractmethod
    def extractor_entity(self):
        pass

