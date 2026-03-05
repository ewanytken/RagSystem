from abc import ABC, abstractmethod
from typing import Dict, List


class AbstractPrompt(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def make_final_prompt(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_final_prompt(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def set_config(self, config: Dict):
        raise NotImplementedError

    @abstractmethod
    def set_chunks(self, chunks: List):
        raise NotImplementedError

    @abstractmethod
    def set_triplet(self, triplet: List):
        raise NotImplementedError

    @abstractmethod
    def set_entities(self, entities: List):
        raise NotImplementedError

    @abstractmethod
    def set_query(self, query: str):
        raise NotImplementedError
