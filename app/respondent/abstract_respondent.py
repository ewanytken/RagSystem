from abc import abstractmethod, ABC
from typing import Optional, Dict, Any


class Respondent(ABC):

    def __init__(self):
        self.config: Optional[Dict[str, Any]] = None

    def set_config(self, config: Dict) -> None:
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, *kwargs) -> str:
        raise NotImplemented
