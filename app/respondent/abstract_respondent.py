from abc import abstractmethod
from typing import Optional, Dict, Any


class Respondent:

    config: Optional[Dict[str, Any]] = None

    def set_config(self, config) -> None:
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, *kwargs) -> str:
        raise NotImplemented
