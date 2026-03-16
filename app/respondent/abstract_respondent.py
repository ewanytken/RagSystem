from abc import abstractmethod, ABC
from typing import Optional, Dict, Any
from app.utils import Utils

class Respondent(ABC):

    def __init__(self):
        pass

    def __repr__(self):
        return f"Respondent Component"

    @abstractmethod
    def generate(self, prompt: str, *kwargs) -> str:
        raise NotImplemented
