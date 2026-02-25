from abc import ABC, abstractmethod
from app.respondent.interface_respondent import Respondent

class AbstractModelExternal(ABC, Respondent):

    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    def set_base_url(self, base_url) -> None:
        self.base_url = base_url

    def set_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
