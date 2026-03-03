from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from app.respondent.abstract_respondent import Respondent

class AbstractModelExternal(Respondent):

    def __init__(self) -> None:
        super().__init__()
        self.model_ticker = ""
        self.api_key = ""
        self.base_url = ""

    def set_base_url(self, base_url) -> None:
        self.base_url = base_url

    def set_model_ticker(self, model_ticker: str) -> None:
        self.model_ticker = model_ticker

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key

