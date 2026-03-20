import time
from typing import Optional

from openai import OpenAI

from app.logger import LoggerWrapper
from app.respondent.abstract_respondent import Respondent
from app.utils import Utils

logger = LoggerWrapper()

class ExternalModel(Respondent):

    def __init__(self, model_ticker=None, base_url=None, api_key=None) -> None:

        self.model_ticker: Optional[str] = "Model don't install"
        self.api_key: Optional[str] = None
        self.base_url: Optional[str] = "URL don't install"
        self.config = Utils.get_config_file()

        if model_ticker and base_url and api_key:
            self.set_model_ticker(model_ticker)
            self.set_base_url(base_url)
            self.set_api_key(api_key)

        elif model_ticker: # condition for remote service
            self.set_model_ticker(model_ticker)
            self.set_base_url(self.config["external_service"]["url"])
            self.set_api_key(self.config["external_service"]["api_key"])

        else:
            self.set_model_ticker(self.config["external_service"]["model"])
            self.set_base_url(self.config["external_service"]["url"])
            self.set_api_key(self.config["external_service"]["api_key"])

        super().__init__()

        logger(f"Model: {self.get_model_ticker()}, Url: {self.get_base_url()}, APIKey: {True if self.get_api_key() else False}")
        self.client = OpenAI(base_url=self.get_base_url(), api_key=self.get_api_key())

    def generate(self, prompt, **kwargs) -> str:
        time.sleep(3)
        try:
            response = self.client.chat.completions.create(
                model=self.get_model_ticker() if self.get_model_ticker() is not "*" else "", # TODO remove for domestic test
                messages = [{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content.strip()

            return answer

        except Exception as e:
            logger(f"Bad connection to Model Service [[51]]: {e}")

    def set_base_url(self, base_url: str) -> None:
        self.base_url = base_url

    def set_model_ticker(self, model_ticker: str) -> None:
        self.model_ticker = model_ticker

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key

    def get_base_url(self) -> str:
        return self.base_url

    def get_model_ticker(self) -> str:
        return self.model_ticker

    def get_api_key(self) -> str:
        return self.api_key

    def __repr__(self):
        return f"Load Remote model with ticket: {self.get_model_ticker()}"




