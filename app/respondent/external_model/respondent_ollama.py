import time

import requests

from app.logger import LoggerWrapper
from app.respondent.abstract_respondent import Respondent
from app.respondent.external_model.abstract_external_model import AbstractModelExternal
from app.utils import Utils

logger = LoggerWrapper()

class OllamaModel(AbstractModelExternal, Respondent):

    def __init__(self) -> None:

        self.config = Utils.get_config_file()

        self.set_model_ticker(self.config["ollama"]["model"])
        self.set_base_url(self.config["ollama"]["url"])

        super().__init__()

    def __repr__(self):
        return f"Load model from Ollama service with ticket: {self.get_model_ticker()}"

    def generate(self, prompt: str, **kwargs) -> str:
        for attempt in range(3):
            try:
                response = requests.post(
                    url=self.base_url,
                    json={
                        "model": self.model_ticker,
                        "prompt": prompt,
                        "options": {
                            "num_predict": kwargs.get("max_length", self.config['ollama']['max_length']),
                            "temperature": kwargs.get("temperature", self.config['ollama']['temperature']),
                            "top_p": kwargs.get("top_p", self.config['ollama']['top_p'])
                        },
                        "stream": False
                    },
                    timeout=self.config['ollama']['timeout']
                )

                result = response.json()
                return result.get("response", "NO RESPONSE FROM OLLAMA API")

            except requests.exceptions.RequestException as e:
                wait_time = (attempt + 1) * 5
                logger(f"Attempt {attempt + 1} don't : {str(e)}. Retry from {wait_time} sec...")
                logger(f"Bad connection to Ollama server [[50]]: {e}")
                time.sleep(wait_time)

