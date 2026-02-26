import time
from typing import Optional, Dict, Any
import requests
from app.logger import LoggerWrapper
from app.respondent.external_model.abstract_external_model import AbstractModelExternal
from app.respondent.abstract_respondent import Respondent

logger = LoggerWrapper()

class OllamaModel(AbstractModelExternal, Respondent):

    config: Optional[Dict[str, Any]] = None

    def __init__(self,
                 model_name: str = None,
                 url: str = "http://localhost:11434/api/generate",
                 json_payload: dict = None) -> None:

        super().__init__(model_name, url)
        self.json_payload = json_payload

    def set_payload(self, json_payload) -> None:
        self.json_payload = json_payload

    def set_config(self, config) -> None:
        self.config = config

    def generate(self, prompt: str, **kwargs) -> str:

        for attempt in range(3):
            try:
                response = requests.post(
                    url=self.base_url,
                    json={
                        "model": self.model_name,
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
                logger(f"Bad connection to Ollama server 50: {e}")
                time.sleep(wait_time)

