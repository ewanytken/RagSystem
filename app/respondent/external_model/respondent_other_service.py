import time
from openai import OpenAI
from app.logger import LoggerWrapper
from app.respondent.external_model.abstract_external_model import AbstractModelExternal
from app.utils import Utils

logger = LoggerWrapper()

class ExternalModel(AbstractModelExternal):

    def __init__(self) -> None:

        self.config = Utils.get_config_file()

        self.set_model_ticker(self.config["external_service"]["model"])
        self.set_base_url(self.config["external_service"]["url"])
        self.set_api_key(self.config["external_service"]["api_key"])

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        super().__init__()

    def generate(self, prompt, **kwargs) -> str:
        time.sleep(3)
        try:
            response = self.client.chat.completions.create(
                model=self.model_ticker,
                messages = prompt,
                max_tokens=1555
            )
            return response.choices[0].message.content

        except Exception as e:
            logger(f"Bad connection OtherService [[51]]: {e}")






