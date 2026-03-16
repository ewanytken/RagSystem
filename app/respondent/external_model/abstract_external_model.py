from app.respondent.abstract_respondent import Respondent

class AbstractModelExternal(Respondent):

    def __init__(self) -> None:
        self.model_ticker = ""
        self.api_key = ""
        self.base_url = ""
        super().__init__()

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

