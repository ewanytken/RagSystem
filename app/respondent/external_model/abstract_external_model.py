from abc import ABC, abstractmethod
from typing import Union, Dict, List


class AbstractModelExternal(ABC):

    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

        self.system_tag = ['your_role', 'instruction', 'constraint', 'clue', 'context']


    def set_system_tag(self, tag: list) -> None:
        self.system_tag = tag

    def set_base_url(self, base_url) -> None:
        self.base_url = base_url

    def set_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key

    @abstractmethod
    def generate(self, json_payload: dict) -> str:
        pass

    def template(self, inst_prompt: dict) -> Union[Dict[str, str], List[Dict[str, str]]]:

        list_of_dict = []
        list_of_system = self.system_tag

        for k, v in inst_prompt.items():
            chat = {}
            if k in list_of_system:
                chat.update({"role": "system"})
                chat.update({"content": f"<{k}>{v}</{k}>"})
            else:
                chat.update({"role": "user"})
                chat.update({"content": f"<{k}>{v}</{k}>"})
            list_of_dict.append(chat)

        return list_of_dict