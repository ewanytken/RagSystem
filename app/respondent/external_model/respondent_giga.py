import json
import time
import requests
from app.respondent.external_model.abstract_external_model import AbstractModelExternal
from app.utils import Utils


class TargetGiga(AbstractModelExternal):

    def __init__(self, authorization: str = None, uuid: str = None):
        super().__init__()

        self.config = Utils.get_config_file()
        if authorization is None or uuid is None:
            authorization = self.config['authorization']
            uuid = self.config['uuid']

        self.giga_rest = GigaRest(authorization, uuid)

    def generate(self, prompt: str, **kwargs) -> str:
        time.sleep(3)
        return self.giga_rest.get_message(prompt)

class GigaRest:

    def __init__(self, authorization: str, uuid: str):
        self.token = self._token_obtain(authorization, uuid)

    def _token_obtain(self, authorization: str, uuid: str) -> str:
        token = None
        try:
            uri = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": uuid,
                "Authorization": "Basic {}".format(authorization)
            }

            token = requests.post(uri,
                                  verify=False,
                                  data="scope=GIGACHAT_API_PERS",
                                  headers=headers)
        except Exception as e:
            print(e)

        return token.json()["access_token"]

    def get_message(self, message: str, system_message: str = None) -> str:
        response = None

        try:
            uri = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer {}".format(self.token)
            }
            messages = []

            if system_message is not None:
                system_role = {
                    "role": "system",
                    "content": system_message
                }
                messages.append(system_role)

            user_role = {
                "role": "system",
                "content": message
            }

            messages.append(user_role)

            json_payload = {
                "model": "GigaChat",
                "messages": messages,
                "stream": False,
                "update_interval": 0
            }

            response = requests.post(uri,
                                     verify=False,
                                     data=json.dumps(json_payload),
                                     headers=headers)

        except Exception as e:
            print(e)

        assert response is not None, "Empty response from GigaChat"

        return response.json()["choices"][0]["message"]["content"]

