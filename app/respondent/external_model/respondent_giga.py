import time
import json

import requests



from app.core.target_external.abstract_model_ext import AbstractModelExternal

from app.respondent.interface_respondent import Respondent


class TargetGiga(AbstractModelExternal, Respondent):

    def __init__(self, authorization: str, uuid: str):
        super().__init__()
        self.giga_rest = GigaRest(authorization, uuid)


    def generate(self, json_payload: dict) -> str:
        time.sleep(3)
        return self.giga_rest.get_message(json_payload["query"])

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

class ResponseFromService:

    @staticmethod
    def send_response(response: str, uri: str = "http://localhost:5005/giga-answer") -> str:
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            json_payload = {
                "query": response
            }

            result = requests.post(uri,
                          data=json_payload,
                          headers=headers)
        except:
            raise RuntimeError

        return result.text