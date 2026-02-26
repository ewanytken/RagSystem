from abc import abstractmethod

class Respondent:

    @abstractmethod
    def generate(self, prompt, **kwargs):
        pass