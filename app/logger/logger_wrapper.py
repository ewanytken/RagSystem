import logging

class LoggerWrapper:

    def __init__(self, level=logging.INFO):
        self.logger = logging.getLogger(str(__package__))

        if not self.logger.handlers:
            self.logger.setLevel(level)
            self.level = level

            handlerFile = logging.FileHandler(str(__package__), mode='w')
            handlerConsole = logging.StreamHandler()

            formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

            handlerFile.setFormatter(formatter)
            handlerConsole.setFormatter(formatter)

            self.logger.addHandler(handlerFile)
            self.logger.addHandler(handlerConsole)
        else:
            self.level = level

    def __call__(self, message):
        self.logger.log(self.level, message)