import logging
from pathlib import Path


class LoggerWrapper:

    def __init__(self, level=logging.INFO):
        self.logger = logging.getLogger(str(__package__))

        if not self.logger.handlers:
            self.logger.setLevel(level)
            self.level = level
            log_path = Path(__file__).parent.parent.parent / "logs"
            log_path.mkdir(exist_ok=True)
            log_file = log_path  / "app.log"
            handlerFile = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            handlerConsole = logging.StreamHandler()

            formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

            handlerFile.setFormatter(formatter)
            handlerConsole.setFormatter(formatter)

            self.logger.addHandler(handlerFile)
            self.logger.addHandler(handlerConsole)

            self.logger.propagate = False
        else:
            self.level = level

    def __call__(self, message):
        self.logger.log(self.level, message)
        if hasattr(self, 'handlerFile'):
            self.handlerFile.flush()