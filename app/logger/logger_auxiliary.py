import logging
from pathlib import Path

"""
Auxiliary logger without Console output. Save to logs/auxiliary_log.log
"""

class AuxiliaryLogger:

    def __init__(self, level=logging.INFO):
        self.logger = logging.getLogger(str(__package__))

        if not self.logger.handlers:
            self.logger.setLevel(level)
            self.level = level
            log_path = Path(__file__).parent.parent.parent / "logs"
            log_path.mkdir(exist_ok=True)
            log_file = log_path  / "auxiliary_log.log"

            if log_file.exists():
                log_file.unlink()

            self.handlerFile = logging.FileHandler(log_file, mode='a', encoding='utf-8')

            formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

            self.handlerFile.setFormatter(formatter)

            self.logger.addHandler(self.handlerFile)

        else:
            self.level = level

    def __call__(self, message):
        self.logger.log(self.level, message)
        if hasattr(self, 'handlerFile'):
            self.handlerFile.flush()
            self.handlerFile.close()
