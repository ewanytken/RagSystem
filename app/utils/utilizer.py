
import os
from pathlib import Path
from typing import List, Any

import yaml

from app.logger import LoggerWrapper

logger = LoggerWrapper()

class Utils:

    @staticmethod
    def get_config_file(config_path: str = "config.yaml") -> Any:
        path = Path(__file__).parent / config_path
        try:
            return yaml.safe_load(open(path, 'r', encoding='utf-8'))
        except FileNotFoundError as e:
            logger(f"Config file not found: 40 by path {path}")
        except Exception as e:
            logger(f"An error occurred: {e}")

    @staticmethod
    def get_docx_files(directory: str) -> List[str]:
        docx_files = []
        for filename in os.listdir(directory):
            if filename.lower().endswith('.docx'):
                docx_files.append(os.path.join(directory, filename))
        return docx_files


