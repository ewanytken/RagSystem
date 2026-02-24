import json
import os
from pathlib import Path
from typing import List, Any, Dict
import yaml
from app.logger import LoggerWrapper

logger = LoggerWrapper()

"""
get_config_file: load yaml
load_dictionary: load jsonl - ["entity": "", "transcript": ""] output: Dict[]
"""

class Utils:

    @staticmethod
    def get_config_file(config_path: str = "config.yaml") -> Any:
        path = Path(__file__).parent.parent / config_path
        try:
            return yaml.safe_load(open(path, 'r', encoding='utf-8'))
        except FileNotFoundError as e:
            logger(f"Config file not found: 40 by path {path}. Stack trace: {e}")
        except Exception as e:
            logger(f"An error occurred: {e}")

    @staticmethod
    def load_dictionary(dictionary_path: str = "dictionary.jsonl") -> Dict[str, str]:

        abbreviations_dictionary = {}
        json_path = Path(__file__).parent.parent / "dictionary" / dictionary_path
        logger(f"Loading dictionary from {json_path}")

        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for item in data:
                    abbreviations_dictionary[item["transcript"]] = item["entity"]

        except FileNotFoundError as e:
            logger(f"Dictionary file not found: 42 by path {json_path}. Stack trace: {e}")
        except Exception as e:
            logger(f"An error occurred: {e}")

        return abbreviations_dictionary


    @staticmethod
    def get_docx_files(directory: str) -> List[str]:
        docx_files = []
        for filename in os.listdir(directory):
            if filename.lower().endswith('.docx'):
                docx_files.append(os.path.join(directory, filename))
        return docx_files


