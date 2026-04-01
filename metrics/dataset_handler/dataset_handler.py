import json
import os
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from app.logger import LoggerWrapper
from app.logger.logger_metrics import LoggerMetrics

logger_metrics = LoggerMetrics()
logger = LoggerWrapper()

class DatasetHandler:

    def __init__(self):
        self.config: Optional[Dict] = None
        self.data_frame: Optional[pd.DataFrame] = None

    def json_convert_to_csv(self) -> None:
        path = Path(__file__).parent.parent.parent / self.config['dataset']['path_dataset']
        logger_metrics(f"Path to datasets: {path}")

        self.data_frame  = pd.DataFrame(columns=['id', 'question', 'golden_answer', 'context'])
        for filename in os.listdir(path):
            if filename.endswith(".json") or filename.endswith(".jsonl"):
                file_path = path / filename
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        temp = pd.DataFrame(data)
                        self.data_frame = pd.concat([self.data_frame, temp], ignore_index=True)
                except FileNotFoundError as e:
                    logger(f"Dataset file not found: [[150]] by path {file_path}. Stack trace: {e}")
                except Exception as e:
                    logger(f"An error occurred: {e}")
                finally:
                    file.close()

        path_to_save = path / self.config['dataset']['file_save']
        self.data_frame.to_csv(path_to_save, encoding=self.config['dataset']['font_coding'], index=False)

    def set_config(self, config: Dict) -> None:
        self.config = config