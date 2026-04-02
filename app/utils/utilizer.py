import json
from pathlib import Path
from typing import Any, Dict, Optional

import pynvml
import torch
import yaml

from app.logger import LoggerWrapper

logger = LoggerWrapper()

"""
get_config_file: load yaml
load_dictionary: load jsonl - ["entity": "", "transcript": ""] output: Dict[]
load_template: load prompts_templates/*_template
"""

class Utils:

    @staticmethod
    def get_config_file(config_path: str = "config_rus.yaml") -> Any:
        path = Path(__file__).parent.parent.parent / config_path
        try:
            with open(path, "r", encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError as e:
            logger(f"Config file not found: [[40]] by path {path}. Stack trace: {e}")
        except Exception as e:
            logger(f"An error occurred: {e}")
        finally:
            file.close()

    @staticmethod
    def load_label_description(dictionary_path: str = "label_description.jsonl") -> Dict[str, str]:

        abbreviations_dictionary = {}
        json_path = Path(__file__).parent.parent.parent/ "dictionary" / dictionary_path
        logger(f"Loading dictionary from {json_path}")

        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for item in data:
                    abbreviations_dictionary[item["entity"]] = item["description"]

        except FileNotFoundError as e:
            logger(f"Dictionary file not found: [[42]] by path {json_path}. Stack trace: {e}")
        except Exception as e:
            logger(f"An error occurred: {e}")
        finally:
            file.close()
        return abbreviations_dictionary

    @staticmethod
    def load_template(template_path: str) -> str:
        path = Path(__file__).parent.parent.parent / "prompts_templates" / template_path
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            logger(f"prompt_template File not Found (without format) [[41]] {path}.")
        except Exception as e:
            logger(f"An error occurred: {e}")
        finally:
            file.close()

    @staticmethod
    def get_gpu_id(size: int = 1_000_000_000) -> int | bool:
        logger("Memory distribution is working...")
        if torch.cuda.is_available():
            find_gpu_id: Optional[int] = None
            try:
                pynvml.nvmlInit()
                for gpu_id in range(torch.cuda.device_count()):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_not_allocated = torch.cuda.get_device_properties(
                        gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
                    vram_not_reserved = torch.cuda.get_device_properties(
                        gpu_id).total_memory - torch.cuda.memory.memory_reserved(gpu_id)
                    vram_not_placed = info.free
                    if (vram_not_allocated > size and vram_not_reserved > size
                            and vram_not_placed > size ):
                        find_gpu_id = gpu_id
                        break
            except Exception as e:
                logger(
                    f"Cannot search GPU or don't have video memory [[10]]. Current Device: {find_gpu_id}. Tracestack {e}")
            finally:
                pynvml.nvmlShutdown()

            logger(f"Current GPU: {find_gpu_id if find_gpu_id is not None else False}")
            return find_gpu_id if find_gpu_id is not None else False
        else:
            logger(f"Current device is CPU")
            return False