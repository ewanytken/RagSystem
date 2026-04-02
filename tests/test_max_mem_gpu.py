import unittest
from typing import Optional

import pynvml
import torch
from joblib.externals.cloudpickle import instance
from txtai import Embeddings

from app.logger import LoggerWrapper
from app.utils import Utils

logger = LoggerWrapper()

def get_gpu_id(size: int = 10_000_000_000) -> int | bool:
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


class Test(unittest.TestCase):

    def setUp(self):
        config = Utils.get_config_file("config_rus.yaml")
        model_ticker = config['embedding']['models']
        device = get_gpu_id(config['gpu']['memory_reserved']) if torch.cuda.is_available() else False
        self.embeddings = Embeddings({
            "path": model_ticker,
            "gpu": device,
            "content": True,
            "batch_size": config['embedding']['batch_size'],
        })

        logger(f"Loaded {model_ticker} Model on {device if device is not False else 'CPU'}")

    def test_document_processing(self):
        print(self.embeddings.info())
    if __name__ == '__main__':
        unittest.main()