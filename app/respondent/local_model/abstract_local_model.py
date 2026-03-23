import math
from typing import Dict, Optional

import psutil
import pynvml
import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from torch import nn

from app.logger import LoggerWrapper
from app.respondent.abstract_respondent import Respondent

logger = LoggerWrapper()

class AbstractLocalRespondent(Respondent):

    def __init__(self, model, tokenizer, use_cpu_only: bool) -> None:

        self.model = model
        self.tokenizer = tokenizer
        self.use_cpu_only: Optional[bool] = use_cpu_only
        self.device: Optional[str] = None

        self.set_gpu_distribution()

        super().__init__()

        assert model is not None, "Exception: NO MODEL"
        assert tokenizer is not None, "Exception: NO TOKENIZER"

    def set_gpu_distribution(self) -> None:
        logger("Memory distribution is working...")
        if torch.cuda.is_available() and self.use_cpu_only is False:
            try:
                pynvml.nvmlInit()
                for gpu_id in range(torch.cuda.device_count()):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_not_allocated = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
                    vram_not_reserved = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory.memory_reserved(gpu_id)
                    vram_not_placed = info.free # OVERALL FREE MEMORY INCLUDE OTHER APP
                    if (vram_not_allocated > self.get_model_size() * 1.07 and vram_not_reserved > self.get_model_size() * 1.07
                            and vram_not_placed > self.get_model_size() * 1.07): # +7% size, 3% sometimes not enough 
                        logger(f"Current CUDA: {gpu_id}")
                        logger(f"Model Size: {self.get_model_size()}")
                        logger(f"Free VRAM: {vram_not_placed}")
                        self.device = "cuda:{}".format(gpu_id)
                        self.model.to(self.device)
                        break

                if self.device is None:
                    self.model = self.multi_gpu()
            except Exception as e:
                logger(f"Cannot search GPU or don't have video memory 10. Current Device: {self.device}. Tracestack {e}")
            finally:
                pynvml.nvmlShutdown()

        else:
            self.device = "cpu"
            self.model.to(self.device)
            logger(f"Current device is CPU")

    def multi_gpu(self) -> nn.Module:
        logger("Multi-GPU mode STARTED")
        model_accelerate: Optional[nn.Module] = None

        try:
            weights_location = snapshot_download(repo_id=self.model.name_or_path)
            logger(f"Downloaded model name: {self.model.name_or_path}")

            with init_empty_weights():
                model_test = self.model
            model_test.tie_weights()

            device_map = infer_auto_device_map(
                model_test,
                max_memory=self.memory_calculation()
            )

            logger(f"Memory distribution: {self.memory_calculation()}")

            model_accelerate = load_checkpoint_and_dispatch(
                model_test, checkpoint=weights_location, device_map=device_map)

            logger(f"Device mapping to memory: {model_accelerate.hf_device_map}")

        except Exception as err:
            logger(err)

        self.device = model_accelerate.device
        return model_accelerate

    # Example {0: "13GiB", "cpu": "60GiB"}
    def memory_calculation(self) -> Dict:

        memory_distribution: Optional[Dict] = {}
        model_size = self.get_model_size()
        memory_allocated_overall: Optional[float] = 0.0

        if torch.cuda.is_available():
            try:
                pynvml.nvmlInit()

                for gpu_id in range(torch.cuda.device_count()):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    memory_available = info.free / (1024 ** 2)

                    memory_allocated_overall += memory_available
                    memory_distribution.update({gpu_id: "{}GiB".format(math.floor(memory_available))})

            except Exception as err:
                logger(err)
            finally:
                pynvml.nvmlShutdown()

        if memory_allocated_overall < model_size:
            memory_distribution.update({"cpu": "{}GiB".format(math.floor(self.get_free_ram()))})
        logger(memory_distribution)
        return memory_distribution

    def get_model_size(self) -> int:
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return param_size + buffer_size

    def get_free_ram(self) -> float:
        return psutil.virtual_memory().available / 1024 ** 3

    def cleanup_memory(self):
        del self.model, self.tokenizer
