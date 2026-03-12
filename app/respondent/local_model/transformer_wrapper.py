from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig, AutoConfig
)
from app.utils import Utils
import math
from typing import Union, Any, Dict, Optional
import psutil
import pynvml
import torch
from huggingface_hub import snapshot_download
from torch import nn
from app.logger import LoggerWrapper
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from app.respondent.abstract_respondent import Respondent

logger = LoggerWrapper()

class TransformerWrapper(Respondent):

    def __init__(self, model_name: str = None, use_cpu_only: bool = False, quantize: bool = False, **kwargs) -> None:

        self.config = Utils.get_config_file()
        self.model_name: Optional[str] = model_name
        self.use_cpu_only: Optional[bool] = use_cpu_only
        self.device: Optional[str] = None

        if self.model_name is None and self.config is not None:
            self.model_name = self.config.get('llm_local', {}).get('model')
            if self.model_name is None:
                raise ValueError("No model name provided and no default model found in config [[130]]")

        quantization_config = None
        if quantize and not self.use_cpu_only:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.config.get('quantization', {}).get('load_in_8bit', True),
                llm_int8_threshold=self.config.get('quantization', {}).get('llm_int8_threshold', 6.0),
                llm_int8_has_fp16_weight=self.config.get('quantization', {}).get('llm_int8_has_fp16_weight', False),
            )

        model_kwargs = {
            "dtype": torch.float16 if self.use_cpu_only else torch.float16,
            **kwargs
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        logger(f"Model ticket: {self.model_name}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=self.config.get('tokenizer_local', {}).get('use_fast', True),
            padding_side=self.config.get('tokenizer_local', {}).get('padding_side', "left"),  # Left padding for causal LM
            truncation_side=self.config.get('tokenizer_local', {}).get('truncation_side', "left")
        )

        self.stream: bool = False

        self.set_gpu_distribution()
        self._setup_special_tokens()
        self.default_gen_params = self._get_default_generation_params()

        super().__init__()

    def _setup_special_tokens(self) -> None:
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '</s>'})

        # Resize model embeddings if tokens were added
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger(f"Resized model embeddings to {len(self.tokenizer)}")

    def _get_default_generation_params(self) -> Dict[str, Any]:
        defaults = {
            "max_new_tokens": self.config.get('llm_local', {}).get('max_new_tokens', 2000),
            "temperature": self.config.get('llm_local', {}).get('temperature', 0.7),
            "top_p": self.config.get('llm_local', {}).get('top_p', 0.9),
            "top_k": self.config.get('llm_local', {}).get('top_k', 50),
            "do_sample": self.config.get('llm_local', {}).get('do_sample', True),
            "num_return_sequences": self.config.get('llm_local', {}).get('num_return_sequences', 1),
            "repetition_penalty": self.config.get('llm_local', {}).get('repetition_penalty', 1.1),
            "no_repeat_ngram_size": self.config.get('llm_local', {}).get('no_repeat_ngram_size', 3),
            # "early_stopping": self.config.get('llm_local', {}).get('early_stopping', True),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        return defaults

    def generate(self,
                 prompt: str,
                 **kwargs) -> Union[str, Any]:

        gen_params = {**self.default_gen_params, **kwargs}
        logger(f"{self.model_name} is generating response on device: {self.device} \n"
               f"Max model context: {self.get_max_context_length(self.model_name)} \n"
               f"Input prompt length: {len(prompt)} \n")

        template = {"role": "user", "content": prompt}
        formatted_chat = self.tokenizer.apply_chat_template(
            [template],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            formatted_chat,
            return_tensors="pt",
            add_special_tokens=self.config.get('tokenizer_local', {}).get('add_special_tokens', True),
            padding=self.config.get('tokenizer_local', {}).get('padding', True),
            truncation=self.config.get('tokenizer_local', {}).get('truncation', True),
            max_length=self.config.get('tokenizer_local', {}).get('max_length', 2000),
        )

        # if len(inputs['input_ids'].size(1)) > self.get_max_context_length(self.model_name):
        #     inputs['input_ids'][1] = inputs['input_ids'][1][:self.get_max_context_length(self.model_name)]
        #     logger(f"Inputs token for generation truncated to {len(inputs['input_ids'])}")
        decoded_text: Optional[str] = "!!!MODEL DON'T GENERATE RESPONSE!!!"
        try:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                if self.stream:
                    return self._stream_generate(inputs, gen_params)
                else:
                    outputs = self.model.generate(
                        **inputs,
                        **gen_params
                    )

            generated_tokens = outputs[0][inputs['input_ids'].size(1):]

            decoded_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()

        except Exception as e:
            logger(f"Error occur while generate response [[131]] {e}")
        return decoded_text

    def _stream_generate(self, inputs: Dict[str, torch.Tensor], gen_params: Dict[str, Any]):
        """
        Stream generation token by token.

        Args:
            inputs: Tokenized inputs
            gen_params: Generation parameters
        """
        from transformers import TextStreamer

        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        self.model.generate(
            **inputs,
            streamer=streamer,
            **gen_params
        )

    def generate_batch(self,
                       prompts: list[str],
                       batch_size: int = 4,
                       **kwargs) -> list[str]:
        """
        Generate text for multiple prompts efficiently.

        Args:
            prompts: List of input prompts
            batch_size: Batch size for processing
            **kwargs: Generation parameters

        Returns:
            List of generated texts
        """
        results = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = []

            # Prepare batch templates
            formatted_chats = []
            for prompt in batch_prompts:
                template = {"role": "user", "content": prompt}
                formatted = self.tokenizer.apply_chat_template(
                    [template],
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted_chats.append(formatted)

            # Tokenize batch with padding
            inputs = self.tokenizer(
                formatted_chats,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Generate for batch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **{**self.default_gen_params, **kwargs}
                )

            # Decode each output
            for j, output in enumerate(outputs):
                generated = output[inputs['input_ids'][j].size(0):]
                decoded = self.tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                batch_results.append(decoded.strip())

            results.extend(batch_results)

        return results

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "model_type": self.model.config.model_type,
            "vocab_size": len(self.tokenizer),
            "max_length": self.model.config.max_position_embeddings if hasattr(self.model.config,
                                                                               'max_position_embeddings') else None,
            "quantized": hasattr(self.model, 'is_quantized'),
        }

    def set_model_name(self, model_name: str) -> None:
        if model_name != self.model_name:
            self.model_name = model_name
            self.__init__(model_name=model_name)

    def clear_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger("Cleared CUDA cache")

    def set_stream(self, stream: bool) -> None:
        self.stream = stream


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_cache()
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_max_context_length(self, model_name):

        try:
            config = AutoConfig.from_pretrained(model_name)

            if hasattr(config, "max_position_embeddings"):
                return config.max_position_embeddings
            elif hasattr(config, "model_max_length"):
                return config.model_max_length
            else:
                return 0

        except Exception as e:
            return f"Error found max context length of model by ticket [[132]]: {e}"

    def set_gpu_distribution(self) -> None:
        logger("Memory's distribution is working...")

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
                logger(f"Cannot search GPU or don't have video memory 10. Current Device [[133]]: {self.device}. Tracestack {e}")
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
            weights_location = snapshot_download(repo_id = str(self.model.name_or_path),
                                                 allow_patterns=["*.bin", "*.safetensors", "*.json"],
                                                 ignore_patterns=["optimizer.pt", "scheduler.pt"],
                                                 )
            logger(f"Downloaded model name: {self.model.name_or_path}")

            with init_empty_weights():
                model_test = self.model
            if hasattr(model_test, 'tie_weights'):
                model_test.tie_weights()

            device_map = infer_auto_device_map(
                model_test,
                max_memory=self.memory_calculation(),
                dtype=torch.float16
            )

            logger(f"Memory distribution: {self.memory_calculation()}")

            model_accelerate = load_checkpoint_and_dispatch(
                model_test, checkpoint=weights_location, device_map=device_map)

            logger(f"Device mapping to memory: {model_accelerate.hf_device_map}")

        except Exception as err:
            logger(f"Multi GPU distribution error. Accelerator didn't work correctly [[134]] {err}")

        if model_accelerate is not None:
            self.device = model_accelerate.device
            logger(f"Final device mapping {self.device}")

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
                    memory_distribution.update({gpu_id: "{memory}GiB".format(memory=math.floor(memory_available))})

            except Exception as err:
                logger(f"Memory calculation Error [[135]]: {err}")
            finally:
                pynvml.nvmlShutdown()

        if memory_allocated_overall < model_size:
            memory_distribution.update({"cpu": "{memory}GiB".format(memory=math.floor(self.get_free_ram()))})
        logger(f"Final memory distribution: {memory_distribution}")
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