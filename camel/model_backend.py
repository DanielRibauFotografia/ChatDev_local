# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from abc import ABC, abstractmethod
import typing

import openai
import tiktoken

from camel.typing import ModelType
from chatdev.statistics import prompt_cost
from chatdev.utils import log_visualize

try:
    from openai.types.chat import ChatCompletion
    openai_new_api = True  # new openai api version
except ImportError:
    openai_new_api = False  # old openai api version

try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

import os

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
if 'BASE_URL' in os.environ:
    BASE_URL = os.environ['BASE_URL']
else:
    BASE_URL = None

# Global model configuration for local models
_LOCAL_MODEL_CONFIG = {}


def set_local_model_config(config: typing.Dict):
    """Set global configuration for local models."""
    global _LOCAL_MODEL_CONFIG
    _LOCAL_MODEL_CONFIG = config


def get_local_model_config() -> typing.Dict:
    """Get global configuration for local models."""
    return _LOCAL_MODEL_CONFIG.copy()


class ModelBackend(ABC):
    r"""Base class for different model backends.
    May be OpenAI API, a local LLM, a stub for unit tests, etc."""

    @abstractmethod
    def run(self, *args, **kwargs):
        r"""Runs the query to the backend model.

        Raises:
            RuntimeError: if the return value from OpenAI API
            is not a dict that is expected.

        Returns:
            typing.Dict[str, typing.typing.Any]: All backends must return a dict in OpenAI format.
        """
        pass


class OpenAIModel(ModelBackend):
    r"""OpenAI API in a unified ModelBackend interface."""

    def __init__(self, model_type: ModelType, model_config_dict: typing.Dict) -> None:
        super().__init__()
        self.model_type = model_type
        self.model_config_dict = model_config_dict

    def run(self, *args, **kwargs):
        string = "\n".join([message["content"] for message in kwargs["messages"]])
        encoding = tiktoken.encoding_for_model(self.model_type.value)
        num_prompt_tokens = len(encoding.encode(string))
        gap_between_send_receive = 15 * len(kwargs["messages"])
        num_prompt_tokens += gap_between_send_receive

        if openai_new_api:
            # Experimental, add base_url
            if BASE_URL:
                client = openai.OpenAI(
                    api_key=OPENAI_API_KEY,
                    base_url=BASE_URL,
                )
            else:
                client = openai.OpenAI(
                    api_key=OPENAI_API_KEY
                )

            num_max_token_map = {
                "gpt-3.5-turbo": 4096,
                "gpt-3.5-turbo-16k": 16384,
                "gpt-3.5-turbo-0613": 4096,
                "gpt-3.5-turbo-16k-0613": 16384,
                "gpt-4": 8192,
                "gpt-4-0613": 8192,
                "gpt-4-32k": 32768,
                "gpt-4-turbo": 100000,
                "gpt-4o": 4096, #100000
                "gpt-4o-mini": 16384, #100000
            }
            num_max_token = num_max_token_map[self.model_type.value]
            num_max_completion_tokens = num_max_token - num_prompt_tokens
            self.model_config_dict['max_tokens'] = num_max_completion_tokens

            response = client.chat.completions.create(*args, **kwargs, model=self.model_type.value,
                                                      **self.model_config_dict)

            cost = prompt_cost(
                self.model_type.value,
                num_prompt_tokens=response.usage.prompt_tokens,
                num_completion_tokens=response.usage.completion_tokens
            )

            log_visualize(
                "**[OpenAI_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\ncost: ${:.6f}\n".format(
                    response.usage.prompt_tokens, response.usage.completion_tokens,
                    response.usage.total_tokens, cost))
            if not isinstance(response, ChatCompletion):
                raise RuntimeError("Unexpected return from OpenAI API")
            return response
        else:
            num_max_token_map = {
                "gpt-3.5-turbo": 4096,
                "gpt-3.5-turbo-16k": 16384,
                "gpt-3.5-turbo-0613": 4096,
                "gpt-3.5-turbo-16k-0613": 16384,
                "gpt-4": 8192,
                "gpt-4-0613": 8192,
                "gpt-4-32k": 32768,
                "gpt-4-turbo": 100000,
                "gpt-4o": 4096, #100000
                "gpt-4o-mini": 16384, #100000
            }
            num_max_token = num_max_token_map[self.model_type.value]
            num_max_completion_tokens = num_max_token - num_prompt_tokens
            self.model_config_dict['max_tokens'] = num_max_completion_tokens

            response = openai.ChatCompletion.create(*args, **kwargs, model=self.model_type.value,
                                                    **self.model_config_dict)

            cost = prompt_cost(
                self.model_type.value,
                num_prompt_tokens=response["usage"]["prompt_tokens"],
                num_completion_tokens=response["usage"]["completion_tokens"]
            )

            log_visualize(
                "**[OpenAI_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\ncost: ${:.6f}\n".format(
                    response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"],
                    response["usage"]["total_tokens"], cost))
            if not isinstance(response, Dict):
                raise RuntimeError("Unexpected return from OpenAI API")
            return response


class StubModel(ModelBackend):
    r"""A dummy model used for unit tests."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def run(self, *args, **kwargs) -> typing.Dict[str, typing.Any]:
        ARBITRARY_STRING = "Lorem Ipsum"

        return dict(
            id="stub_model_id",
            usage=dict(),
            choices=[
                dict(finish_reason="stop",
                     message=dict(content=ARBITRARY_STRING, role="assistant"))
            ],
        )


class HuggingFaceModel(ModelBackend):
    r"""HuggingFace local model backend."""

    def __init__(self, model_type: ModelType, model_config_dict: typing.Dict) -> None:
        super().__init__()
        self.model_type = model_type
        
        # Merge global config with instance config
        global_config = get_local_model_config()
        merged_config = {**global_config, **model_config_dict}
        self.model_config_dict = merged_config
        
        self.model_name = merged_config.get('model_name', 'microsoft/DialoGPT-medium')
        self.device = merged_config.get('device', 'auto')
        self.max_length = merged_config.get('max_length', 1024)
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device if self.device != 'auto' else None
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except ImportError:
            raise ImportError("Please install transformers and torch: pip install transformers torch")

    def run(self, *args, **kwargs) -> typing.Dict[str, typing.Any]:
        messages = kwargs.get("messages", [])
        
        # Convert messages to prompt format
        prompt = self._messages_to_prompt(messages)
        
        # Tokenize and generate
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        if hasattr(self.model, 'device'):
            inputs = inputs.to(self.model.device)
            
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=min(len(inputs[0]) + self.max_length, 2048),
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=self.model_config_dict.get('temperature', 0.7),
                top_p=self.model_config_dict.get('top_p', 0.9),
            )
        
        # Decode response
        response_text = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        # Return in OpenAI format
        return {
            "id": f"huggingface-{self.model_name}",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_text.strip()
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(inputs[0]),
                "completion_tokens": len(outputs[0]) - len(inputs[0]),
                "total_tokens": len(outputs[0])
            }
        }
    
    def _messages_to_prompt(self, messages):
        """Convert OpenAI message format to prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)


class LlamaCppModel(ModelBackend):
    r"""llama.cpp local model backend."""

    def __init__(self, model_type: ModelType, model_config_dict: typing.Dict) -> None:
        super().__init__()
        self.model_type = model_type
        
        # Merge global config with instance config
        global_config = get_local_model_config()
        merged_config = {**global_config, **model_config_dict}
        self.model_config_dict = merged_config
        
        self.model_path = merged_config.get('model_path', '')
        self.n_ctx = merged_config.get('n_ctx', 2048)
        self.n_gpu_layers = merged_config.get('n_gpu_layers', 0)
        
        if not self.model_path:
            raise ValueError("model_path must be specified for llama.cpp models")
            
        try:
            from llama_cpp import Llama
            
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
        except ImportError:
            raise ImportError("Please install llama-cpp-python: pip install llama-cpp-python")

    def run(self, *args, **kwargs) -> typing.Dict[str, typing.Any]:
        messages = kwargs.get("messages", [])
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Generate response
        response = self.llm(
            prompt,
            max_tokens=self.model_config_dict.get('max_tokens', 512),
            temperature=self.model_config_dict.get('temperature', 0.7),
            top_p=self.model_config_dict.get('top_p', 0.9),
            stop=["User:", "System:"],
            echo=False
        )
        
        # Return in OpenAI format
        return {
            "id": f"llama-cpp-{os.path.basename(self.model_path)}",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response['choices'][0]['text'].strip()
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": response['usage']['prompt_tokens'],
                "completion_tokens": response['usage']['completion_tokens'],
                "total_tokens": response['usage']['total_tokens']
            }
        }
    
    def _messages_to_prompt(self, messages):
        """Convert OpenAI message format to prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)


class OllamaModel(ModelBackend):
    r"""Ollama local model backend."""

    def __init__(self, model_type: ModelType, model_config_dict: typing.Dict) -> None:
        super().__init__()
        self.model_type = model_type
        
        # Merge global config with instance config
        global_config = get_local_model_config()
        merged_config = {**global_config, **model_config_dict}
        self.model_config_dict = merged_config
        
        self.model_name = merged_config.get('model_name', 'llama2')
        self.base_url = merged_config.get('base_url', 'http://localhost:11434')
        
        try:
            import requests
            # Test connection to Ollama
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")
        except ImportError:
            raise ImportError("Please install requests: pip install requests")
        except Exception as e:
            raise ConnectionError(f"Ollama not available: {e}")

    def run(self, *args, **kwargs) -> typing.Dict[str, typing.Any]:
        import requests
        import json
        
        messages = kwargs.get("messages", [])
        
        # Convert to Ollama format
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.get('role'),
                "content": msg.get('content')
            })
        
        payload = {
            "model": self.model_name,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": self.model_config_dict.get('temperature', 0.7),
                "top_p": self.model_config_dict.get('top_p', 0.9),
                "num_predict": self.model_config_dict.get('max_tokens', 512)
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.text}")
        
        result = response.json()
        
        # Return in OpenAI format
        return {
            "id": f"ollama-{self.model_name}",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": result['message']['content']
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": result.get('prompt_eval_count', 0),
                "completion_tokens": result.get('eval_count', 0),
                "total_tokens": result.get('prompt_eval_count', 0) + result.get('eval_count', 0)
            }
        }


class LocalAIModel(ModelBackend):
    r"""LocalAI model backend."""

    def __init__(self, model_type: ModelType, model_config_dict: typing.Dict) -> None:
        super().__init__()
        self.model_type = model_type
        
        # Merge global config with instance config
        global_config = get_local_model_config()
        merged_config = {**global_config, **model_config_dict}
        self.model_config_dict = merged_config
        
        self.model_name = merged_config.get('model_name', 'gpt-3.5-turbo')
        self.base_url = merged_config.get('base_url', 'http://localhost:8080')
        
        try:
            import requests
            # Test connection to LocalAI
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Cannot connect to LocalAI at {self.base_url}")
        except ImportError:
            raise ImportError("Please install requests: pip install requests")
        except Exception as e:
            raise ConnectionError(f"LocalAI not available: {e}")

    def run(self, *args, **kwargs) -> typing.Dict[str, typing.Any]:
        import requests
        
        # LocalAI uses OpenAI-compatible API
        payload = {
            "model": self.model_name,
            "messages": kwargs.get("messages", []),
            "temperature": self.model_config_dict.get('temperature', 0.7),
            "max_tokens": self.model_config_dict.get('max_tokens', 512),
            "top_p": self.model_config_dict.get('top_p', 0.9)
        }
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"LocalAI API error: {response.text}")
        
        return response.json()


class ModelFactory:
    r"""Factory of backend models.

    Raises:
        ValueError: in case the provided model type is unknown.
    """

    @staticmethod
    def create(model_type: ModelType, model_config_dict: typing.Dict) -> ModelBackend:
        default_model_type = ModelType.GPT_3_5_TURBO

        if model_type in {
            ModelType.GPT_3_5_TURBO,
            ModelType.GPT_3_5_TURBO_NEW,
            ModelType.GPT_4,
            ModelType.GPT_4_32k,
            ModelType.GPT_4_TURBO,
            ModelType.GPT_4_TURBO_V,
            ModelType.GPT_4O,
            ModelType.GPT_4O_MINI,
            None
        }:
            model_class = OpenAIModel
        elif model_type == ModelType.STUB:
            model_class = StubModel
        elif model_type == ModelType.HUGGINGFACE:
            model_class = HuggingFaceModel
        elif model_type == ModelType.LLAMA_CPP:
            model_class = LlamaCppModel
        elif model_type == ModelType.OLLAMA:
            model_class = OllamaModel
        elif model_type == ModelType.LOCALAI:
            model_class = LocalAIModel
        else:
            raise ValueError("Unknown model")

        if model_type is None:
            model_type = default_model_type

        # log_visualize("Model Type: {}".format(model_type))
        inst = model_class(model_type, model_config_dict)
        return inst
