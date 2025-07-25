# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========

from typing import Dict, List, Any, Optional
import time
import uuid
import platform
from dataclasses import dataclass

@dataclass
class LocalLLMConfig:
    """Configuration for local LLM inference."""
    backend: str = "huggingface"
    model_name: str = "codellama/CodeLlama-7b-Python-hf"
    device: str = "auto"
    quantization: Optional[str] = "4bit"
    max_memory_gb: int = 16
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    simulation_mode: bool = False  # For testing without loading actual models


class LocalLLMInterface:
    """
    Local LLM Interface that provides OpenAI-compatible API responses.
    Supports multiple backends with ARM64 optimization for Apple Silicon.
    """
    
    def __init__(self, config: LocalLLMConfig = None):
        self.config = config or LocalLLMConfig()
        self.model = None
        self.tokenizer = None
        self.device = None
        self._setup_device()
        self._load_model()
    
    def _setup_device(self):
        """Setup optimal device for inference."""
        try:
            import torch
            
            if self.config.device == "auto":
                # Auto-detect optimal device
                if torch.backends.mps.is_available() and platform.processor() == 'arm':
                    self.device = "mps"  # Apple Silicon
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            else:
                self.device = self.config.device
                
            print(f"Using device: {self.device}")
            
        except ImportError:
            print("PyTorch not available, falling back to CPU")
            self.device = "cpu"
    
    def _load_model(self):
        """Load the model and tokenizer with optimizations."""
        # Check if we're in simulation mode
        if self.config.simulation_mode:
            print(f"Simulation mode: Mocking model {self.config.model_name}")
            self.model = "mock_model"
            self.tokenizer = "mock_tokenizer"
            return
            
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print(f"Loading model: {self.config.model_name}")
            
            # Setup model loading arguments
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "trust_remote_code": True,
            }
            
            # Add quantization if specified
            if self.config.quantization == "4bit":
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    print("Using 4-bit quantization")
                except ImportError:
                    print("BitsAndBytesConfig not available, loading without quantization")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Move to device if not using quantization
            if self.config.quantization is None and self.device != "cpu":
                self.model = self.model.to(self.device)
            
            print(f"Model loaded successfully on {self.device}")
            
        except ImportError as e:
            raise ImportError(f"Required dependencies not available: {e}. Please install transformers and torch.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.config.model_name}: {e}")
    
    def _format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a prompt suitable for code generation models."""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    def _generate_response(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate response using the loaded model."""
        # Check if we're in simulation mode
        if self.config.simulation_mode:
            # Return a simulated response based on the model type
            if "codellama" in self.config.model_name.lower():
                return """# Here's a Python function based on your request:

def solve_problem():
    '''
    This function demonstrates CodeLlama's ability to generate
    clean, well-documented Python code.
    '''
    print("CodeLlama generated solution")
    return "success"

# Example usage:
if __name__ == "__main__":
    result = solve_problem()
    print(f"Result: {result}")"""
            elif "wizard" in self.config.model_name.lower():
                return """# WizardCoder Solution:

class Solution:
    def __init__(self):
        self.name = "WizardCoder"
    
    def generate_code(self, requirements):
        '''
        WizardCoder specializes in creating optimized solutions
        '''
        return f"Optimized solution for: {requirements}"

# Implementation
solution = Solution()
result = solution.generate_code("user requirements")
print(result)"""
            else:
                return f"This is a simulated response from {self.config.model_name}. The model would generate appropriate code here."
        
        try:
            import torch
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048  # Leave room for generation
            )
            
            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generation parameters
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": self.config.top_p,
                "do_sample": self.config.do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                **kwargs
            }
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")
    
    def _format_as_openai_response(self, response_text: str) -> Dict[str, Any]:
        """Format response in OpenAI ChatCompletion format."""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # Simplified for now
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(response_text.split())
            }
        }
    
    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = None, 
                       temperature: float = None, **kwargs) -> Dict[str, Any]:
        """
        OpenAI-compatible chat completion interface.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Dict in OpenAI ChatCompletion format
        """
        # Use config defaults if not specified
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        # Format messages to prompt
        prompt = self._format_messages_to_prompt(messages)
        
        # Generate response
        response_text = self._generate_response(prompt, max_tokens, temperature, **kwargs)
        
        # Format as OpenAI response
        return self._format_as_openai_response(response_text)


def create_local_llm_interface(model_name: str = "codellama/CodeLlama-7b-Python-hf") -> LocalLLMInterface:
    """Factory function to create LocalLLMInterface with sensible defaults."""
    config = LocalLLMConfig(model_name=model_name)
    return LocalLLMInterface(config)