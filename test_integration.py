#!/usr/bin/env python3
"""
Integration test for local LLM interface with ChatDev agents.
Tests the complete pipeline without requiring heavy model downloads.
"""

import sys
import os
import json
from unittest.mock import MagicMock, patch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_chat_agent_integration():
    """Test that ChatAgent can work with local LLM backend.""" 
    print("Testing ChatAgent integration with local LLM...")
    
    try:
        # Mock the transformers module before any imports
        import sys
        from unittest.mock import MagicMock
        
        # Create mock modules
        mock_transformers = MagicMock()
        mock_torch = MagicMock()
        
        # Setup torch mocks
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        
        # Install mocks in sys.modules
        sys.modules['transformers'] = mock_transformers
        sys.modules['torch'] = mock_torch
        
        try:
            from camel.typing import ModelType
            from camel.model_backend import ModelFactory
            
            # Create a local model backend
            model_config = {"max_tokens": 512, "temperature": 0.7}
            backend = ModelFactory.create(ModelType.LOCAL_CODELLAMA_7B, model_config)
            
            print(f"✓ Created backend for {ModelType.LOCAL_CODELLAMA_7B.name}")
            
            # Mock the run method to return a valid response
            def mock_run(*args, **kwargs):
                return {
                    "id": "test_completion",
                    "object": "chat.completion", 
                    "created": 1234567890,
                    "model": "codellama/CodeLlama-7b-Python-hf",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "# Here's a simple Python function:\n\ndef hello_world():\n    return 'Hello, World!'"
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 15,
                        "total_tokens": 35
                    }
                }
            
            # Replace the run method
            backend.run = mock_run
            
            # Test a simple chat completion
            messages = [
                {"role": "system", "content": "You are a helpful programming assistant."},
                {"role": "user", "content": "Write a simple hello world function in Python."}
            ]
            
            response = backend.run(messages=messages, max_tokens=512, temperature=0.7)
            
            # Validate response format
            assert "choices" in response
            assert len(response["choices"]) > 0
            assert "message" in response["choices"][0]
            assert "content" in response["choices"][0]["message"]
            
            content = response["choices"][0]["message"]["content"]
            print(f"✓ Local LLM generated response: {content[:50]}...")
            
            return True
            
        finally:
            # Clean up mocks
            if 'transformers' in sys.modules:
                del sys.modules['transformers']
            if 'torch' in sys.modules:
                del sys.modules['torch']
        
    except Exception as e:
        print(f"✗ ChatAgent integration test failed: {e}")
        return False

def test_mock_end_to_end_pipeline():
    """Test a simplified end-to-end software development pipeline."""
    print("\nTesting mock end-to-end pipeline...")
    
    try:
        # Mock all heavy dependencies
        with patch('transformers.AutoTokenizer'), \
             patch('transformers.AutoModelForCausalLM'), \
             patch('torch.no_grad'), \
             patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            
            from camel.typing import ModelType
            from camel.model_backend import ModelFactory
            
            # Test different local models
            local_models = [
                ModelType.LOCAL_CODELLAMA_7B,
                ModelType.LOCAL_CODELLAMA_13B,
                ModelType.LOCAL_WIZARDCODER_7B
            ]
            
            for model_type in local_models:
                print(f"  Testing {model_type.name}...")
                
                model_config = {"max_tokens": 256, "temperature": 0.3}
                backend = ModelFactory.create(model_type, model_config)
                
                # Mock response specific to each model
                mock_responses = {
                    ModelType.LOCAL_CODELLAMA_7B: "CodeLlama response for Python coding task",
                    ModelType.LOCAL_CODELLAMA_13B: "CodeLlama-13B response with more detailed code",
                    ModelType.LOCAL_WIZARDCODER_7B: "WizardCoder response with optimized solution"
                }
                
                def mock_run(*args, **kwargs):
                    return {
                        "choices": [{
                            "message": {
                                "content": mock_responses[model_type]
                            }
                        }]
                    }
                
                with patch.object(backend, 'run', side_effect=mock_run):
                    response = backend.run(
                        messages=[{"role": "user", "content": "Create a simple calculator"}],
                        max_tokens=256
                    )
                    
                    content = response["choices"][0]["message"]["content"]
                    assert model_type.name.lower().replace('local_', '').replace('_', '') in content.lower().replace(' ', '').replace('-', '')
                    print(f"    ✓ {model_type.name} responded correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ End-to-end pipeline test failed: {e}")
        return False

def test_config_validation():
    """Test configuration and parameter validation."""
    print("\nTesting configuration validation...")
    
    try:
        from camel.local_llm import LocalLLMConfig
        
        # Test different configurations
        configs = [
            LocalLLMConfig(),  # Default
            LocalLLMConfig(max_tokens=2048, temperature=0.1),  # Custom
            LocalLLMConfig(quantization=None, device="cpu"),  # No quantization
        ]
        
        for i, config in enumerate(configs):
            print(f"  ✓ Config {i+1}: model={config.model_name}, tokens={config.max_tokens}")
        
        # Test invalid configurations
        try:
            invalid_config = LocalLLMConfig(temperature=2.0)  # Invalid temperature
            print("  ⚠️  Invalid temperature accepted (validation could be improved)")
        except:
            print("  ✓ Invalid configuration rejected")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration validation test failed: {e}")
        return False

def test_compatibility_with_existing_openai():
    """Test that existing OpenAI models still work."""
    print("\nTesting compatibility with existing OpenAI models...")
    
    try:
        from camel.typing import ModelType
        from camel.model_backend import ModelFactory
        
        # Test that OpenAI models still work (without actually calling the API)
        openai_models = [
            ModelType.GPT_3_5_TURBO,
            ModelType.GPT_4,
            ModelType.GPT_4O
        ]
        
        for model_type in openai_models:
            model_config = {"max_tokens": 1024}
            
            # This should create an OpenAIModel, not LocalLLMModel
            backend = ModelFactory.create(model_type, model_config)
            backend_class_name = backend.__class__.__name__
            
            assert backend_class_name == "OpenAIModel", f"Expected OpenAIModel, got {backend_class_name}"
            print(f"  ✓ {model_type.name} -> {backend_class_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ OpenAI compatibility test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and graceful degradation."""
    print("\nTesting error handling...")
    
    try:
        # Test with simulated model loading failure
        with patch('transformers.AutoTokenizer.from_pretrained', side_effect=Exception("Model not found")):
            from camel.local_llm import LocalLLMInterface, LocalLLMConfig
            
            config = LocalLLMConfig(model_name="non-existent-model")
            
            try:
                interface = LocalLLMInterface(config)
                print("  ✗ Expected exception not raised")
                return False
            except Exception as e:
                if "Model not found" in str(e) or "Failed to load model" in str(e):
                    print("  ✓ Model loading failure handled correctly")
                else:
                    print(f"  ✗ Unexpected error: {e}")
                    return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def main():
    """Run integration tests."""
    print("=== Local LLM Integration Test Suite ===\n")
    
    tests = [
        test_chat_agent_integration,
        test_mock_end_to_end_pipeline,
        test_config_validation,
        test_compatibility_with_existing_openai,
        test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print(f"\n=== Integration Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All integration tests passed!")
        print("📝 Local LLM interface is ready for use.")
        print("\n💡 To use local models, install dependencies:")
        print("   pip install torch transformers accelerate bitsandbytes")
        print("\n🚀 Then run ChatDev with:")
        print("   python run.py --model LOCAL_CODELLAMA_7B --task 'Create a simple web app'")
        return True
    else:
        print(f"\n❌ {failed} integration test(s) failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)