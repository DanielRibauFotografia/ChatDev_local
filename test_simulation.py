#!/usr/bin/env python3
"""
End-to-end simulation test for local LLM interface.
Tests the complete pipeline using simulation mode.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_local_llm_simulation():
    """Test local LLM in simulation mode."""
    print("Testing Local LLM in simulation mode...")
    
    try:
        from camel.typing import ModelType
        from camel.model_backend import ModelFactory
        
        # Test with simulation mode enabled
        model_config = {
            "max_tokens": 512,
            "temperature": 0.7,
            "simulation_mode": True
        }
        
        # Test CodeLlama
        backend = ModelFactory.create(ModelType.LOCAL_CODELLAMA_7B, model_config)
        
        messages = [
            {"role": "system", "content": "You are a helpful programming assistant."},
            {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."}
        ]
        
        response = backend.run(messages=messages, max_tokens=512, temperature=0.7)
        
        # Validate response
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "message" in response["choices"][0]
        assert "content" in response["choices"][0]["message"]
        
        content = response["choices"][0]["message"]["content"]
        print(f"✓ CodeLlama simulation response: {content[:100]}...")
        
        # Test WizardCoder
        backend2 = ModelFactory.create(ModelType.LOCAL_WIZARDCODER_7B, model_config)
        response2 = backend2.run(messages=messages)
        
        content2 = response2["choices"][0]["message"]["content"]
        print(f"✓ WizardCoder simulation response: {content2[:100]}...")
        
        # Verify different models give different responses
        assert content != content2, "Different models should give different responses"
        
        return True
        
    except Exception as e:
        print(f"✗ Simulation test failed: {e}")
        return False

def test_chatdev_integration_simulation():
    """Test integration with ChatDev using simulation mode."""
    print("\nTesting ChatDev integration with simulation...")
    
    try:
        from camel.typing import ModelType
        from camel.model_backend import ModelFactory
        
        # Simulate a software development conversation
        model_config = {"simulation_mode": True, "max_tokens": 256}
        backend = ModelFactory.create(ModelType.LOCAL_CODELLAMA_7B, model_config)
        
        # CEO role conversation
        ceo_messages = [
            {"role": "system", "content": "You are a CEO planning a software project."},
            {"role": "user", "content": "We need to develop a simple calculator app. What are the requirements?"}
        ]
        
        ceo_response = backend.run(messages=ceo_messages)
        print(f"✓ CEO simulation: {ceo_response['choices'][0]['message']['content'][:80]}...")
        
        # Programmer role conversation  
        programmer_messages = [
            {"role": "system", "content": "You are a programmer implementing software."},
            {"role": "user", "content": "Create a Python calculator class with basic operations."}
        ]
        
        programmer_response = backend.run(messages=programmer_messages)
        print(f"✓ Programmer simulation: {programmer_response['choices'][0]['message']['content'][:80]}...")
        
        # Tester role conversation
        tester_messages = [
            {"role": "system", "content": "You are a software tester writing test cases."},
            {"role": "user", "content": "Write unit tests for the calculator class."}
        ]
        
        tester_response = backend.run(messages=tester_messages)
        print(f"✓ Tester simulation: {tester_response['choices'][0]['message']['content'][:80]}...")
        
        # All responses should contain code-related content
        all_responses = [ceo_response, programmer_response, tester_response]
        for response in all_responses:
            content = response['choices'][0]['message']['content']
            assert any(keyword in content.lower() for keyword in ['def', 'class', 'function', 'code', 'python']), \
                "Response should be code-related"
        
        return True
        
    except Exception as e:
        print(f"✗ ChatDev integration test failed: {e}")
        return False

def test_command_line_simulation():
    """Test command line argument parsing for local models."""
    print("\nTesting command line simulation...")
    
    try:
        # Import the args2type mapping
        with open('run.py', 'r') as f:
            content = f.read()
        
        # Verify all local models are in the mapping
        local_models = ['LOCAL_CODELLAMA_7B', 'LOCAL_CODELLAMA_13B', 'LOCAL_WIZARDCODER_7B']
        
        for model in local_models:
            assert f"'{model}'" in content, f"{model} not found in run.py args2type mapping"
            print(f"✓ {model} is available as command line option")
        
        # Test help text includes local models
        assert 'LOCAL_CODELLAMA_7B' in content, "Help text should mention local models"
        print("✓ Command line help includes local models")
        
        return True
        
    except Exception as e:
        print(f"✗ Command line test failed: {e}")
        return False

def main():
    """Run simulation tests."""
    print("=== Local LLM Simulation Test Suite ===\n")
    
    tests = [
        test_local_llm_simulation,
        test_chatdev_integration_simulation,
        test_command_line_simulation
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
    
    print(f"\n=== Simulation Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All simulation tests passed!")
        print("\n📋 Next Steps:")
        print("1. Install local LLM dependencies:")
        print("   pip install torch transformers accelerate bitsandbytes")
        print("\n2. Test with actual local model:")
        print("   python run.py --model LOCAL_CODELLAMA_7B --task 'Create a simple web app'")
        print("\n3. Monitor memory usage and adjust quantization settings as needed")
        return True
    else:
        print(f"\n❌ {failed} simulation test(s) failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)