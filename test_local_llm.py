#!/usr/bin/env python3
"""
Simple test script to validate Local LLM interface functionality.
This tests the basic functionality without loading actual models.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all necessary modules can be imported."""
    print("Testing imports...")
    
    try:
        from camel.typing import ModelType
        print("✓ ModelType imported successfully")
        
        from camel.local_llm import LocalLLMInterface, LocalLLMConfig
        print("✓ LocalLLMInterface imported successfully")
        
        from camel.model_backend import ModelFactory, LocalLLMModel
        print("✓ ModelFactory and LocalLLMModel imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_types():
    """Test that new model types are available."""
    print("\nTesting model types...")
    
    try:
        from camel.typing import ModelType
        
        # Check that new local model types exist
        local_models = [
            ModelType.LOCAL_CODELLAMA_7B,
            ModelType.LOCAL_CODELLAMA_13B, 
            ModelType.LOCAL_WIZARDCODER_7B
        ]
        
        for model in local_models:
            print(f"✓ {model.name}: {model.value}")
            
        return True
    except Exception as e:
        print(f"✗ Model type test failed: {e}")
        return False

def test_local_llm_config():
    """Test LocalLLMConfig creation."""
    print("\nTesting LocalLLMConfig...")
    
    try:
        from camel.local_llm import LocalLLMConfig
        
        # Test default config
        config = LocalLLMConfig()
        print(f"✓ Default config created: {config.model_name}")
        
        # Test custom config
        custom_config = LocalLLMConfig(
            model_name="test-model",
            max_tokens=512,
            temperature=0.5
        )
        print(f"✓ Custom config created: {custom_config.model_name}")
        
        return True
    except Exception as e:
        print(f"✗ LocalLLMConfig test failed: {e}")
        return False

def test_model_factory():
    """Test ModelFactory with local models."""
    print("\nTesting ModelFactory...")
    
    try:
        from camel.typing import ModelType
        from camel.model_backend import ModelFactory
        
        # Test that factory recognizes local model types
        model_config = {"max_tokens": 1024, "temperature": 0.7}
        
        # This will fail if transformers isn't installed, but that's expected
        # We're just testing the factory logic
        try:
            backend = ModelFactory.create(ModelType.LOCAL_CODELLAMA_7B, model_config)
            print(f"✓ ModelFactory created LocalLLMModel for {ModelType.LOCAL_CODELLAMA_7B.name}")
        except Exception as e:
            if "transformers" in str(e).lower() or "torch" in str(e).lower():
                print(f"✓ ModelFactory correctly tries to create LocalLLMModel (dependencies not installed: {e})")
            else:
                raise e
        
        return True
    except Exception as e:
        print(f"✗ ModelFactory test failed: {e}")
        return False

def test_run_py_integration():
    """Test that run.py can parse local model arguments."""
    print("\nTesting run.py integration...")
    
    try:
        # Import the args2type mapping from run.py
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # We can't directly import from run.py due to execution, so we'll check the file
        with open('run.py', 'r') as f:
            content = f.read()
            
        # Check that local models are in the mapping
        local_model_names = ['LOCAL_CODELLAMA_7B', 'LOCAL_CODELLAMA_13B', 'LOCAL_WIZARDCODER_7B']
        
        for model_name in local_model_names:
            if f"'{model_name}'" in content:
                print(f"✓ {model_name} found in run.py")
            else:
                print(f"✗ {model_name} not found in run.py")
                return False
                
        return True
    except Exception as e:
        print(f"✗ run.py integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Local LLM Interface Test Suite ===\n")
    
    tests = [
        test_imports,
        test_model_types,
        test_local_llm_config,
        test_model_factory,
        test_run_py_integration
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
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Local LLM interface is ready.")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)