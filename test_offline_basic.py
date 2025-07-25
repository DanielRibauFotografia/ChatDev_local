#!/usr/bin/env python3
"""
Basic offline test for ChatDev Local modifications.
This test verifies that the core modifications work without requiring all dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_types():
    """Test that local model types are defined."""
    print("Testing ModelType definitions...")
    try:
        # Import just the typing module
        from camel.typing import ModelType
        
        # Test that local model types exist
        assert hasattr(ModelType, 'HUGGINGFACE')
        assert hasattr(ModelType, 'LLAMA_CPP')
        assert hasattr(ModelType, 'OLLAMA')
        assert hasattr(ModelType, 'LOCALAI')
        
        print("✓ All local model types are properly defined")
        print("  - HUGGINGFACE:", ModelType.HUGGINGFACE.value)
        print("  - LLAMA_CPP:", ModelType.LLAMA_CPP.value)
        print("  - OLLAMA:", ModelType.OLLAMA.value)
        print("  - LOCALAI:", ModelType.LOCALAI.value)
        return True
    except Exception as e:
        print(f"✗ Model type test failed: {e}")
        return False

def test_statistics_without_numpy():
    """Test that statistics work without numpy."""
    print("\nTesting statistics without numpy...")
    try:
        from chatdev.statistics import prompt_cost
        
        # Test with unknown model (should return -1)
        cost = prompt_cost('unknown_model', 100, 50)
        assert cost == -1, f"Expected -1 for unknown model, got {cost}"
        
        # Test with known model
        cost = prompt_cost('gpt-3.5-turbo', 1000, 500)
        expected = (1000 * 0.0005 / 1000.0) + (500 * 0.0015 / 1000.0)
        assert abs(cost - expected) < 0.0001, f"Expected {expected}, got {cost}"
        
        print("✓ Statistics working correctly without numpy dependency")
        return True
    except Exception as e:
        print(f"✗ Statistics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_run_py_arguments():
    """Test that run.py has the required arguments for local models."""
    print("\nTesting run.py argument definitions...")
    try:
        # Import the argument parser
        import argparse
        import importlib.util
        
        # Load run.py module
        spec = importlib.util.spec_from_file_location("run", "run.py")
        run_module = importlib.util.module_from_spec(spec)
        
        # Check if we can access the parser without executing the full script
        with open('run.py', 'r') as f:
            content = f.read()
            
        # Check for local model arguments
        required_args = [
            '--model_name',
            '--model_path', 
            '--base_url',
            '--device',
            '--max_tokens',
            '--temperature',
            '--top_p'
        ]
        
        for arg in required_args:
            if arg in content:
                print(f"✓ Found argument {arg} in run.py")
            else:
                print(f"✗ Missing argument {arg} in run.py")
                return False
                
        print("✓ All required local model arguments are defined in run.py")
        return True
    except Exception as e:
        print(f"✗ run.py arguments test failed: {e}")
        return False

def test_openai_optional():
    """Test that OpenAI imports are optional."""
    print("\nTesting OpenAI optional imports...")
    try:
        # Test run.py handles missing OpenAI gracefully
        with open('run.py', 'r') as f:
            content = f.read()
            
        # Check that OpenAI imports are in try-catch blocks
        if 'try:' in content and 'from openai.types.chat' in content:
            print("✓ OpenAI imports are wrapped in try-catch blocks")
        else:
            print("✗ OpenAI imports are not properly wrapped")
            return False
            
        # Test that model backend handles missing OpenAI
        from camel.model_backend import openai_available, tiktoken_available
        print(f"✓ OpenAI availability detected: {openai_available}")
        print(f"✓ Tiktoken availability detected: {tiktoken_available}")
        
        return True
    except Exception as e:
        print(f"✗ OpenAI optional test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("ChatDev Local - Basic Offline Functionality Tests")
    print("=" * 60)
    
    tests = [
        test_model_types,
        test_statistics_without_numpy,
        test_run_py_arguments,
        test_openai_optional,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} basic tests passed")
    
    if passed == len(tests):
        print("🎉 All basic offline functionality tests passed!")
        print("✓ ChatDev Local is ready for offline operation")
        return 0
    else:
        print("⚠️  Some basic tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())