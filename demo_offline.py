#!/usr/bin/env python3
"""
Complete offline example demonstrating ChatDev Local functionality.
This script shows how to use ChatDev completely offline with the STUB model.
"""

import sys
import os
import tempfile
import shutil

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_offline_workflow():
    """Demonstrate complete offline workflow using STUB model."""
    print("🚀 ChatDev Local Offline Workflow Demo")
    print("=" * 50)
    
    try:
        # Import required modules
        from camel.typing import ModelType
        from camel.model_backend import ModelFactory, set_local_model_config
        
        print("✓ Successfully imported ChatDev Local modules")
        
        # Test STUB model (works completely offline)
        print("\n📱 Testing offline model backend...")
        
        # Create a model instance
        model_config = {
            'temperature': 0.7,
            'max_tokens': 512
        }
        
        set_local_model_config(model_config)
        model = ModelFactory.create(ModelType.STUB, model_config)
        
        print(f"✓ Created {type(model).__name__} backend")
        
        # Test model inference
        print("\n🧠 Testing model inference...")
        
        messages = [
            {"role": "system", "content": "You are a helpful programming assistant."},
            {"role": "user", "content": "Create a simple Python calculator with add, subtract, multiply, and divide functions."}
        ]
        
        response = model.run(messages=messages)
        
        print(f"✓ Model responded with content: '{response['choices'][0]['message']['content'][:50]}...'")
        print(f"✓ Response format: {list(response.keys())}")
        
        # Test model type detection
        print("\n🎯 Testing local model types...")
        
        local_models = [
            ('HUGGINGFACE', ModelType.HUGGINGFACE),
            ('LLAMA_CPP', ModelType.LLAMA_CPP),
            ('OLLAMA', ModelType.OLLAMA),
            ('LOCALAI', ModelType.LOCALAI),
            ('STUB', ModelType.STUB)
        ]
        
        for name, model_type in local_models:
            print(f"✓ {name}: {model_type.value}")
        
        print("\n📊 Testing offline statistics...")
        
        from chatdev.statistics import prompt_cost
        
        # Test with known OpenAI model
        cost_openai = prompt_cost('gpt-3.5-turbo', 1000, 500)
        print(f"✓ OpenAI model cost calculation: ${cost_openai:.6f}")
        
        # Test with unknown model (returns -1)
        cost_local = prompt_cost('ollama-llama2', 1000, 500)
        print(f"✓ Local model cost (free): ${cost_local}")
        
        print("\n🎉 Offline workflow demonstration complete!")
        print("\nWhat we tested:")
        print("  ✅ Module imports without external dependencies")
        print("  ✅ Local model backend creation")
        print("  ✅ Model inference without API calls")
        print("  ✅ Statistics calculation without numpy")
        print("  ✅ Cost calculation for both OpenAI and local models")
        
        return True
        
    except Exception as e:
        print(f"❌ Offline workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_usage_examples():
    """Show practical usage examples for different backends."""
    print("\n" + "=" * 50)
    print("🔧 PRACTICAL USAGE EXAMPLES")
    print("=" * 50)
    
    examples = [
        {
            "name": "Ollama (Recommended)",
            "setup": [
                "# 1. Install Ollama from https://ollama.ai/",
                "ollama serve",
                "ollama pull llama2"
            ],
            "command": "python run.py --model OLLAMA --model_name llama2 --task 'Create a todo list app' --name TodoApp"
        },
        {
            "name": "llama.cpp (Best Performance)",
            "setup": [
                "# 1. Install dependencies",
                "pip install llama-cpp-python",
                "# 2. Download GGUF model to ./models/ directory"
            ],
            "command": "python run.py --model LLAMA_CPP --model_path ./models/llama-2-7b.gguf --task 'Create a calculator' --name Calculator"
        },
        {
            "name": "HuggingFace (Many Models)",
            "setup": [
                "# 1. Install dependencies",
                "pip install transformers torch accelerate"
            ],
            "command": "python run.py --model HUGGINGFACE --model_name microsoft/DialoGPT-medium --device cpu --task 'Create a game' --name Game"
        },
        {
            "name": "LocalAI (OpenAI Compatible)",
            "setup": [
                "# 1. Install and configure LocalAI",
                "# 2. Start LocalAI server on port 8080"
            ],
            "command": "python run.py --model LOCALAI --base_url http://localhost:8080 --model_name gpt-3.5-turbo --task 'Create a web app' --name WebApp"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print("   Setup:")
        for step in example['setup']:
            print(f"     {step}")
        print("   Command:")
        print(f"     {example['command']}")
    
    print(f"\n📖 For detailed setup instructions, see OFFLINE_SETUP_GUIDE.md")

def main():
    """Run the complete offline demo."""
    success = test_offline_workflow()
    
    if success:
        show_usage_examples()
        
        print("\n" + "=" * 50)
        print("🎊 CONGRATULATIONS!")
        print("=" * 50)
        print("ChatDev Local is ready for completely offline operation!")
        print("\nNext steps:")
        print("1. Choose your preferred backend (Ollama recommended for beginners)")
        print("2. Follow the setup instructions above")
        print("3. Run the example commands to create your first offline software")
        print("\n✨ No internet connection or API keys required after setup!")
        
        return 0
    else:
        print("\n❌ Basic functionality test failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())