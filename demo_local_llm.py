#!/usr/bin/env python3
"""
Demonstration script showing ChatDev working with local LLM models.
This script can run in both simulation mode and with actual models.
"""

import sys
import os
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demonstrate_local_llm(model_name="LOCAL_CODELLAMA_7B", simulation=True):
    """Demonstrate local LLM functionality."""
    print(f"=== ChatDev Local LLM Demo ===")
    print(f"Model: {model_name}")
    print(f"Simulation Mode: {simulation}")
    print()
    
    try:
        from camel.typing import ModelType
        from camel.model_backend import ModelFactory
        
        # Get model type
        model_mapping = {
            'LOCAL_CODELLAMA_7B': ModelType.LOCAL_CODELLAMA_7B,
            'LOCAL_CODELLAMA_13B': ModelType.LOCAL_CODELLAMA_13B,
            'LOCAL_WIZARDCODER_7B': ModelType.LOCAL_WIZARDCODER_7B,
        }
        
        if model_name not in model_mapping:
            print(f"❌ Unknown model: {model_name}")
            print(f"Available models: {list(model_mapping.keys())}")
            return False
        
        model_type = model_mapping[model_name]
        
        # Create model config
        model_config = {
            "max_tokens": 512,
            "temperature": 0.3,  # Lower temperature for more deterministic code
            "simulation_mode": simulation
        }
        
        print("🔧 Creating model backend...")
        backend = ModelFactory.create(model_type, model_config)
        print("✅ Model backend created successfully!")
        print()
        
        # Simulate ChatDev software development process
        development_phases = [
            {
                "role": "CEO",
                "system": "You are a CEO defining software requirements.",
                "task": "Define requirements for a simple todo list application."
            },
            {
                "role": "CTO", 
                "system": "You are a CTO designing system architecture.",
                "task": "Design the architecture for a Python todo list app with file storage."
            },
            {
                "role": "Programmer",
                "system": "You are a programmer implementing software.",
                "task": "Implement a Python todo list class with add, remove, and list methods."
            },
            {
                "role": "Tester",
                "system": "You are a software tester creating test cases.",
                "task": "Write unit tests for the todo list class."
            },
            {
                "role": "Reviewer",
                "system": "You are a code reviewer checking code quality.",
                "task": "Review the todo list implementation and suggest improvements."
            }
        ]
        
        print("🚀 Starting software development process...\n")
        
        for i, phase in enumerate(development_phases, 1):
            print(f"Phase {i}: {phase['role']}")
            print("-" * 40)
            
            messages = [
                {"role": "system", "content": phase['system']},
                {"role": "user", "content": phase['task']}
            ]
            
            print(f"Task: {phase['task']}")
            print("Generating response...")
            
            response = backend.run(messages=messages, max_tokens=512, temperature=0.3)
            
            if "choices" in response and response["choices"]:
                content = response["choices"][0]["message"]["content"]
                print(f"\n{phase['role']} Response:")
                print(content[:500] + ("..." if len(content) > 500 else ""))
                print()
            else:
                print("❌ No response generated")
                return False
        
        print("✅ Software development process completed!")
        print("\n📊 Summary:")
        print(f"- Model used: {model_type.value}")
        print(f"- Simulation mode: {simulation}")
        print(f"- Phases completed: {len(development_phases)}")
        print("- All agents responded successfully")
        
        if simulation:
            print("\n💡 To test with real models:")
            print("1. Install dependencies: pip install torch transformers accelerate")
            print("2. Run without --simulation flag")
            print("3. First run will download the model (several GB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description='ChatDev Local LLM Demo')
    parser.add_argument('--model', type=str, 
                       choices=['LOCAL_CODELLAMA_7B', 'LOCAL_CODELLAMA_13B', 'LOCAL_WIZARDCODER_7B'],
                       default='LOCAL_CODELLAMA_7B',
                       help='Local model to use')
    parser.add_argument('--simulation', action='store_true', default=True,
                       help='Run in simulation mode (default: True)')
    parser.add_argument('--real', action='store_true', default=False,
                       help='Run with real models (overrides --simulation)')
    
    args = parser.parse_args()
    
    # If --real is specified, turn off simulation
    simulation = args.simulation and not args.real
    
    success = demonstrate_local_llm(args.model, simulation)
    
    if success:
        print("\n🎉 Demo completed successfully!")
        if simulation:
            print("\n🔄 Try running with --real to test actual models")
    else:
        print("\n💥 Demo failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)