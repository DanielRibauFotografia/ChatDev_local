#!/bin/bash
# Example scripts for running ChatDev with different local models

echo "ChatDev Local Model Examples"
echo "============================"

# Example 1: Using Ollama with Llama 2
echo "Example 1: Ollama with Llama 2"
echo "python run.py \\"
echo "  --model OLLAMA \\"
echo "  --model_name llama2 \\"
echo "  --temperature 0.8 \\"
echo "  --task 'Build a simple calculator app with GUI'"
echo ""

# Example 2: Using HuggingFace model with GPU acceleration
echo "Example 2: HuggingFace with MPS (Apple Silicon GPU)"
echo "python run.py \\"
echo "  --model HUGGINGFACE \\"
echo "  --model_name microsoft/DialoGPT-medium \\"
echo "  --device mps \\"
echo "  --temperature 0.7 \\"
echo "  --task 'Create a todo list application'"
echo ""

# Example 3: Using llama.cpp with local GGUF file
echo "Example 3: llama.cpp with local model file"
echo "python run.py \\"
echo "  --model LLAMA_CPP \\"
echo "  --model_path ./models/llama-2-7b-chat.gguf \\"
echo "  --temperature 0.7 \\"
echo "  --max_tokens 1024 \\"
echo "  --task 'Develop a weather app'"
echo ""

# Example 4: Using LocalAI
echo "Example 4: LocalAI with OpenAI-compatible API"
echo "python run.py \\"
echo "  --model LOCALAI \\"
echo "  --base_url http://localhost:8080 \\"
echo "  --model_name gpt-3.5-turbo \\"
echo "  --task 'Create a file manager application'"
echo ""

echo "Note: Make sure the respective services (Ollama, LocalAI) are running"
echo "and required dependencies are installed before running these examples."