# ChatDev Local LLM Integration

This document explains how to use ChatDev with local LLM models for offline operation on macOS ARM64.

## Overview

ChatDev now supports multiple local LLM backends:
- **HuggingFace Transformers**: Local inference with popular models
- **llama.cpp**: Efficient C++ implementation for GGUF models
- **Ollama**: Local server for easy model management
- **LocalAI**: OpenAI-compatible local API server

## Installation

### Base Requirements
```bash
pip install -r requirements.txt
```

### Local Model Dependencies

Choose and install dependencies for your preferred backend:

#### HuggingFace Transformers
```bash
pip install transformers torch accelerate
```

#### llama.cpp
```bash
pip install llama-cpp-python
```

#### Ollama
1. Install Ollama from https://ollama.ai/
2. Start the Ollama service:
   ```bash
   ollama serve
   ```
3. Pull a model:
   ```bash
   ollama pull llama2
   ```

#### LocalAI
1. Install LocalAI from https://localai.io/
2. Start LocalAI server on default port 8080

## Usage

### Command Line Arguments

ChatDev now supports these additional arguments for local models:

```bash
python run.py \
  --model OLLAMA \
  --model_name llama2 \
  --base_url http://localhost:11434 \
  --temperature 0.7 \
  --max_tokens 512 \
  --task "Create a simple calculator app"
```

### Model-Specific Arguments

| Backend | Required Arguments | Example |
|---------|-------------------|---------|
| HUGGINGFACE | --model_name | `--model HUGGINGFACE --model_name microsoft/DialoGPT-medium` |
| LLAMA_CPP | --model_path | `--model LLAMA_CPP --model_path ./models/llama-2-7b.gguf` |
| OLLAMA | --model_name, --base_url (optional) | `--model OLLAMA --model_name llama2` |
| LOCALAI | --model_name, --base_url (optional) | `--model LOCALAI --model_name gpt-3.5-turbo` |

### Common Arguments

- `--device`: Device for inference (auto, cpu, cuda, mps) - for HuggingFace models
- `--temperature`: Sampling temperature (0.0-2.0)
- `--max_tokens`: Maximum tokens to generate
- `--top_p`: Top-p sampling parameter

## Examples

### Using Ollama with Llama 2
```bash
python run.py \
  --model OLLAMA \
  --model_name llama2 \
  --temperature 0.8 \
  --task "Build a todo list application"
```

### Using HuggingFace Model
```bash
python run.py \
  --model HUGGINGFACE \
  --model_name microsoft/DialoGPT-medium \
  --device mps \
  --task "Create a weather app"
```

### Using llama.cpp with Local GGUF File
```bash
python run.py \
  --model LLAMA_CPP \
  --model_path ./models/llama-2-7b-chat.gguf \
  --temperature 0.7 \
  --task "Develop a simple game"
```

### Using LocalAI
```bash
python run.py \
  --model LOCALAI \
  --base_url http://localhost:8080 \
  --model_name gpt-3.5-turbo \
  --task "Create a file manager"
```

## macOS ARM64 Optimization

For Apple Silicon Macs:

1. **HuggingFace Models**: Use `--device mps` for GPU acceleration
2. **llama.cpp**: Automatically optimized for ARM64 with Metal support
3. **Ollama**: Native ARM64 builds available
4. **Memory Management**: Start with smaller models for testing

## Model Recommendations

### For Development (Fast, Good Quality)
- Ollama with `llama2:7b`
- HuggingFace `microsoft/DialoGPT-medium`

### For Production (High Quality)
- Ollama with `llama2:13b` or `codellama:13b`
- HuggingFace larger models like `facebook/blenderbot-3B`

### For Code Generation
- Ollama with `codellama` models
- HuggingFace `Salesforce/codegen-*` models

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce model size or use smaller models
2. **Slow Performance**: Use GPU acceleration (`--device mps` on macOS)
3. **Connection Errors**: Ensure Ollama/LocalAI services are running
4. **Import Errors**: Install the required dependencies for your chosen backend

### Performance Tips

1. Use smaller models for faster iteration during development
2. Enable GPU acceleration when available
3. Adjust `max_tokens` based on your use case
4. Use appropriate `temperature` values (0.7-1.0 for creative tasks, 0.1-0.3 for factual tasks)

## Architecture

The local LLM integration maintains full compatibility with ChatDev's multi-agent architecture:
- All agent roles (CEO, CTO, Programmer, Tester, Reviewer) work with local models
- Complete development pipeline preserved (analysis, design, coding, testing, documentation)
- Web visualizer supports local model outputs
- Existing configuration and customization options remain available

Local models integrate at the model backend layer, ensuring no changes to the core ChatDev workflow or agent communication patterns.