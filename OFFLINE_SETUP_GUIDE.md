# ChatDev Local - Complete Offline Setup Guide

## Overview

This guide will help you set up ChatDev to run completely offline using local LLM models. No internet connection or OpenAI API keys required after setup!

## System Requirements

- Python 3.9 or higher
- 4GB+ RAM (8GB+ recommended for larger models)
- 2GB+ storage for models
- Windows, macOS, or Linux

## Step 1: Download and Setup ChatDev Local

```bash
# Clone the repository
git clone https://github.com/DanielRibauFotografia/ChatDev_local.git
cd ChatDev_local

# Create a Python virtual environment (recommended)
python -m venv chatdev_env
source chatdev_env/bin/activate  # On Windows: chatdev_env\Scripts\activate

# Install basic dependencies
pip install tenacity tiktoken requests openai markdown
```

## Step 2: Choose Your Local LLM Backend

ChatDev Local supports 4 different offline backends. Choose the one that best fits your needs:

### Option A: Ollama (Recommended for Beginners)

**Pros:** Easy to install, manages models automatically, good performance
**Cons:** Requires separate service installation

1. Install Ollama from https://ollama.ai/
2. Start Ollama service:
   ```bash
   ollama serve
   ```
3. Download a model:
   ```bash
   ollama pull llama2          # 7B model (~4GB)
   ollama pull codellama       # Code-focused model  
   ollama pull llama2:13b      # Larger model for better quality
   ```

### Option B: llama.cpp (Best Performance)

**Pros:** Excellent performance, low memory usage, many model options
**Cons:** Manual model download required

1. Install llama-cpp-python:
   ```bash
   pip install llama-cpp-python
   ```
2. Download GGUF models from Hugging Face:
   ```bash
   # Create models directory
   mkdir models
   cd models
   
   # Download a model (example: Llama 2 7B)
   wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
   ```

### Option C: HuggingFace Transformers

**Pros:** Many model options, integrated with Python ML ecosystem
**Cons:** Higher memory usage, slower inference

1. Install dependencies:
   ```bash
   pip install transformers torch accelerate
   ```
2. Models download automatically on first use

### Option D: LocalAI

**Pros:** OpenAI-compatible API, supports multiple model formats
**Cons:** More complex setup

1. Install LocalAI from https://localai.io/
2. Start LocalAI server on port 8080
3. Configure your models in LocalAI

## Step 3: Test Your Setup

Run the basic test to verify everything works:

```bash
python test_offline_basic.py
```

You should see all tests pass with ✓ marks.

## Step 4: Run ChatDev with Local Models

### Using Ollama:
```bash
python run.py \
  --model OLLAMA \
  --model_name llama2 \
  --task "Create a simple calculator app" \
  --name "Calculator"
```

### Using llama.cpp:
```bash
python run.py \
  --model LLAMA_CPP \
  --model_path ./models/llama-2-7b-chat.Q4_K_M.gguf \
  --task "Create a simple calculator app" \
  --name "Calculator"
```

### Using HuggingFace:
```bash
python run.py \
  --model HUGGINGFACE \
  --model_name microsoft/DialoGPT-medium \
  --device cpu \
  --task "Create a simple calculator app" \
  --name "Calculator"
```

### Using LocalAI:
```bash
python run.py \
  --model LOCALAI \
  --base_url http://localhost:8080 \
  --model_name gpt-3.5-turbo \
  --task "Create a simple calculator app" \
  --name "Calculator"
```

## Configuration Options

### Common Parameters:
- `--model`: Choose backend (OLLAMA, LLAMA_CPP, HUGGINGFACE, LOCALAI)
- `--task`: Describe the software you want to create
- `--name`: Name of your project
- `--temperature`: Creativity level (0.1-1.0, default: 0.7)
- `--max_tokens`: Maximum tokens per response (default: 512)

### Backend-Specific Parameters:
- `--model_name`: Model name (for Ollama, HuggingFace, LocalAI)
- `--model_path`: Path to model file (for llama.cpp)
- `--base_url`: API endpoint URL (for Ollama, LocalAI)
- `--device`: Computing device (cpu, cuda, mps) - for HuggingFace

## Recommended Model Configurations

### For Fast Development (Lower Quality):
```bash
# Ollama with Llama 2 7B
python run.py --model OLLAMA --model_name llama2 --temperature 0.8

# HuggingFace with DialoGPT
python run.py --model HUGGINGFACE --model_name microsoft/DialoGPT-medium --device cpu
```

### For High Quality (Slower):
```bash
# Ollama with Llama 2 13B
python run.py --model OLLAMA --model_name llama2:13b --temperature 0.7

# llama.cpp with larger model
python run.py --model LLAMA_CPP --model_path ./models/llama-2-13b-chat.Q4_K_M.gguf
```

### For Code Generation:
```bash
# Ollama with CodeLlama
python run.py --model OLLAMA --model_name codellama --temperature 0.3

# HuggingFace with CodeGen
python run.py --model HUGGINGFACE --model_name Salesforce/codegen-350M-mono --device cpu
```

## Tips for Better Results

1. **Use appropriate temperature:**
   - 0.1-0.3: For factual, structured tasks
   - 0.7-0.9: For creative tasks
   - 1.0+: For very creative tasks

2. **Adjust max_tokens based on task:**
   - Simple tasks: 256-512 tokens
   - Complex tasks: 1024-2048 tokens

3. **Model selection:**
   - 7B models: Fast, good for simple tasks
   - 13B+ models: Better quality, slower
   - Code-specific models: Better for programming tasks

## Troubleshooting

### "Model not found" errors:
- Verify model is downloaded and path is correct
- Check that Ollama service is running (for Ollama)
- Ensure LocalAI is configured properly (for LocalAI)

### Memory errors:
- Use smaller models (7B instead of 13B)
- Reduce max_tokens parameter
- Close other applications to free memory

### Slow performance:
- Use GPU acceleration if available (`--device cuda` or `--device mps`)
- Switch to llama.cpp backend for better performance
- Use quantized models (Q4_K_M format)

### Import errors:
- Install missing dependencies: `pip install transformers torch accelerate`
- Check Python version (3.9+ required)

## Advanced Configuration

### Multiple Models:
You can switch between different models for different phases by modifying the configuration files in `CompanyConfig/`.

### Custom Prompts:
Edit the role and phase configuration files to customize how agents behave.

### Offline Image Generation:
When running offline, placeholder images will be created instead of using OpenAI's DALL-E. Install Pillow for better placeholders:
```bash
pip install Pillow
```

## Performance Comparison

| Backend | Setup Difficulty | Performance | Memory Usage | Model Variety |
|---------|-----------------|-------------|--------------|---------------|
| Ollama | Easy | Good | Medium | Good |
| llama.cpp | Medium | Excellent | Low | Excellent |
| HuggingFace | Easy | Medium | High | Excellent |
| LocalAI | Hard | Good | Medium | Good |

## Conclusion

You now have ChatDev running completely offline! Your software development will be:
- ✅ Private and secure (no data sent to external APIs)
- ✅ Fast (no network latency)
- ✅ Cost-free (no API fees)
- ✅ Always available (no internet required)

Experiment with different models and settings to find what works best for your projects.

## Support

If you encounter issues:
1. Run `python test_offline_basic.py` to verify basic functionality
2. Check the troubleshooting section above
3. Review the model-specific documentation for your chosen backend
4. Check system resources (RAM, disk space)

Happy coding with ChatDev Local! 🚀