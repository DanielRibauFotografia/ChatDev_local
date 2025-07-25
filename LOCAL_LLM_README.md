# ChatDev Local LLM Interface

This implementation adds support for running ChatDev completely offline using local Language Models, eliminating the dependency on OpenAI's API while maintaining full compatibility with the existing multi-agent architecture.

## 🎯 Overview

The Local LLM Interface allows ChatDev to work with locally hosted models such as:
- **CodeLlama-7B/13B**: Specialized for Python code generation
- **WizardCoder-7B**: Optimized for programming tasks
- Support ready for additional models (Ollama, llama.cpp backends)

## 🚀 Quick Start

### 1. Test with Simulation Mode (No Downloads Required)

```bash
# Test the interface without downloading models
python test_local_llm.py
python test_simulation.py

# Run demo with different models
python demo_local_llm.py --model LOCAL_CODELLAMA_7B
python demo_local_llm.py --model LOCAL_WIZARDCODER_7B
```

### 2. Install Dependencies for Real Models

```bash
# Install local LLM dependencies
pip install torch transformers accelerate bitsandbytes

# For Apple Silicon users (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Run ChatDev with Local Models

```bash
# Use CodeLlama for software development
python run.py --model LOCAL_CODELLAMA_7B --task "Create a simple calculator app"

# Use WizardCoder for more complex programming tasks  
python run.py --model LOCAL_WIZARDCODER_7B --task "Build a REST API with authentication"

# Use larger model for better quality (requires more RAM)
python run.py --model LOCAL_CODELLAMA_13B --task "Develop a machine learning pipeline"
```

## 📋 Available Models

| Model | Size | Memory Usage* | Best For |
|-------|------|---------------|----------|
| `LOCAL_CODELLAMA_7B` | ~13GB | 8-12GB RAM | General Python coding |
| `LOCAL_CODELLAMA_13B` | ~25GB | 16-24GB RAM | Complex software projects |
| `LOCAL_WIZARDCODER_7B` | ~13GB | 8-12GB RAM | Algorithm implementation |

*With 4-bit quantization enabled

## ⚙️ Configuration

### Default Configuration

The system automatically detects and configures:
- **Device**: MPS (Apple Silicon) > CUDA > CPU
- **Quantization**: 4-bit for memory efficiency  
- **Memory Management**: Optimized for 16GB RAM systems

### Custom Configuration

Create a custom configuration by modifying the model_config in your code:

```python
model_config = {
    "max_tokens": 1024,           # Response length
    "temperature": 0.3,           # Creativity (0.0-1.0)
    "quantization": "4bit",       # Memory optimization
    "device": "auto",             # Device selection
    "simulation_mode": False      # Use real models
}
```

## 🏗️ Architecture

### Component Overview

```
ChatDev
├── camel/
│   ├── typing.py              # Model type definitions
│   ├── model_backend.py       # Model factory and backends  
│   ├── local_llm.py          # Local LLM interface
│   └── agents/chat_agent.py   # Agent implementations
├── run.py                     # Entry point with model selection
└── requirements.txt           # Dependencies
```

### Integration Points

1. **Model Types** (`camel/typing.py`): Defines new local model enums
2. **Model Factory** (`camel/model_backend.py`): Creates appropriate backends
3. **LLM Interface** (`camel/local_llm.py`): Handles model loading and inference
4. **Command Line** (`run.py`): Supports local model selection

## 🧪 Testing

### Test Suite

```bash
# Basic functionality tests
python test_local_llm.py

# Integration tests with mocking
python test_integration.py  

# Simulation mode tests
python test_simulation.py

# Full demo workflow
python demo_local_llm.py
```

### Simulation Mode

All tests run in simulation mode by default, which:
- ✅ Tests complete integration without downloading models
- ✅ Validates API compatibility and response formats
- ✅ Demonstrates multi-agent workflows
- ✅ Allows development without heavy dependencies

## 💾 Memory Requirements

### Recommended System Specs

| Configuration | RAM | Storage | Performance |
|---------------|-----|---------|-------------|
| **Minimum** | 16GB | 50GB | CodeLlama-7B with 4-bit quantization |
| **Recommended** | 32GB | 100GB | CodeLlama-13B with 8-bit quantization |
| **Optimal** | 64GB+ | 200GB+ | Multiple models, no quantization |

### Memory Optimization

The system automatically applies optimizations:
- **4-bit quantization** reduces memory usage by ~75%
- **Model offloading** to disk when not in use  
- **Gradient checkpointing** for memory-efficient inference
- **MPS acceleration** on Apple Silicon

## 🐛 Troubleshooting

### Common Issues

**Error: "Required dependencies not available"**
```bash
pip install torch transformers accelerate bitsandbytes
```

**Error: "CUDA out of memory"**
- Use 4-bit quantization: `quantization: "4bit"`
- Try smaller model: `LOCAL_CODELLAMA_7B` instead of 13B
- Close other applications to free memory

**Error: "Model loading timeout"**
- First run downloads models (can take 30+ minutes)
- Use `simulation_mode: True` for testing
- Check internet connection for model downloads

**Slow inference on Apple Silicon**
- Ensure MPS is enabled: `device: "mps"`
- Install PyTorch with MPS support
- Use quantized models for faster inference

### Performance Tips

1. **First Run**: Models download automatically but take time
2. **Disk Space**: Ensure 50GB+ free for model storage
3. **Internet**: Initial download requires stable connection
4. **Memory**: Close unnecessary applications before running

## 🔄 Migration from OpenAI

### Existing Code Compatibility

No changes needed! The local LLM interface provides the same API:

```python
# This works exactly the same way
response = model_backend.run(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a Python function."}
    ],
    max_tokens=1024,
    temperature=0.7
)
```

### Gradual Migration

1. **Test**: Start with simulation mode
2. **Pilot**: Try LOCAL_CODELLAMA_7B for small projects  
3. **Scale**: Use LOCAL_CODELLAMA_13B for production
4. **Fallback**: Keep OpenAI as backup option

## 📊 Performance Comparison

| Metric | OpenAI GPT-4 | CodeLlama-7B | CodeLlama-13B |
|--------|--------------|--------------|---------------|
| **Latency** | 2-5s | 5-15s | 10-30s |
| **Cost** | $0.03/1k tokens | Free | Free |
| **Privacy** | Cloud | Local | Local |
| **Offline** | ❌ | ✅ | ✅ |
| **Customization** | Limited | Full | Full |

## 🛣️ Future Enhancements

### Planned Features

- [ ] **Ollama Backend**: Easy model management
- [ ] **llama.cpp Integration**: Maximum performance
- [ ] **Apple MLX Support**: Native Apple Silicon acceleration  
- [ ] **Model Switching**: Dynamic model selection per agent
- [ ] **Fine-tuning Support**: Custom model training
- [ ] **Distributed Inference**: Multi-GPU support

### Contributing

1. **Test new models**: Add support for additional HuggingFace models
2. **Optimize performance**: Improve inference speed and memory usage
3. **Add backends**: Implement Ollama, llama.cpp integrations
4. **Enhance quantization**: Support for newer quantization methods

## 📄 License

This implementation follows the same Apache 2.0 license as the original ChatDev project.

## 🙏 Acknowledgments

- **CAMEL-AI** for the original ChatDev architecture
- **Meta AI** for CodeLlama models  
- **Microsoft** for WizardCoder models
- **HuggingFace** for the transformers library