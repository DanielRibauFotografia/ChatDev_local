# ChatDev Local - Quick Reference

## 🚀 Quick Start (30 seconds to offline!)

### Option 1: Ollama (Easiest)
```bash
# Install Ollama from https://ollama.ai/
ollama serve
ollama pull llama2

# Install minimal dependencies
pip install tenacity tiktoken requests markdown

# Run ChatDev offline!
python run.py --model OLLAMA --model_name llama2 --task "Create a calculator" --name Calculator
```

### Option 2: HuggingFace (No external setup)
```bash
# Install dependencies
pip install tenacity tiktoken requests markdown transformers torch

# Run immediately (model downloads automatically)
python run.py --model HUGGINGFACE --model_name microsoft/DialoGPT-medium --device cpu --task "Create a todo app" --name TodoApp
```

## 📋 Command Reference

### Basic Syntax
```bash
python run.py --model BACKEND --model_name MODEL --task "DESCRIPTION" --name PROJECT_NAME
```

### Backend Options
| Backend | Setup | Model Parameter | Example |
|---------|-------|----------------|---------|
| `OLLAMA` | Install Ollama | `--model_name llama2` | Easy, recommended |
| `HUGGINGFACE` | pip install transformers torch | `--model_name microsoft/DialoGPT-medium` | No external setup |
| `LLAMA_CPP` | pip install llama-cpp-python | `--model_path ./models/model.gguf` | Best performance |
| `LOCALAI` | Install LocalAI server | `--base_url http://localhost:8080` | OpenAI compatible |

### Common Parameters
- `--temperature 0.7`: Creativity (0.1=factual, 1.0=creative)
- `--max_tokens 512`: Response length
- `--device cpu`: Force CPU (or `cuda`, `mps` for GPU)

## 📁 Project Structure After Running
```
WareHouse/
└── ProjectName_YourOrg_20231201120000/
    ├── main.py              # Your generated app
    ├── requirements.txt     # App dependencies  
    ├── manual.md           # User manual
    └── [other files]       # Supporting files
```

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Model not found" | Check model is downloaded (`ollama list`) |
| Memory error | Use smaller model or reduce `--max_tokens` |
| Import error | Install backend dependencies |
| Slow generation | Add `--device cuda` or `--device mps` for GPU |

## 📖 Need More Help?

- **Complete Setup**: See `OFFLINE_SETUP_GUIDE.md`
- **Test System**: Run `python demo_offline.py`
- **Verify Install**: Run `python test_offline_basic.py`

## 💡 Pro Tips

1. **Start with Ollama** - easiest to set up
2. **Use temperature 0.3** for coding tasks
3. **Try different models** for different tasks
4. **Use GPU acceleration** when available (`--device cuda` or `--device mps`)

---
🎉 **You're ready to create software completely offline!** 🎉