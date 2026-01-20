# 🤖 Ollama LLM Integration Guide

## Overview

CARE-AD+ uses **Ollama** to run a local Large Language Model (LLM) for generating AI-powered explanations of Alzheimer's disease predictions. This guide covers installation, configuration, and advanced usage.

---

## 📦 What is Ollama?

Ollama is a tool that lets you run large language models locally on your machine. Benefits:

- ✅ **Privacy**: All data stays on your machine
- ✅ **No API costs**: Free to use
- ✅ **Offline**: Works without internet
- ✅ **Fast**: Local inference
- ✅ **Customizable**: Choose your model

---

## 🚀 Quick Setup

### Step 1: Install Ollama

**Windows:**
```bash
# Download installer from:
https://ollama.ai/download

# Run the installer
# Ollama will be added to your PATH
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS:**
```bash
# Download from:
https://ollama.ai/download
```

### Step 2: Pull the Model

```bash
# Pull the phi3 model (~2.3GB)
ollama pull phi3

# Verify installation
ollama list
```

### Step 3: Start Ollama Server

```bash
# Start the server
ollama serve

# Server runs on http://localhost:11434
```

### Step 4: Test the Integration

```bash
# Test with curl
curl http://localhost:11434/api/generate -d '{
  "model": "phi3",
  "prompt": "Explain Alzheimer'\''s disease in simple terms"
}'
```

---

## 🎯 Model Selection

### Recommended Models

| Model | Size | RAM Required | Use Case |
|-------|------|--------------|----------|
| **phi3** | 2.3GB | 4GB | ✅ **Recommended** - Fast, accurate medical text |
| **mistral** | 4.1GB | 8GB | More detailed explanations |
| **llama3** | 4.7GB | 8GB | Best quality, slower |
| **gemma2:2b** | 1.6GB | 2GB | Lightweight, basic responses |

### Changing Models

Edit `backend/app/config.py`:

```python
class Settings(BaseSettings):
    OLLAMA_MODEL: str = "phi3"  # Change to: mistral, llama3, etc.
```

Then pull the new model:

```bash
ollama pull mistral
```

---

## 🔧 Configuration

### Backend Configuration

File: `backend/app/config.py`

```python
class Settings(BaseSettings):
    # Ollama Settings
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "phi3"
    OLLAMA_TIMEOUT: int = 60  # seconds
```

### LLM Service

File: `backend/app/services/llm_service.py`

The service handles:
- ✅ Connection checking
- ✅ Prompt engineering
- ✅ Fallback responses
- ✅ Async requests

---

## 💬 How It Works

### 1. Prediction Context

When a prediction is made, the system sends context to the LLM:

```python
context = {
    "predicted_class": "MildDemented",
    "confidence_score": 0.87,
    "patient_name": "John Doe",
    "patient_age": 72
}
```

### 2. Prompt Engineering

The system builds specialized prompts:

**Technical Mode (for clinicians):**
```
You are a clinical decision support AI for neurologists analyzing 
Alzheimer's disease predictions. Provide technical, evidence-based 
explanations referencing relevant biomarkers and staging criteria.

Context:
- Prediction: MildDemented
- Confidence: 87.0%
- Patient: John Doe
- Age: 72

User: Explain this prediction result.
```

**Patient Mode (for patients/families):**
```
You are a caring medical assistant explaining Alzheimer's disease 
test results to patients and families. Use simple, compassionate 
language. Avoid medical jargon. Be reassuring but honest.

Context:
- Prediction: MildDemented
- Confidence: 87.0%

User: What do these results mean?
```

### 3. Response Generation

The LLM generates context-aware responses:

```json
{
  "response": "Based on the brain scan analysis, the findings suggest mild cognitive impairment consistent with early-stage Alzheimer's disease...",
  "mode": "technical",
  "prediction_context": {...}
}
```

---

## 🎨 Customization

### Custom Prompts

Edit `backend/app/services/llm_service.py`:

```python
def _build_prompt(self, message, context, mode):
    if mode == "patient":
        system = """Your custom patient-friendly prompt here"""
    else:
        system = """Your custom technical prompt here"""
    
    # Add your custom logic
    return prompt
```

### Temperature & Parameters

Adjust generation parameters:

```python
response = httpx.post(
    f"{self.host}/api/generate",
    json={
        "model": self.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,  # 0.0-1.0 (lower = more focused)
            "top_p": 0.9,        # Nucleus sampling
            "num_predict": 500,  # Max tokens
            "stop": ["\n\n"]     # Stop sequences
        }
    }
)
```

---

## 🔍 Advanced Features

### RAG (Retrieval-Augmented Generation)

To add medical knowledge base:

```python
# 1. Create knowledge base
knowledge_base = {
    "alzheimers_stages": {
        "mild": "CDR 1: Mild memory loss, difficulty with complex tasks...",
        "moderate": "CDR 2: Significant memory impairment...",
    },
    "biomarkers": {
        "hippocampal_atrophy": "Indicator of neurodegeneration...",
    }
}

# 2. Enhance prompt with relevant context
def _build_prompt_with_rag(self, message, context, mode):
    # Retrieve relevant knowledge
    relevant_docs = self._retrieve_relevant_docs(context)
    
    prompt = f"{system}\n\nKnowledge Base:\n{relevant_docs}\n\n"
    prompt += f"Context: {context}\n\nUser: {message}"
    
    return prompt
```

### Streaming Responses

For real-time chat:

```python
async def generate_stream(self, message, context, mode):
    prompt = self._build_prompt(message, context, mode)
    
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": True}
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    yield data.get("response", "")
```

### Multi-Model Ensemble

Use multiple models for better results:

```python
class MultiModelLLMService:
    def __init__(self):
        self.models = ["phi3", "mistral", "gemma2:2b"]
    
    async def generate_ensemble(self, message, context):
        responses = []
        for model in self.models:
            response = await self._generate_with_model(model, message, context)
            responses.append(response)
        
        # Combine or vote on responses
        return self._combine_responses(responses)
```

---

## 📊 Monitoring

### Check Ollama Status

```bash
# List running models
ollama ps

# View model info
ollama show phi3

# Check server health
curl http://localhost:11434/api/tags
```

### Performance Metrics

Monitor in your application:

```python
import time

start = time.time()
response = await llm_service.generate_response(message, context)
duration = time.time() - start

print(f"LLM response time: {duration:.2f}s")
```

---

## 🐛 Troubleshooting

### Ollama Not Starting

```bash
# Check if port is in use
netstat -ano | findstr :11434

# Kill existing process
taskkill /PID <PID> /F

# Restart Ollama
ollama serve
```

### Model Not Found

```bash
# List available models
ollama list

# Pull missing model
ollama pull phi3
```

### Slow Responses

```bash
# Use smaller model
ollama pull gemma2:2b

# Or reduce max tokens
"num_predict": 200  # Instead of 500
```

### Out of Memory

```bash
# Use quantized model
ollama pull phi3:q4_0  # 4-bit quantization

# Or increase system RAM
# Or use cloud deployment
```

---

## 🚀 Production Deployment

### Docker Setup

```dockerfile
# Add to docker-compose.yml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    command: serve

volumes:
  ollama_data:
```

### Cloud Deployment

For production, consider:

1. **Dedicated GPU server** for faster inference
2. **Load balancing** for multiple requests
3. **Caching** frequently asked questions
4. **Fallback** to cloud LLM APIs (OpenAI, Anthropic)

---

## 📚 Resources

- **Ollama Docs**: https://github.com/ollama/ollama
- **Model Library**: https://ollama.ai/library
- **Phi-3 Paper**: https://arxiv.org/abs/2404.14219
- **Prompt Engineering**: https://www.promptingguide.ai/

---

## 🎯 Best Practices

1. **Always have fallback responses** when Ollama is unavailable
2. **Cache common queries** to reduce load
3. **Monitor response times** and adjust model if needed
4. **Use appropriate temperature** (0.7 for medical, 0.3 for factual)
5. **Validate responses** for medical accuracy
6. **Log all interactions** for quality improvement

---

**Made with ❤️ for Better Healthcare**
