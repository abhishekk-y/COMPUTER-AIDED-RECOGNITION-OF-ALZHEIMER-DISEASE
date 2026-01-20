# 🚀 CARE-AD+ Installation Guide

## Prerequisites

Before you begin, ensure you have the following installed:

### Required Software

| Software | Version | Download Link |
|----------|---------|---------------|
| **Python** | 3.10+ | [python.org](https://www.python.org/downloads/) |
| **Node.js** | 18+ | [nodejs.org](https://nodejs.org/) |
| **Git** | Latest | [git-scm.com](https://git-scm.com/) |
| **Ollama** | Latest | [ollama.ai](https://ollama.ai/download) |

---

## 📦 Quick Installation (Windows)

### Option 1: One-Click Setup (Recommended)

```bash
# Just double-click this file:
QUICK_START.bat
```

This will automatically:
- ✅ Create Python virtual environment
- ✅ Install all backend dependencies
- ✅ Install frontend dependencies
- ✅ Pull Ollama phi3 model
- ✅ Start backend & frontend servers

### Option 2: Manual Setup

#### Step 1: Clone Repository

```bash
git clone https://github.com/abhishekk-y/COMPUTER-AIDED-RECOGNITION-OF-ALZHEIMER-DISEASE.git
cd COMPUTER-AIDED-RECOGNITION-OF-ALZHEIMER-DISEASE
```

#### Step 2: Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from app.database import init_db; import asyncio; asyncio.run(init_db())"
```

#### Step 3: Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install
```

#### Step 4: Ollama LLM Setup

```bash
# Run the setup script
..\setup_ollama.bat

# OR manually:
# 1. Download Ollama from https://ollama.ai/download
# 2. Install Ollama
# 3. Open terminal and run:
ollama pull phi3
ollama serve
```

---

## 🎯 Running the Application

### Start All Services

```bash
# From project root:
start_app.bat
```

This starts:
- **Backend API**: http://localhost:8000
- **Frontend UI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

### Start Services Individually

**Backend Only:**
```bash
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

**Frontend Only:**
```bash
cd frontend
npm run dev
```

**Ollama LLM:**
```bash
ollama serve
```

---

## 🧠 Training the Model

### Quick Training

```bash
# From project root:
train_model.bat
```

### Custom Training

```bash
cd backend
venv\Scripts\activate

python -m ml.train --dataset ../archive/combined_images --epochs 50 --batch-size 32 --lr 0.001
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | `../archive/combined_images` | Path to dataset |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--save-dir` | `../models` | Model save directory |

---

## 📊 Dataset Setup

### Expected Structure

```
archive/
└── combined_images/
    ├── MildDemented/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── ModerateDemented/
    ├── NonDemented/
    └── VeryMildDemented/
```

### Dataset Requirements

- **Format**: JPG, JPEG, or PNG
- **Size**: Any (will be resized to 224x224)
- **Classes**: 4 (MildDemented, ModerateDemented, NonDemented, VeryMildDemented)
- **Recommended**: At least 100 images per class

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# API Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Database
DATABASE_URL=sqlite+aiosqlite:///./care_ad.db

# Security
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Ollama LLM
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=phi3

# Paths (relative to backend/)
UPLOAD_DIR=../uploads
REPORTS_DIR=../reports
DATASET_PATH=../archive/combined_images
MODEL_PATH=models/alzheimer_cnn.pth
```

---

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services

- **backend**: FastAPI server (port 8000)
- **frontend**: React app (port 3000)
- **ollama**: LLM service (port 11434)

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Ollama Not Found

**Problem**: `ollama: command not found`

**Solution**:
```bash
# Download and install Ollama from:
https://ollama.ai/download

# After installation, restart terminal and run:
ollama pull phi3
ollama serve
```

#### 2. Model Not Loading

**Problem**: "Model not loaded" error

**Solution**:
```bash
# Train a model first:
train_model.bat

# Or download a pre-trained model and place in:
backend/models/alzheimer_cnn.pth
```

#### 3. Port Already in Use

**Problem**: Port 8000 or 3000 already in use

**Solution**:
```bash
# Find and kill process using the port:
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or change port in config
```

#### 4. CUDA/GPU Issues

**Problem**: CUDA errors or GPU not detected

**Solution**:
```bash
# The system works on CPU too
# To force CPU mode, set in config.py:
device = "cpu"
```

#### 5. Frontend Can't Connect to Backend

**Problem**: API calls failing

**Solution**:
```bash
# Check backend is running:
curl http://localhost:8000/api/auth/me

# Check Vite proxy in frontend/vite.config.js:
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true
  }
}
```

---

## 📚 API Documentation

Once the backend is running, visit:

**Interactive API Docs**: http://localhost:8000/docs

**Alternative Docs**: http://localhost:8000/redoc

---

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest tests/
```

### Frontend Tests

```bash
cd frontend
npm test
```

---

## 📝 Default Credentials

### For Testing

| Role | Username | Password |
|------|----------|----------|
| Clinician | `clinician` | `password123` |
| Admin | `admin` | `admin123` |

> ⚠️ **Change these in production!**

---

## 🔐 Security Checklist

Before deploying to production:

- [ ] Change `SECRET_KEY` in config
- [ ] Update default passwords
- [ ] Enable HTTPS
- [ ] Configure CORS properly
- [ ] Set up proper database (PostgreSQL)
- [ ] Enable rate limiting
- [ ] Set up monitoring and logging
- [ ] Regular security audits

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/abhishekk-y/COMPUTER-AIDED-RECOGNITION-OF-ALZHEIMER-DISEASE/issues)
- **Discussions**: [GitHub Discussions](https://github.com/abhishekk-y/COMPUTER-AIDED-RECOGNITION-OF-ALZHEIMER-DISEASE/discussions)
- **Email**: support@care-ad.com

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ for Better Healthcare**
