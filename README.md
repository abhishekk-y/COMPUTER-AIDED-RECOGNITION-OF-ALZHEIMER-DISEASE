<div align="center">
  
  <img src="./assets/logo.png" alt="CARE-AD+ Logo" width="200"/>
  
  # 🧠 CARE-AD+ 
  
  ### **Computer-Aided Recognition of Alzheimer's Disease**
  
  [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![React](https://img.shields.io/badge/React-18.2+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
  
  [![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](http://makeapullrequest.com)
  [![Maintenance](https://img.shields.io/badge/Maintained-Yes-green.svg?style=for-the-badge)](https://github.com/abhishekk-y/COMPUTER-AIDED-RECOGNITION-OF-ALZHEIMER-DISEASE/graphs/commit-activity)

  ---
  
  **An advanced AI-powered clinical decision support system for early Alzheimer's disease detection**  
  Deep Learning • Explainable AI (XAI) • LLM Assistant • Clinical Reports • Real-time Analytics
  
  [Features](#-key-features) • [Quick Start](#-quick-start) • [Documentation](#-project-structure) • [Screenshots](#-screenshots)

</div>

---

## 📋 Overview

**CARE-AD+** (Computer-Aided Recognition of Alzheimer's Disease Plus) is a comprehensive, multi-modal AI system designed to assist healthcare professionals in early detection and diagnosis of Alzheimer's disease. The system combines state-of-the-art deep learning with explainable AI techniques to provide transparent, clinically-relevant insights.

### 🎯 Mission

Early detection of Alzheimer's disease is crucial for patient care planning and potential intervention. CARE-AD+ provides clinicians with AI-powered analysis of brain MRI scans, backed by visual explanations and natural language interpretations.

---

## ✨ Key Features

<table>
  <tr>
    <td width="33%" align="center">
      <h4>🧠 Deep Learning Analysis</h4>
      <p>EfficientNet/ResNet CNN for accurate MRI classification across 4 dementia stages</p>
    </td>
    <td width="33%" align="center">
      <h4>🔍 Explainable AI (XAI)</h4>
      <p>Grad-CAM heatmaps and SHAP analysis for transparent, interpretable predictions</p>
    </td>
    <td width="33%" align="center">
      <h4>💬 LLM Assistant</h4>
      <p>Ollama-powered AI chat with technical and patient-friendly explanation modes</p>
    </td>
  </tr>
  <tr>
    <td width="33%" align="center">
      <h4>📄 Clinical Reports</h4>
      <p>Professional PDF reports with visualizations, recommendations, and branding</p>
    </td>
    <td width="33%" align="center">
      <h4>📊 Real-time Dashboard</h4>
      <p>Live analytics, prediction tracking, and model performance monitoring</p>
    </td>
    <td width="33%" align="center">
      <h4>⚙️ Admin Control</h4>
      <p>Dataset management, model retraining, and system configuration</p>
    </td>
  </tr>
</table>

---

## 🏗️ System Architecture

```
📦 CARE-AD+ System
│
├── 🖥️ Frontend (React + Vite)
│   ├── 📊 Dashboard - Real-time statistics & charts
│   ├── 🔬 Prediction - MRI upload & analysis
│   ├── 📈 Results - Detailed findings with heatmaps
│   ├── 💬 Chat - AI Assistant interface
│   ├── 📄 Reports - PDF generation & download
│   └── ⚙️ Admin - System management
│
├── ⚡ Backend (FastAPI)
│   ├── 🔐 Authentication - JWT-based security
│   ├── 👤 Patients - CRUD operations
│   ├── 🧠 Predictions - ML inference pipeline
│   ├── 💬 Chat - LLM integration
│   ├── 📄 Reports - PDF generation
│   └── ⚙️ Admin - Training & metrics
│
├── 🤖 ML Pipeline (PyTorch)
│   ├── 📦 Dataset - Data loading & augmentation
│   ├── 🏗️ Model - EfficientNet/ResNet architecture
│   ├── 🏋️ Training - Complete training pipeline
│   └── 📊 Evaluation - Metrics & visualization
│
├── 🔍 XAI Services
│   ├── 🔥 Grad-CAM - Visual explanations
│   └── 📊 SHAP - Feature importance
│
└── 💬 LLM Service (Ollama)
    ├── 👨‍⚕️ Technical Mode - For clinicians
    └── 👤 Patient Mode - Simplified explanations
```

---

## 🏥 Classification Categories

| Class | Description | Color Code |
|-------|-------------|------------|
| 🟢 **NonDemented** | Cognitively normal, no signs of dementia | Green |
| 🟡 **VeryMildDemented** | Very mild cognitive impairment, early changes | Amber |
| 🟠 **MildDemented** | Mild dementia, consistent with early-stage AD | Orange |
| 🔴 **ModerateDemented** | Moderate dementia, significant impairment | Red |

---

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have the following installed:

| Requirement | Version | Purpose |
|-------------|---------|---------|
| ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white) | 3.10+ | Backend & ML |
| ![Node.js](https://img.shields.io/badge/Node.js-18+-339933?logo=node.js&logoColor=white) | 18+ | Frontend |
| ![Ollama](https://img.shields.io/badge/Ollama-Latest-000000?logo=ollama&logoColor=white) | Latest | LLM Service |

### ⚡ One-Click Setup (Windows)

```bash
# Just double-click QUICK_START.bat
# It will:
# ✅ Create Python virtual environment
# ✅ Install all dependencies
# ✅ Pull Ollama phi3 model
# ✅ Start backend & frontend servers
```

### 🔧 Manual Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/abhishekk-y/COMPUTER-AIDED-RECOGNITION-OF-ALZHEIMER-DISEASE.git
cd COMPUTER-AIDED-RECOGNITION-OF-ALZHEIMER-DISEASE

# 2️⃣ Setup Backend
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 3️⃣ Setup Frontend
cd ../frontend
npm install

# 4️⃣ Setup Ollama LLM
ollama pull phi3
ollama serve

# 5️⃣ Start Backend Server
cd ../backend
uvicorn app.main:app --reload --port 8000

# 6️⃣ Start Frontend Server (new terminal)
cd frontend
npm run dev

# 7️⃣ Open Browser
# Navigate to http://localhost:3000
```

### 🎯 Quick Commands

| Command | Description |
|---------|-------------|
| `QUICK_START.bat` | Complete setup & launch |
| `train_model.bat` | Train model on dataset |
| `start_app.bat` | Start servers only |

---

## 🎨 Tech Stack

<div align="center">

### Backend
| Technology | Purpose | Version |
|-----------|---------|---------|
| ![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) | REST API Framework | 0.104+ |
| ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) | Deep Learning | 2.0+ |
| ![SQLAlchemy](https://img.shields.io/badge/-SQLAlchemy-D71F00?style=flat-square&logo=sqlalchemy&logoColor=white) | Database ORM | 2.0+ |
| ![Ollama](https://img.shields.io/badge/-Ollama-000000?style=flat-square&logo=ollama&logoColor=white) | Local LLM | Latest |

### Frontend
| Technology | Purpose | Version |
|-----------|---------|---------|
| ![React](https://img.shields.io/badge/-React-61DAFB?style=flat-square&logo=react&logoColor=black) | UI Framework | 18.2+ |
| ![Vite](https://img.shields.io/badge/-Vite-646CFF?style=flat-square&logo=vite&logoColor=white) | Build Tool | 5.0+ |
| ![Recharts](https://img.shields.io/badge/-Recharts-FF6384?style=flat-square) | Data Visualization | 2.10+ |

### AI/ML
| Technology | Purpose |
|-----------|---------|
| ![EfficientNet](https://img.shields.io/badge/-EfficientNet-FF9900?style=flat-square) | Image Classification Backbone |
| ![Grad-CAM](https://img.shields.io/badge/-Grad--CAM-E34F26?style=flat-square) | Visual Explanations |
| ![SHAP](https://img.shields.io/badge/-SHAP-00ADD8?style=flat-square) | Feature Importance |

</div>

---

## 📁 Project Structure

```
COMPUTER-AIDED-RECOGNITION-OF-ALZHEIMER-DISEASE/
│
├── 📂 assets/
│   └── logo.png                    # Project logo
│
├── 📂 backend/
│   ├── 📂 app/
│   │   ├── 📄 main.py              # FastAPI application entry
│   │   ├── 📄 config.py            # Configuration settings
│   │   ├── 📄 database.py          # SQLAlchemy setup
│   │   ├── 📄 schemas.py           # Pydantic models
│   │   │
│   │   ├── 📂 models/
│   │   │   └── models.py           # Database ORM models
│   │   │
│   │   ├── 📂 routers/
│   │   │   ├── auth.py             # Authentication endpoints
│   │   │   ├── patients.py         # Patient CRUD
│   │   │   ├── predictions.py      # ML inference API
│   │   │   ├── chat.py             # LLM chat interface
│   │   │   ├── reports.py          # PDF generation
│   │   │   └── admin.py            # Admin operations
│   │   │
│   │   └── 📂 services/
│   │       ├── ml_service.py       # ML model loading & inference
│   │       ├── xai_service.py      # Grad-CAM & SHAP
│   │       ├── llm_service.py      # Ollama integration
│   │       └── report_service.py   # PDF report generation
│   │
│   ├── 📂 ml/
│   │   ├── 📄 model.py             # CNN architecture definition
│   │   ├── 📄 dataset.py           # Data loading & augmentation
│   │   ├── 📄 train.py             # Training pipeline
│   │   └── 📄 evaluate.py          # Evaluation & metrics
│   │
│   ├── 📄 requirements.txt         # Python dependencies
│   └── 📄 Dockerfile               # Container configuration
│
├── 📂 frontend/
│   ├── 📂 src/
│   │   ├── 📄 App.jsx              # Main React component
│   │   ├── 📄 main.jsx             # Entry point
│   │   │
│   │   ├── 📂 components/
│   │   │   └── Layout.jsx          # App layout with sidebar
│   │   │
│   │   ├── 📂 pages/
│   │   │   ├── Dashboard.jsx       # Statistics & charts
│   │   │   ├── Prediction.jsx      # MRI upload & analysis
│   │   │   ├── Results.jsx         # Detailed findings
│   │   │   ├── Chat.jsx            # AI assistant
│   │   │   ├── Reports.jsx         # PDF management
│   │   │   ├── Admin.jsx           # System admin
│   │   │   └── Login.jsx           # Authentication
│   │   │
│   │   ├── 📂 services/
│   │   │   └── api.js              # Axios API client
│   │   │
│   │   └── 📂 styles/
│   │       └── index.css           # Global styles
│   │
│   ├── 📄 package.json             # Node dependencies
│   ├── 📄 vite.config.js           # Vite configuration
│   └── 📄 index.html               # HTML template
│
├── 📂 archive/                     # MRI Dataset (not in repo)
│   ├── MildDemented/
│   ├── ModerateDemented/
│   ├── NonDemented/
│   └── VeryMildDemented/
│
├── 📂 models/                      # Trained model weights
├── 📂 uploads/                     # Uploaded MRI images
├── 📂 reports/                     # Generated PDF reports
│
├── 📄 QUICK_START.bat              # One-click setup script
├── 📄 train_model.bat              # Model training script
├── 📄 start_app.bat                # Server startup script
├── 📄 docker-compose.yml           # Docker orchestration
├── 📄 LICENSE                      # MIT License
└── 📄 README.md                    # This file
```

---

## 🔐 Security Features

<table>
  <tr>
    <td width="50%">
      <h4>✅ JWT Authentication</h4>
      <p>Secure token-based user authentication</p>
    </td>
    <td width="50%">
      <h4>✅ Role-Based Access</h4>
      <p>Clinician and Admin role separation</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h4>✅ Password Hashing</h4>
      <p>BCrypt secure password storage</p>
    </td>
    <td width="50%">
      <h4>✅ Protected Routes</h4>
      <p>API endpoint authorization</p>
    </td>
  </tr>
</table>

---

## 👥 Default Login Credentials

### 🧪 For Testing

| Role | Username | Password | Access |
|------|----------|----------|--------|
| **👨‍⚕️ Clinician** | `clinician` | `password123` | Standard |
| **⚙️ Admin** | `admin` | `admin123` | Full |

> ⚠️ **Important**: Change these credentials before production deployment!

---

## 📦 API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | User login |
| GET | `/api/auth/me` | Get current user |

### Patients
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/patients/` | Create patient |
| GET | `/api/patients/` | List all patients |
| GET | `/api/patients/{id}` | Get patient details |

### Predictions
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predictions/` | Run prediction |
| GET | `/api/predictions/` | List predictions |
| GET | `/api/predictions/{id}` | Get prediction details |
| GET | `/api/predictions/{id}/gradcam` | Get Grad-CAM |

### Reports
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/reports/generate` | Generate PDF |
| GET | `/api/reports/download/{id}` | Download PDF |

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat/` | Send message to LLM |

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Services:
# - Backend:  http://localhost:8000
# - Frontend: http://localhost:3000
# - Ollama:   http://localhost:11434
```

### Docker Compose Services

```yaml
services:
  backend:   # FastAPI server
  frontend:  # React app
  ollama:    # LLM service
```

---

## 🧠 Model Training

```bash
# Train on your dataset
train_model.bat

# Or manually:
cd backend
python -m ml.train --dataset ../archive --epochs 50 --batch-size 32
```

### Training Features
- ✅ Data augmentation
- ✅ Class weight balancing
- ✅ Early stopping
- ✅ Learning rate scheduling
- ✅ Model checkpointing
- ✅ Live progress tracking

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | ~94% |
| **Precision** | ~92% |
| **Recall** | ~91% |
| **F1 Score** | ~92% |

*Metrics may vary based on dataset and training configuration*

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/AmazingFeature`
3. **Commit** changes: `git commit -m 'Add AmazingFeature'`
4. **Push** branch: `git push origin feature/AmazingFeature`
5. **Open** Pull Request

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

<table>
  <tr>
    <td align="center" width="33%">
      <h4>🎓 Academic Guidance</h4>
      <p>University project supervision</p>
    </td>
    <td align="center" width="33%">
      <h4>🌐 Open Source Community</h4>
      <p>PyTorch, FastAPI, React teams</p>
    </td>
    <td align="center" width="33%">
      <h4>🏥 Medical Research</h4>
      <p>Alzheimer's disease research community</p>
    </td>
  </tr>
</table>

---

## ⚠️ Medical Disclaimer

> **IMPORTANT**: CARE-AD+ is a clinical decision **support** tool. It is NOT intended to replace professional medical judgment, diagnosis, or treatment. All predictions should be reviewed by qualified healthcare professionals in conjunction with clinical examination and patient history.

---

<div align="center">

### 🌟 If this project helped you, please consider giving it a ⭐!

---

**Made with ❤️ for Better Healthcare**

[![Follow on GitHub](https://img.shields.io/github/followers/abhishekk-y?label=Follow&style=social)](https://github.com/abhishekk-y)

</div>
