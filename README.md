<div align="center">
  
  <img src="https://raw.githubusercontent.com/abhishekk-y/COMPUTER-AIDED-RECOGNITION-OF-ALZHEIMER-DISEASE/main/assets/logo.png" alt="CARE-AD+ Logo" width="200"/>
  
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
  Deep Learning • Explainable AI (XAI) • RAG-Enhanced LLM • Clinical Reports • Real-time Analytics
  
  [Features](#-key-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Demo](#-screenshots)

</div>

---

## 📋 Overview

**CARE-AD+** (Computer-Aided Recognition of Alzheimer's Disease Plus) is a comprehensive, multi-modal AI system designed to assist healthcare professionals in early detection and diagnosis of Alzheimer's disease. The system combines state-of-the-art deep learning with explainable AI techniques and RAG-enhanced LLM to provide transparent, clinically-relevant insights.

### 🎯 Mission

Early detection of Alzheimer's disease is crucial for patient care planning and potential intervention. CARE-AD+ provides clinicians with AI-powered analysis of brain MRI scans, backed by visual explanations, medical knowledge retrieval, and natural language interpretations.

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
      <p>Grad-CAM heatmaps for transparent, interpretable predictions</p>
    </td>
    <td width="33%" align="center">
      <h4>💬 RAG-Enhanced LLM</h4>
      <p>Ollama + Medical Knowledge Base for evidence-based explanations</p>
    </td>
  </tr>
  <tr>
    <td width="33%" align="center">
      <h4>📄 Clinical Reports</h4>
      <p>Professional PDF reports with visualizations and recommendations</p>
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
│   ├── 💬 Chat - RAG-enhanced AI Assistant
│   ├── 📄 Reports - PDF generation & download
│   └── ⚙️ Admin - System management
│
├── ⚡ Backend (FastAPI)
│   ├── 🔐 Authentication - JWT-based security
│   ├── 👤 Patients - Simplified CRUD (ID, Name, Age)
│   ├── 🧠 Predictions - ML inference pipeline
│   ├── 💬 Chat - LLM with RAG integration
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
│   └── 🔥 Grad-CAM - Visual explanations
│
├── 📚 RAG Pipeline
│   ├── 🏥 Medical Knowledge Base
│   ├── 🔎 Context Retrieval
│   └── 💡 Prompt Enhancement
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

| Requirement | Version | Download |
|-------------|---------|----------|
| **Python** | 3.10+ | [python.org](https://python.org/downloads/) |
| **Node.js** | 18+ | [nodejs.org](https://nodejs.org/) |
| **Ollama** | Latest | [ollama.ai](https://ollama.ai/download) |

### ⚡ One-Click Setup (Windows)

```bash
# Just double-click:
QUICK_START.bat
```

This automatically:
- ✅ Creates Python virtual environment
- ✅ Installs all dependencies
- ✅ Pulls Ollama phi3 model
- ✅ Starts backend & frontend servers

### 🔧 Manual Installation

```bash
# 1. Clone repository
git clone https://github.com/abhishekk-y/COMPUTER-AIDED-RECOGNITION-OF-ALZHEIMER-DISEASE.git
cd COMPUTER-AIDED-RECOGNITION-OF-ALZHEIMER-DISEASE

# 2. Setup backend
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Setup frontend
cd ../frontend
npm install

# 4. Setup Ollama
ollama pull phi3
ollama serve

# 5. Start servers
cd ..
start_app.bat
```

**Access**: http://localhost:3000

---

## 🎨 Tech Stack

<div align="center">

### Backend
| Technology | Purpose |
|-----------|---------|
| ![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) | REST API Framework |
| ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) | Deep Learning |
| ![SQLAlchemy](https://img.shields.io/badge/-SQLAlchemy-D71F00?style=flat-square) | Database ORM |
| ![Ollama](https://img.shields.io/badge/-Ollama-000000?style=flat-square) | Local LLM |

### Frontend
| Technology | Purpose |
|-----------|---------|
| ![React](https://img.shields.io/badge/-React-61DAFB?style=flat-square&logo=react&logoColor=black) | UI Framework |
| ![Vite](https://img.shields.io/badge/-Vite-646CFF?style=flat-square&logo=vite&logoColor=white) | Build Tool |
| ![Recharts](https://img.shields.io/badge/-Recharts-FF6384?style=flat-square) | Data Visualization |

</div>

---

## 📁 Project Structure

```
COMPUTER-AIDED-RECOGNITION-OF-ALZHEIMER-DISEASE/
│
├── 📂 backend/
│   ├── 📂 app/
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Configuration
│   │   ├── 📂 routers/          # API endpoints
│   │   ├── 📂 services/         # Business logic
│   │   │   ├── ml_service.py    # ML inference
│   │   │   ├── xai_service.py   # Grad-CAM
│   │   │   ├── llm_service.py   # LLM integration
│   │   │   ├── rag_service.py   # RAG pipeline
│   │   │   └── report_service.py # PDF generation
│   │   └── 📂 models/           # Database models
│   ├── 📂 ml/
│   │   ├── model.py             # CNN architecture
│   │   ├── dataset.py           # Data loading
│   │   ├── train.py             # Training pipeline
│   │   └── evaluate.py          # Evaluation
│   └── requirements.txt
│
├── 📂 frontend/
│   ├── 📂 src/
│   │   ├── 📂 pages/            # React pages
│   │   ├── 📂 components/       # Reusable components
│   │   ├── 📂 services/         # API client
│   │   └── 📂 styles/           # CSS
│   └── package.json
│
├── 📂 assets/                   # Project assets
├── QUICK_START.bat              # One-click setup
├── setup_ollama.bat             # Ollama setup
├── train_model.bat              # Model training
├── INSTALLATION.md              # Installation guide
├── OLLAMA_GUIDE.md              # LLM + RAG guide
└── README.md                    # This file
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [INSTALLATION.md](INSTALLATION.md) | Complete installation guide |
| [OLLAMA_GUIDE.md](OLLAMA_GUIDE.md) | LLM setup & RAG pipeline |

---

## 🧠 Model Training

```bash
# Quick training
train_model.bat

# Custom training
cd backend
python -m ml.train --dataset ../archive/combined_images --epochs 50
```

---

## 🤖 RAG Pipeline

The system includes a **Retrieval-Augmented Generation** pipeline that enhances LLM responses with medical knowledge:

- **Medical Knowledge Base**: CDR staging, biomarkers, treatments
- **Context Retrieval**: Automatic relevant knowledge extraction
- **Prompt Enhancement**: Evidence-based medical facts
- **Clinical Guidelines**: Recommendations per disease stage

See [OLLAMA_GUIDE.md](OLLAMA_GUIDE.md) for details.

---

## 🐳 Docker Deployment

```bash
docker-compose up -d
```

Services:
- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- Ollama: http://localhost:11434

---

## 👥 Default Credentials

| Role | Username | Password |
|------|----------|----------|
| Clinician | `clinician` | `password123` |
| Admin | `admin` | `admin123` |

> ⚠️ Change in production!

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | ~94% |
| **Precision** | ~92% |
| **Recall** | ~91% |
| **F1 Score** | ~92% |

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add AmazingFeature'`
4. Push: `git push origin feature/AmazingFeature`
5. Open Pull Request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file.

---

## ⚠️ Medical Disclaimer

> **IMPORTANT**: CARE-AD+ is a clinical decision **support** tool. It is NOT intended to replace professional medical judgment, diagnosis, or treatment. All predictions should be reviewed by qualified healthcare professionals.

---

## 🙏 Acknowledgments

- **Academic Guidance**: University project supervision
- **Open Source**: PyTorch, FastAPI, React communities
- **Medical Research**: Alzheimer's disease research community

---

<div align="center">

### 🌟 Star this repo if it helped you!

**Made with ❤️ for Better Healthcare**

[![GitHub stars](https://img.shields.io/github/stars/abhishekk-y/COMPUTER-AIDED-RECOGNITION-OF-ALZHEIMER-DISEASE?style=social)](https://github.com/abhishekk-y/COMPUTER-AIDED-RECOGNITION-OF-ALZHEIMER-DISEASE/stargazers)

</div>
