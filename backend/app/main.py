"""
CARE-AD+ Main FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

from app.config import settings
from app.database import init_db
from app.routers import auth, patients, predictions, chat, reports, admin


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print(f"🧠 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    await init_db()
    print("✅ Database initialized")
    
    # Ensure directories exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.REPORTS_DIR, exist_ok=True)
    os.makedirs("../models", exist_ok=True)
    
    yield
    
    # Shutdown
    print("👋 Shutting down CARE-AD+")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="""
    ## CARE-AD+: Computer-Aided Recognition of Alzheimer's Disease Plus
    
    A multi-modal explainable deep learning system for early Alzheimer's disease detection 
    with LLM-powered explanations and clinical reporting.
    
    ### Features
    - 🧠 **MRI Analysis**: CNN-based classification
    - 📊 **Explainable AI**: Grad-CAM and SHAP visualizations
    - 💬 **LLM Assistant**: Context-aware explanations
    - 📄 **Clinical Reports**: Professional PDF generation
    - ⚙️ **Admin Dashboard**: Model management
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving images and reports
if os.path.exists(settings.UPLOAD_DIR):
    app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
if os.path.exists(settings.REPORTS_DIR):
    app.mount("/reports", StaticFiles(directory=settings.REPORTS_DIR), name="reports")

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(patients.router, prefix="/api/patients", tags=["Patients"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat Assistant"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "healthy",
        "message": "Welcome to CARE-AD+ API"
    }


@app.get("/api/health", tags=["Health"])
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "database": "connected",
        "ml_model": "loaded" if os.path.exists(settings.MODEL_PATH) else "not_loaded",
        "llm_status": "available"
    }
