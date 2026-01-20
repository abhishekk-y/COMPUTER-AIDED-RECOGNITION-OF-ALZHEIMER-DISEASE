"""
CARE-AD+ Configuration Settings
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application configuration settings."""
    
    # Application
    APP_NAME: str = "CARE-AD+"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./care_ad.db"
    
    # Security
    SECRET_KEY: str = "care-ad-plus-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # Machine Learning
    MODEL_PATH: str = "../models/alzheimer_cnn.pth"
    IMAGE_SIZE: int = 224
    NUM_CLASSES: int = 4
    CLASS_NAMES: list = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    
    # LLM (Ollama) - Using Phi-3 mini for lightweight, quality responses
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "phi3"  # Lightweight: phi3, mistral, gemma2:2b
    
    # Paths
    UPLOAD_DIR: str = "../uploads"
    REPORTS_DIR: str = "../reports"
    DATASET_DIR: str = "../archive"
    
    # Institution Branding
    INSTITUTION_NAME: str = "CARE-AD+ Medical Center"
    INSTITUTION_LOGO: Optional[str] = None
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Create directories if they don't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.REPORTS_DIR, exist_ok=True)
