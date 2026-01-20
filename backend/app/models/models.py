"""
CARE-AD+ Database Models (Simplified)
Patient requires only: ID, Name, Age
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    """User model for authentication"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(String(20), default="clinician")  # clinician, admin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    predictions = relationship("Prediction", back_populates="user")


class Patient(Base):
    """Simplified Patient model - only essential fields"""
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    predictions = relationship("Prediction", back_populates="patient")


class Prediction(Base):
    """Prediction result with XAI data"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Image
    image_path = Column(String(500))
    
    # Prediction results
    predicted_class = Column(String(50), nullable=False)
    confidence_score = Column(Float, nullable=False)
    probabilities = Column(JSON)  # {"NonDemented": 0.8, ...}
    
    # XAI outputs
    gradcam_path = Column(String(500))
    heatmap_data = Column(JSON)  # Raw heatmap values
    shap_values = Column(JSON)
    highlighted_regions = Column(JSON)  # List of affected brain regions
    
    # LLM explanations
    llm_explanation_technical = Column(Text)
    llm_explanation_patient = Column(Text)
    
    # Report
    report_path = Column(String(500))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="predictions")
    user = relationship("User", back_populates="predictions")
    chat_history = relationship("ChatHistory", back_populates="prediction")


class ModelMetrics(Base):
    """Model performance tracking"""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String(50))
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    confusion_matrix = Column(JSON)
    class_metrics = Column(JSON)  # Per-class precision/recall
    training_history = Column(JSON)  # Epoch-by-epoch metrics
    dataset_stats = Column(JSON)  # Class distribution
    trained_at = Column(DateTime, default=datetime.utcnow)


class ChatHistory(Base):
    """Chat conversation history"""
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    role = Column(String(20))  # user, assistant
    content = Column(Text, nullable=False)
    mode = Column(String(20), default="technical")  # technical, patient
    created_at = Column(DateTime, default=datetime.utcnow)
    
    prediction = relationship("Prediction", back_populates="chat_history")
