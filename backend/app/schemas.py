"""
CARE-AD+ API Schemas (Simplified Patient Input)
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr


# ============ Auth Schemas ============
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role: str = "clinician"

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str


# ============ Patient Schemas (Simplified) ============
class PatientCreate(BaseModel):
    patient_id: str
    name: str
    age: int

class PatientResponse(BaseModel):
    id: int
    patient_id: str
    name: str
    age: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============ Prediction Schemas ============
class PredictionCreate(BaseModel):
    patient_id: int

class PredictionResponse(BaseModel):
    id: int
    patient_id: int
    predicted_class: str
    confidence_score: float
    probabilities: Optional[Dict[str, float]]
    gradcam_path: Optional[str]
    heatmap_data: Optional[Dict[str, Any]]
    highlighted_regions: Optional[List[str]]
    created_at: datetime
    
    class Config:
        from_attributes = True

class PredictionDetail(PredictionResponse):
    patient: PatientResponse
    llm_explanation_technical: Optional[str]
    llm_explanation_patient: Optional[str]
    shap_values: Optional[Dict[str, float]]
    report_path: Optional[str]


# ============ XAI Schemas ============
class GradCAMResult(BaseModel):
    heatmap_path: str
    heatmap_data: List[List[float]]
    highlighted_regions: List[str]
    overlay_path: Optional[str]

class SHAPResult(BaseModel):
    feature_importance: Dict[str, float]
    plot_path: Optional[str]


# ============ Chat Schemas ============
class ChatMessage(BaseModel):
    message: str
    prediction_id: Optional[int] = None
    mode: str = "technical"  # technical or patient

class ChatResponse(BaseModel):
    response: str
    mode: str
    prediction_context: Optional[Dict[str, Any]] = None


# ============ Report Schemas ============
class ReportRequest(BaseModel):
    prediction_id: int
    include_gradcam: bool = True
    include_heatmap: bool = True
    include_explanation: bool = True

class ReportResponse(BaseModel):
    report_id: str
    report_path: str
    download_url: str
    generated_at: datetime


# ============ Admin/Stats Schemas ============
class DashboardStats(BaseModel):
    total_patients: int
    total_predictions: int
    predictions_today: int
    model_accuracy: float
    class_distribution: Dict[str, int]
    recent_predictions: List[Dict[str, Any]]

class ModelMetricsResponse(BaseModel):
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: List[List[int]]
    training_history: List[Dict[str, float]]
    dataset_stats: Dict[str, int]
    trained_at: datetime

class TrainingRequest(BaseModel):
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001

class TrainingStatus(BaseModel):
    status: str  # idle, training, completed, failed
    current_epoch: int
    total_epochs: int
    current_loss: float
    current_accuracy: float
    progress_percent: float
