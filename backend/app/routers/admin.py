"""
CARE-AD+ Admin Router - Real-time Stats and Training
"""
import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database import get_db
from app.models.models import Patient, Prediction, ModelMetrics
from app.schemas import TrainingRequest, TrainingStatus, DashboardStats
from app.routers.auth import get_current_active_user, get_admin_user
from app.config import settings

router = APIRouter(prefix="/api/admin", tags=["admin"])

# Global training status
_training_status = {
    "status": "idle",
    "current_epoch": 0,
    "total_epochs": 0,
    "current_loss": 0,
    "current_accuracy": 0,
    "progress_percent": 0
}


@router.get("/dashboard")
async def get_dashboard_stats(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """Get comprehensive dashboard statistics"""
    # Count patients
    patients_result = await db.execute(select(func.count(Patient.id)))
    total_patients = patients_result.scalar() or 0
    
    # Count predictions
    predictions_result = await db.execute(select(func.count(Prediction.id)))
    total_predictions = predictions_result.scalar() or 0
    
    # Predictions today
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_result = await db.execute(
        select(func.count(Prediction.id))
        .where(Prediction.created_at >= today)
    )
    predictions_today = today_result.scalar() or 0
    
    # Class distribution
    class_result = await db.execute(
        select(Prediction.predicted_class, func.count(Prediction.id))
        .group_by(Prediction.predicted_class)
    )
    class_distribution = {row[0]: row[1] for row in class_result.fetchall()}
    
    # Dataset statistics
    dataset_images = 0
    dataset_path = settings.DATASET_PATH
    if os.path.exists(dataset_path):
        for class_dir in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_dir)
            if os.path.isdir(class_path):
                dataset_images += len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    return {
        "total_patients": total_patients,
        "total_predictions": total_predictions,
        "predictions_today": predictions_today,
        "class_distribution": class_distribution,
        "dataset_images": dataset_images
    }


@router.get("/model-metrics")
async def get_model_metrics(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """Get current model performance metrics"""
    # Try to load from file first
    metrics_path = os.path.join(settings.MODELS_DIR, "model_metrics.json")
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            return {
                "model_version": metrics.get("model_version", "1.0"),
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1_score", 0),
                "auc_roc": metrics.get("auc_roc", 0),
                "confusion_matrix": metrics.get("confusion_matrix", []),
                "training_history": metrics.get("training_history", []),
                "trained_at": metrics.get("trained_at", None)
            }
    
    # Try database
    result = await db.execute(
        select(ModelMetrics).order_by(ModelMetrics.trained_at.desc()).limit(1)
    )
    metrics = result.scalar_one_or_none()
    
    if metrics:
        return {
            "model_version": metrics.model_version,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "auc_roc": metrics.auc_roc,
            "confusion_matrix": metrics.confusion_matrix,
            "training_history": metrics.training_history,
            "trained_at": metrics.trained_at.isoformat() if metrics.trained_at else None
        }
    
    # Return empty metrics if none available
    return {
        "model_version": None,
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1_score": 0,
        "auc_roc": 0,
        "confusion_matrix": [],
        "training_history": [],
        "trained_at": None
    }


@router.get("/training-status")
async def get_training_status():
    """Get current training status"""
    # Check for training status file
    status_path = os.path.join(settings.MODELS_DIR, "training_status.json")
    
    if os.path.exists(status_path):
        try:
            with open(status_path, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return _training_status


@router.post("/train")
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_admin_user)
):
    """Start model training in background"""
    global _training_status
    
    if _training_status["status"] == "training":
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    _training_status = {
        "status": "training",
        "current_epoch": 0,
        "total_epochs": request.epochs,
        "current_loss": 0,
        "current_accuracy": 0,
        "progress_percent": 0
    }
    
    # Save status
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    status_path = os.path.join(settings.MODELS_DIR, "training_status.json")
    with open(status_path, 'w') as f:
        json.dump(_training_status, f)
    
    # Start training in background
    background_tasks.add_task(
        run_training,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate
    )
    
    return {"message": "Training started", "status": _training_status}


async def run_training(epochs: int, batch_size: int, learning_rate: float):
    """Run training in background"""
    global _training_status
    status_path = os.path.join(settings.MODELS_DIR, "training_status.json")
    
    try:
        from ml.train import train_model
        
        # This will update training_status.json during training
        result = train_model(
            dataset_path=settings.DATASET_PATH,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        _training_status = {
            "status": "completed",
            "current_epoch": epochs,
            "total_epochs": epochs,
            "current_loss": result.get("final_loss", 0),
            "current_accuracy": result.get("final_accuracy", 0),
            "progress_percent": 100
        }
        
    except Exception as e:
        _training_status = {
            "status": "failed",
            "error": str(e),
            "current_epoch": 0,
            "total_epochs": epochs,
            "current_loss": 0,
            "current_accuracy": 0,
            "progress_percent": 0
        }
    
    with open(status_path, 'w') as f:
        json.dump(_training_status, f)


@router.get("/dataset-stats")
async def get_dataset_stats(
    current_user = Depends(get_current_active_user)
):
    """Get detailed dataset statistics"""
    dataset_path = settings.DATASET_PATH
    
    if not os.path.exists(dataset_path):
        return {"error": "Dataset not found", "path": dataset_path}
    
    stats = {
        "path": dataset_path,
        "classes": {},
        "total_images": 0
    }
    
    for class_dir in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_dir)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            stats["classes"][class_dir] = len(images)
            stats["total_images"] += len(images)
    
    return stats


@router.post("/reload-model")
async def reload_model(
    current_user = Depends(get_admin_user)
):
    """Reload the ML model from disk"""
    try:
        from app.services.ml_service import get_ml_service
        ml_service = get_ml_service()
        ml_service.reload_model()
        
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


@router.get("/llm-status")
async def get_llm_status(
    current_user = Depends(get_current_active_user)
):
    """Check LLM (Ollama) availability"""
    from app.services.llm_service import get_llm_service
    llm_service = get_llm_service()
    return llm_service.check_availability()
