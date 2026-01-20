"""
CARE-AD+ Predictions Router - Real-time with XAI
"""
import os
import uuid
import json
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.models import Patient, Prediction
from app.schemas import PredictionResponse, PredictionDetail
from app.routers.auth import get_current_active_user
from app.config import settings
from app.services.ml_service import get_ml_service
from app.services.xai_service import get_xai_service
from app.services.llm_service import get_llm_service

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


@router.post("/", response_model=PredictionDetail)
async def create_prediction(
    patient_id: int = Form(...),
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """
    Run prediction on MRI image with full XAI analysis.
    Returns prediction, Grad-CAM heatmap, and LLM explanation.
    """
    # Verify patient exists
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Save uploaded image
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_ext = os.path.splitext(image.filename)[1] or ".jpg"
    file_id = uuid.uuid4().hex[:8]
    image_filename = f"{patient.patient_id}_{file_id}{file_ext}"
    image_path = os.path.join(settings.UPLOAD_DIR, image_filename)
    
    content = await image.read()
    with open(image_path, "wb") as f:
        f.write(content)
    
    # Get services
    ml_service = get_ml_service()
    xai_service = get_xai_service()
    llm_service = get_llm_service()
    
    try:
        # Run ML prediction
        prediction_result = await ml_service.predict(image_path)
        
        # Generate Grad-CAM
        gradcam_result = await xai_service.generate_gradcam(
            image_path,
            prediction_result["predicted_class"]
        )
        
        # Generate LLM explanations
        context = {
            "predicted_class": prediction_result["predicted_class"],
            "confidence_score": prediction_result["confidence_score"],
            "probabilities": prediction_result["probabilities"],
            "patient_name": patient.name,
            "patient_age": patient.age
        }
        
        technical_explanation = await llm_service.generate_response(
            "Provide a clinical explanation of this prediction result.",
            context,
            mode="technical"
        )
        
        patient_explanation = await llm_service.generate_response(
            "Explain this result in simple terms for the patient.",
            context,
            mode="patient"
        )
        
        # Create prediction record
        db_prediction = Prediction(
            patient_id=patient.id,
            user_id=current_user.id if hasattr(current_user, 'id') else None,
            image_path=image_path,
            predicted_class=prediction_result["predicted_class"],
            confidence_score=prediction_result["confidence_score"],
            probabilities=prediction_result["probabilities"],
            gradcam_path=gradcam_result.get("heatmap_path"),
            heatmap_data=gradcam_result.get("heatmap_data"),
            highlighted_regions=gradcam_result.get("highlighted_regions"),
            llm_explanation_technical=technical_explanation,
            llm_explanation_patient=patient_explanation
        )
        
        db.add(db_prediction)
        await db.commit()
        await db.refresh(db_prediction)
        
        return PredictionDetail(
            id=db_prediction.id,
            patient_id=db_prediction.patient_id,
            predicted_class=db_prediction.predicted_class,
            confidence_score=db_prediction.confidence_score,
            probabilities=db_prediction.probabilities,
            gradcam_path=db_prediction.gradcam_path,
            heatmap_data=db_prediction.heatmap_data,
            highlighted_regions=db_prediction.highlighted_regions,
            created_at=db_prediction.created_at,
            patient=patient,
            llm_explanation_technical=db_prediction.llm_explanation_technical,
            llm_explanation_patient=db_prediction.llm_explanation_patient,
            shap_values=None,
            report_path=db_prediction.report_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/", response_model=list[PredictionResponse])
async def list_predictions(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """Get all predictions with pagination"""
    result = await db.execute(
        select(Prediction)
        .options(selectinload(Prediction.patient))
        .order_by(Prediction.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


@router.get("/{prediction_id}", response_model=PredictionDetail)
async def get_prediction(
    prediction_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """Get detailed prediction by ID"""
    result = await db.execute(
        select(Prediction)
        .options(selectinload(Prediction.patient))
        .where(Prediction.id == prediction_id)
    )
    prediction = result.scalar_one_or_none()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return prediction


@router.get("/{prediction_id}/gradcam")
async def get_gradcam(
    prediction_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get Grad-CAM visualization for a prediction"""
    result = await db.execute(
        select(Prediction).where(Prediction.id == prediction_id)
    )
    prediction = result.scalar_one_or_none()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return {
        "prediction_id": prediction_id,
        "gradcam_path": prediction.gradcam_path,
        "heatmap_data": prediction.heatmap_data,
        "highlighted_regions": prediction.highlighted_regions
    }


@router.get("/stats/summary")
async def get_prediction_stats(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """Get prediction statistics for dashboard"""
    from sqlalchemy import func
    
    # Total counts
    total_result = await db.execute(select(func.count(Prediction.id)))
    total = total_result.scalar() or 0
    
    # Class distribution
    class_result = await db.execute(
        select(Prediction.predicted_class, func.count(Prediction.id))
        .group_by(Prediction.predicted_class)
    )
    class_distribution = {row[0]: row[1] for row in class_result.fetchall()}
    
    # Recent predictions
    recent_result = await db.execute(
        select(Prediction)
        .options(selectinload(Prediction.patient))
        .order_by(Prediction.created_at.desc())
        .limit(10)
    )
    recent = recent_result.scalars().all()
    
    return {
        "total_predictions": total,
        "class_distribution": class_distribution,
        "recent_predictions": [
            {
                "id": p.id,
                "patient_id": p.patient.patient_id,
                "patient_name": p.patient.name,
                "predicted_class": p.predicted_class,
                "confidence": p.confidence_score,
                "created_at": p.created_at.isoformat()
            }
            for p in recent
        ]
    }
