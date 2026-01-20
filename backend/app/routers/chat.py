"""
CARE-AD+ Chat Router - LLM Assistant
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json

from app.database import get_db
from app.models.models import Prediction, ChatHistory, User
from app.schemas import ChatRequest, ChatResponse
from app.routers.auth import get_current_active_user
from app.services.llm_service import LLMService

router = APIRouter()

# Initialize LLM service
llm_service = LLMService()


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Chat with the LLM assistant.
    
    Modes:
    - technical: Medical terminology for clinicians
    - patient: Simplified explanations for patients/families
    """
    context = None
    
    # If prediction_id provided, get context
    if request.prediction_id:
        result = await db.execute(
            select(Prediction).where(Prediction.id == request.prediction_id)
        )
        prediction = result.scalar_one_or_none()
        
        if prediction:
            context = {
                "predicted_class": prediction.predicted_class,
                "confidence_score": prediction.confidence_score,
                "probabilities": json.loads(prediction.probabilities) if prediction.probabilities else None,
                "clinical_features": json.loads(prediction.clinical_features) if prediction.clinical_features else None,
                "shap_values": json.loads(prediction.shap_values) if prediction.shap_values else None
            }
    
    # Generate response
    try:
        response = await llm_service.generate_response(
            user_message=request.message,
            context=context,
            mode=request.mode
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
    
    # Save chat history
    user_msg = ChatHistory(
        prediction_id=request.prediction_id,
        user_id=current_user.id,
        role="user",
        content=request.message
    )
    assistant_msg = ChatHistory(
        prediction_id=request.prediction_id,
        user_id=current_user.id,
        role="assistant",
        content=response
    )
    
    db.add(user_msg)
    db.add(assistant_msg)
    await db.commit()
    
    return ChatResponse(response=response, context_used=context)


@router.post("/explain/{prediction_id}")
async def explain_prediction(
    prediction_id: int,
    mode: str = "technical",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Generate an automatic explanation for a prediction.
    
    This saves the explanation to the prediction record.
    """
    result = await db.execute(
        select(Prediction).where(Prediction.id == prediction_id)
    )
    prediction = result.scalar_one_or_none()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Build context
    context = {
        "predicted_class": prediction.predicted_class,
        "confidence_score": prediction.confidence_score,
        "probabilities": json.loads(prediction.probabilities) if prediction.probabilities else None,
        "clinical_features": json.loads(prediction.clinical_features) if prediction.clinical_features else None,
        "shap_values": json.loads(prediction.shap_values) if prediction.shap_values else None
    }
    
    # Generate explanation
    prompt = f"Please provide a comprehensive explanation of the Alzheimer's disease prediction results."
    
    try:
        explanation = await llm_service.generate_response(
            user_message=prompt,
            context=context,
            mode=mode
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
    
    # Save explanation to prediction
    if mode == "technical":
        prediction.llm_explanation_technical = explanation
    else:
        prediction.llm_explanation_patient = explanation
    
    await db.commit()
    
    return {
        "prediction_id": prediction_id,
        "mode": mode,
        "explanation": explanation
    }


@router.get("/history/{prediction_id}")
async def get_chat_history(
    prediction_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get chat history for a prediction."""
    result = await db.execute(
        select(ChatHistory)
        .where(ChatHistory.prediction_id == prediction_id)
        .order_by(ChatHistory.created_at)
    )
    messages = result.scalars().all()
    
    return [
        {
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at
        }
        for msg in messages
    ]
