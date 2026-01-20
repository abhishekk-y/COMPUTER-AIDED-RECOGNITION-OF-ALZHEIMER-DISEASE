"""
CARE-AD+ Reports Router - Real PDF Generation
"""
import os
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.models import Prediction, Patient
from app.schemas import ReportRequest, ReportResponse
from app.routers.auth import get_current_active_user
from app.services.report_service import get_report_service
from app.config import settings

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    request: ReportRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """Generate a clinical PDF report for a prediction"""
    # Get prediction with patient
    result = await db.execute(
        select(Prediction)
        .options(selectinload(Prediction.patient))
        .where(Prediction.id == request.prediction_id)
    )
    prediction = result.scalar_one_or_none()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Generate report
    report_service = get_report_service()
    
    try:
        report_path = report_service.generate_clinical_report(
            prediction=prediction,
            patient=prediction.patient,
            include_gradcam=request.include_gradcam,
            include_shap=False,
            include_llm_explanation=request.include_explanation
        )
        
        # Update prediction with report path
        prediction.report_path = report_path
        await db.commit()
        
        # Generate report ID
        report_id = os.path.basename(report_path).replace(".pdf", "")
        
        return ReportResponse(
            report_id=report_id,
            report_path=report_path,
            download_url=f"/api/reports/download/{report_id}",
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/download/{report_id}")
async def download_report(
    report_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Download a generated report PDF"""
    # Find report file
    reports_dir = settings.REPORTS_DIR
    
    for filename in os.listdir(reports_dir):
        if report_id in filename and filename.endswith('.pdf'):
            file_path = os.path.join(reports_dir, filename)
            return FileResponse(
                path=file_path,
                filename=filename,
                media_type='application/pdf'
            )
    
    raise HTTPException(status_code=404, detail="Report not found")


@router.get("/")
async def list_reports(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """List all generated reports"""
    result = await db.execute(
        select(Prediction)
        .options(selectinload(Prediction.patient))
        .where(Prediction.report_path.isnot(None))
        .order_by(Prediction.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    predictions = result.scalars().all()
    
    return [
        {
            "id": p.id,
            "patient_id": p.patient.patient_id,
            "patient_name": p.patient.name,
            "predicted_class": p.predicted_class,
            "report_path": p.report_path,
            "created_at": p.created_at.isoformat(),
            "download_url": f"/api/reports/download/{os.path.basename(p.report_path).replace('.pdf', '')}" if p.report_path else None
        }
        for p in predictions
    ]


@router.get("/prediction/{prediction_id}")
async def get_report_for_prediction(
    prediction_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """Get or generate report for a specific prediction"""
    result = await db.execute(
        select(Prediction)
        .options(selectinload(Prediction.patient))
        .where(Prediction.id == prediction_id)
    )
    prediction = result.scalar_one_or_none()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    if prediction.report_path and os.path.exists(prediction.report_path):
        report_id = os.path.basename(prediction.report_path).replace(".pdf", "")
        return {
            "exists": True,
            "report_id": report_id,
            "download_url": f"/api/reports/download/{report_id}"
        }
    
    return {"exists": False, "message": "Report not yet generated"}
