"""
CARE-AD+ Patients Router (Simplified)
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.models import Patient, Prediction
from app.schemas import PatientCreate, PatientResponse
from app.routers.auth import get_current_active_user

router = APIRouter(prefix="/api/patients", tags=["patients"])


@router.post("/", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(
    patient: PatientCreate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """Create a new patient with simplified info (ID, name, age)"""
    # Check if patient_id exists
    result = await db.execute(
        select(Patient).where(Patient.patient_id == patient.patient_id)
    )
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Patient ID already exists"
        )
    
    db_patient = Patient(
        patient_id=patient.patient_id,
        name=patient.name,
        age=patient.age
    )
    
    db.add(db_patient)
    await db.commit()
    await db.refresh(db_patient)
    
    return db_patient


@router.get("/", response_model=List[PatientResponse])
async def list_patients(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """List all patients"""
    result = await db.execute(
        select(Patient).offset(skip).limit(limit).order_by(Patient.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """Get patient by ID"""
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return patient


@router.get("/search/{query}")
async def search_patients(
    query: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """Search patients by ID or name"""
    result = await db.execute(
        select(Patient).where(
            (Patient.patient_id.ilike(f"%{query}%")) |
            (Patient.name.ilike(f"%{query}%"))
        ).limit(20)
    )
    return result.scalars().all()


@router.get("/{patient_id}/predictions")
async def get_patient_predictions(
    patient_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """Get all predictions for a patient"""
    result = await db.execute(
        select(Prediction)
        .where(Prediction.patient_id == patient_id)
        .order_by(Prediction.created_at.desc())
    )
    return result.scalars().all()


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(
    patient_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """Delete a patient"""
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    await db.delete(patient)
    await db.commit()
