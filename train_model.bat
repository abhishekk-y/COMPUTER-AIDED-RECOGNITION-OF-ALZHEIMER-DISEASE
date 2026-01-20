@echo off
echo ===============================================
echo   CARE-AD+ Model Training
echo ===============================================

cd /d "%~dp0"

:: Check for virtual environment
if not exist "backend\venv\Scripts\activate" (
    echo Creating virtual environment...
    cd backend
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
    cd ..
) else (
    call backend\venv\Scripts\activate
)

echo.
echo Starting model training...
echo Dataset: archive\combined_images
echo.

cd backend
python -c "
import sys
sys.path.insert(0, '.')
from ml.train import train_model

train_model(
    dataset_dir='../archive/combined_images',
    epochs=30,
    batch_size=16,
    learning_rate=0.0001,
    save_dir='models',
    image_size=224
)
"

echo.
echo ===============================================
echo   Training Complete!
echo ===============================================
echo Model saved in: backend\models\
echo.
pause
