@echo off
echo ===============================================
echo   CARE-AD+ Model Training
echo ===============================================

cd /d "%~dp0"
cd backend

echo.
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Starting model training...
echo Dataset: ..\archive
echo.

python -m ml.train --dataset "..\archive" --epochs 50 --batch-size 32

echo.
echo ===============================================
echo   Training Complete!
echo ===============================================
pause
