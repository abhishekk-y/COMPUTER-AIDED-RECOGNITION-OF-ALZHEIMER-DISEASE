@echo off
echo ===============================================
echo   CARE-AD+ Backend Setup
echo ===============================================

cd /d "%~dp0"
cd backend

echo.
echo [1/4] Creating virtual environment...
python -m venv venv

echo.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate

echo.
echo [3/4] Installing dependencies...
pip install -r requirements.txt

echo.
echo [4/4] Initializing database...
python -c "from app.database import init_db; import asyncio; asyncio.run(init_db())"

echo.
echo ===============================================
echo   Setup Complete!
echo ===============================================
echo.
echo To start the backend server, run:
echo   cd backend
echo   venv\Scripts\activate
echo   uvicorn app.main:app --reload
echo.
pause
