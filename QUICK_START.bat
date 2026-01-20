@echo off
echo ===============================================
echo   CARE-AD+ QUICK START - FULL SETUP
echo ===============================================

cd /d "%~dp0"

echo.
echo [Step 1/6] Creating Python virtual environment...
cd backend
python -m venv venv
call venv\Scripts\activate

echo.
echo [Step 2/6] Installing Python dependencies...
pip install fastapi uvicorn[standard] python-multipart sqlalchemy aiosqlite python-jose[cryptography] passlib[bcrypt] torch torchvision numpy pandas scikit-learn Pillow shap matplotlib seaborn ollama reportlab pydantic pydantic-settings aiofiles

echo.
echo [Step 3/6] Setting up frontend...
cd ..\frontend
call npm install

echo.
echo [Step 4/6] Creating required directories...
cd ..
if not exist "models" mkdir models
if not exist "uploads" mkdir uploads  
if not exist "reports" mkdir reports

echo.
echo [Step 5/6] Pulling Ollama LLM model (phi3)...
echo This may take a few minutes on first run...
where ollama >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Ollama found. Pulling phi3 model...
    ollama pull phi3
    echo.
    echo Model downloaded successfully!
) else (
    echo.
    echo ================================================
    echo   OLLAMA NOT INSTALLED
    echo ================================================
    echo   Please install Ollama from: https://ollama.ai
    echo   Then run: ollama pull phi3
    echo   The app will work without LLM, but AI chat
    echo   will use fallback responses.
    echo ================================================
    echo.
)

echo.
echo [Step 6/6] Starting servers...
echo.
echo ===============================================
echo   STARTING CARE-AD+ APPLICATION
echo ===============================================
echo.

:: Start Ollama serve in background (if available)
where ollama >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Starting Ollama LLM server...
    start "Ollama Server" cmd /c "ollama serve"
    timeout /t 3 /nobreak >nul
)

:: Start backend in new window
start "CARE-AD+ Backend" cmd /k "cd backend && venv\Scripts\activate && echo. && echo Backend running at http://localhost:8000 && echo API Docs: http://localhost:8000/docs && echo. && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

:: Wait a moment for backend to start
timeout /t 5 /nobreak >nul

:: Start frontend in new window
start "CARE-AD+ Frontend" cmd /k "cd frontend && echo. && echo Frontend running at http://localhost:3000 && echo. && npm run dev"

echo.
echo ===============================================
echo   CARE-AD+ IS STARTING!
echo ===============================================
echo.
echo   Frontend: http://localhost:3000
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo   LLM:      Ollama with phi3 model
echo.
echo   Terminal windows opened:
echo   - Ollama LLM Server (if installed)
echo   - Backend server (Python/FastAPI)
echo   - Frontend server (React/Vite)
echo.
echo   Press any key to close this window
echo   (Servers will keep running)
echo ===============================================
pause >nul
