@echo off
echo ===============================================
echo   CARE-AD+ Full Stack Startup
echo ===============================================

echo.
echo Starting Backend Server...
start "CARE-AD+ Backend" cmd /k "cd backend && venv\Scripts\activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak >nul

echo.
echo Starting Frontend Server...
start "CARE-AD+ Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ===============================================
echo   Servers Started!
echo ===============================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to exit (servers will keep running)
pause >nul
