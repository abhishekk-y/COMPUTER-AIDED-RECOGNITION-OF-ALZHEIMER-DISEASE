@echo off
echo ===============================================
echo   CARE-AD+ Ollama LLM Setup
echo ===============================================
echo.

:: Check if Ollama is installed
where ollama >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Ollama is not installed!
    echo.
    echo Please install Ollama first:
    echo 1. Download from: https://ollama.ai/download
    echo 2. Run the installer
    echo 3. Restart this script
    echo.
    pause
    exit /b 1
)

echo ✅ Ollama is installed
echo.

:: Check if Ollama is running
echo Checking Ollama service...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Starting Ollama service...
    start "Ollama Server" cmd /c "ollama serve"
    timeout /t 5 /nobreak >nul
)

echo ✅ Ollama service is running
echo.

:: Pull the phi3 model
echo ===============================================
echo   Downloading phi3 Model
echo ===============================================
echo.
echo This will download ~2.3GB. Please wait...
echo.

ollama pull phi3

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===============================================
    echo   ✅ Setup Complete!
    echo ===============================================
    echo.
    echo Model: phi3
    echo Status: Ready to use
    echo.
    echo The LLM assistant is now configured for:
    echo - Technical medical explanations
    echo - Patient-friendly summaries
    echo - Interactive Q&A
    echo.
) else (
    echo.
    echo ❌ Failed to download model
    echo Please check your internet connection
    echo.
)

pause
