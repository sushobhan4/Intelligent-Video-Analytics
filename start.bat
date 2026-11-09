@echo off
setlocal

cd /d "%~dp0"

echo ====================================================================
echo   NEURAL CCTV ANALYSER - AUTONOMOUS FORENSIC SYSTEM
echo ====================================================================
echo.

:: Detect Python 3.13 executable with PyTorch & ChromaDB installed
set "PYTHON_EXE=C:\Users\halda\AppData\Local\Programs\Python\Python313\python.exe"
if not exist "%PYTHON_EXE%" (
    if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    ) else (
        set "PYTHON_EXE=python"
    )
)

echo [*] Using Python: %PYTHON_EXE%
echo [*] Checking dependencies...

:: Test python dependencies
"%PYTHON_EXE%" -c "import fastapi, chromadb, keras_hub, torch; print('[+] Python dependencies verified!')" 2>nul
if errorlevel 1 (
    echo [!] Warning: Could not verify Python packages. Continuing anyway...
)

echo.
echo ====================================================================
echo   Starting Services:
echo     1. FastAPI Backend (Qwen3-VL + Google VideoPrism 768-d + ChromaDB)
echo     2. Vite Frontend   (React 19 Dashboard + Semantic Search)
echo ====================================================================
echo.

:: 1. Launch FastAPI Backend in a dedicated window
start "Neural CCTV - Backend API (Port 8000)" "%PYTHON_EXE%" server.py

:: 2. Launch Vite Frontend in a dedicated window
start "Neural CCTV - Frontend (Port 5173)" cmd /k "cd frontend && npm run dev"

echo [*] Waiting 3 seconds for servers to initialize...
ping 127.0.0.1 -n 4 >nul

:: 3. Open Web Browser
echo [*] Opening Neural CCTV Dashboard in browser...
start "" "http://localhost:5173"

echo.
echo ====================================================================
echo   Both services are now running in background windows!
echo     - Web Dashboard:  http://localhost:5173
echo     - Backend API:    http://127.0.0.1:8000
echo     - API Docs:       http://127.0.0.1:8000/docs
echo.
echo   To stop all services, double-click stop.bat
echo ====================================================================
echo.
pause
