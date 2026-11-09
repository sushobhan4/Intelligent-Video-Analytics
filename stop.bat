@echo off
setlocal

cd /d "%~dp0"

echo ====================================================================
echo   STOPPING NEURAL CCTV SERVICES
echo ====================================================================
echo.

:: 1. Free Port 8000 (FastAPI backend)
echo [*] Checking Port 8000 (Backend)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo [*] Terminating Backend process PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

:: 2. Free Port 5173 (Vite frontend)
echo [*] Checking Port 5173 (Frontend)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173 ^| findstr LISTENING') do (
    echo [*] Terminating Frontend process PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo ====================================================================
echo   [+] All Neural CCTV services have been stopped successfully!
echo ====================================================================
echo.
ping 127.0.0.1 -n 3 >nul
