@echo off
echo ============================================================
echo   QueueFlux - Starting Docker Multi-Container Stack
echo ============================================================
echo Checking Docker engine status...
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker CLI was not found in PATH.
    echo Please ensure Docker Desktop is installed and running:
    echo https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

echo Building and starting QueueFlux App + TimescaleDB...
docker compose up --build -d

if %ERRORLEVEL% equ 0 (
    echo ============================================================
    echo   QueueFlux is now live!
    echo   App & Consumer UI : http://localhost:8000
    echo   Interactive API Docs: http://localhost:8000/docs
    echo   TimescaleDB Port    : 5432
    echo ============================================================
    timeout /t 3 >nul
    start http://localhost:8000
) else (
    echo [ERROR] Docker compose failed to start.
)
pause
