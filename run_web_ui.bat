@echo off
title QueueFlux - Web Control Tower
echo ========================================================
echo  QueueFlux: Canteen Digital Twin Web Control Tower
echo ========================================================
echo Starting FastAPI Web Server at http://localhost:8000 ...
echo.

start "" "http://localhost:8000"
python api.py

pause
