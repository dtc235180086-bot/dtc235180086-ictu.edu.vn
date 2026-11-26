@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist .venv\Scripts\python.exe call setup_env.bat
.venv\Scripts\python src\predict_app.py
