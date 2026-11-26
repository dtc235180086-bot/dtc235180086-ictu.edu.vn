@echo off
cd /d "%~dp0"
where py >nul 2>&1
if not errorlevel 1 (
  py -3 src\web_app.py
) else (
  python src\web_app.py
)
pause
