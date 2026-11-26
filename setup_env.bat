@echo off
rem Tao moi truong ao .venv va cai thu vien (chi can chay 1 lan)
cd /d "%~dp0"
where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
if errorlevel 1 (echo Khong tao duoc moi truong ao. Hay cai Python 3.10+ tu python.org & pause & exit /b 1)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
echo.
echo Cai dat xong. Chay run_all.bat de tai lap thuc nghiem, app.bat de mo chuong trinh du doan.
pause
