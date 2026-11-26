@echo off
rem Chay lai toan bo thuc nghiem: Chuong 1 (du lieu) va Chuong 2-3 (mo hinh, danh gia)
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist .venv\Scripts\python.exe call setup_env.bat
.venv\Scripts\python src\chapter1_pipeline.py data\insurance.csv || goto :err
.venv\Scripts\python src\chapter2_models.py || goto :err
.venv\Scripts\python src\make_html_report.py || goto :err
echo.
echo Xong. Ket qua: results\  (mo results\BaoCao_TrucQuan.html de xem)
pause
exit /b 0
:err
echo Co loi khi chay. Xem thong bao phia tren.
pause
exit /b 1
