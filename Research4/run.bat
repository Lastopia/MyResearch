@echo off
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
python main.py run
if errorlevel 1 (
  echo.
  echo Run failed. See output\scheduler and output\runs logs.
  pause
  exit /b 1
)
echo.
echo Run completed. Open output\reports for the dashboard.
pause
