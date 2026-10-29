@echo off
setlocal
cd /d "%~dp0"

py -3.11 --version >nul 2>nul
if errorlevel 1 (
  echo Python 3.11 not found.
  echo Install it with:  winget install -e --id Python.Python.3.11
  echo Then close this window, open a new one, and run setup.bat again.
  pause
  exit /b 1
)

if not exist venv (
  echo Creating virtual environment...
  py -3.11 -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Install failed. Copy the error above and send it to me.
  pause
  exit /b 1
)

echo.
echo Setup complete. Put audio files in the "input" folder and run denoise.bat
pause
