@echo off
cd /d "%~dp0"
echo Installing requirements (first run only takes a few minutes)...
python -m pip install -r requirements.txt
echo.
echo Starting the website. Open http://127.0.0.1:5000 in your browser.
python app.py
pause
