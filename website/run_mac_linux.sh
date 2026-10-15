#!/bin/sh
cd "$(dirname "$0")"
echo "Installing requirements (first run only takes a few minutes)..."
python3 -m pip install -r requirements.txt
echo "Starting the website. Open http://127.0.0.1:5000 in your browser."
python3 app.py
