@echo off
cd /d "%~dp0"
if not exist .venv python -m venv .venv
rem installs or completes the dependencies (quick when they are already there)
.venv\Scripts\python -m pip install -q --disable-pip-version-check -r requirements.txt
rem --open: the application opens the browser itself, on the port it actually uses
.venv\Scripts\python app.py --open
