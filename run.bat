@echo off
REM Decision Debugger - local run (cmd)
setlocal
set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" (
    echo Setting up virtual environment...
    python -m venv .venv
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements.txt
)
echo Decision Debugger -^> http://localhost:8000
"%PY%" -m uvicorn backend.main:app --reload --port 8000
