@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" run_ik.py --demo %*
if errorlevel 1 pause
