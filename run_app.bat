@echo off
REM Double-click launcher: bypasses PowerShell execution-policy problems for this one script only.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_app.ps1"
pause
