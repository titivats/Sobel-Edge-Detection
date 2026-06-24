@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_yolo_training.ps1" -DryRun
set EXIT_CODE=%ERRORLEVEL%
pause
exit /b %EXIT_CODE%
