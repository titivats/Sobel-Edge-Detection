@echo off
set "APP=%~dp0finetune\Finetune.exe"
if not exist "%APP%" (
  echo Fine Tune application not found:
  echo %APP%
  pause
  exit /b 1
)
start "" "%APP%"
