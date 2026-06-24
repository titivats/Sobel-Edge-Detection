@echo off
set "APP=%~dp0apps\Realtime_Sobel_YOLO\Realtime_Sobel_YOLO.exe"
if not exist "%APP%" (
  echo Real-time monitor not found:
  echo %APP%
  pause
  exit /b 1
)
start "" "%APP%"
