@echo off
set "APP=%~dp0apps\Sobel_YOLO_Fine_Tune\Sobel_YOLO_Fine_Tune.exe"
if not exist "%APP%" (
  echo Fine Tune application not found:
  echo %APP%
  pause
  exit /b 1
)
start "" "%APP%"
