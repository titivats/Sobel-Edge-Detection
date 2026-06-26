@echo off
set "APP=%~dp0apps\Realtime_Sobel_YOLO\Realtime_Sobel_YOLO.exe"
if exist "%APP%" (
  start "" "%APP%"
  exit /b 0
)

call "%~dp0image_classification\Realtime_UI.bat"
exit /b %ERRORLEVEL%
