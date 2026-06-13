@echo off
REM ultraeasy-upscaler launcher
chcp 65001 >nul
set PYTHONUTF8=1
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -m app.main %*
endlocal
