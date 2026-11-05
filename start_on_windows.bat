@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_maintenance.ps1" -Launch
exit /b %ERRORLEVEL%
