@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "APP_DIR=%ROOT%app"
set "SERVER_DIR=%APP_DIR%\server"
set "LOG_DIR=%APP_DIR%\resources\logs"
set "INIT_DB=%APP_DIR%\scripts\initialize_database.py"
set "TAURI_CLEAN=%ROOT%release\tauri\scripts\clean-tauri-build.ps1"
set "RUNTIMES=%ROOT%runtimes"
set "UV_EXE=%RUNTIMES%\uv\uv.exe"
set "PYTHON_EXE=%RUNTIMES%\python\python.exe"

:menu
cls
echo ================= Setup and Maintenance ================
echo 1. Initialize database
echo 2. Remove logs
echo 3. Clean desktop build artifacts
echo 4. Uninstall runtime tools
echo 5. Exit
set /p choice=Select an option (1-5): 
if "%choice%"=="1" goto initdb
if "%choice%"=="2" goto logs
if "%choice%"=="3" goto cleantauri
if "%choice%"=="4" goto uninstall
if "%choice%"=="5" goto done
goto menu

:initdb
if not exist "%INIT_DB%" (
  echo [FATAL] Missing script: %INIT_DB%
  pause
  goto menu
)
pushd "%SERVER_DIR%" >nul
if exist "%UV_EXE%" (
  "%UV_EXE%" run --python "%PYTHON_EXE%" python "%INIT_DB%"
) else (
  python "%INIT_DB%"
)
popd >nul
pause
goto menu

:logs
if exist "%LOG_DIR%\*.log" del /q "%LOG_DIR%\*.log"
echo [INFO] Log cleanup done.
pause
goto menu

:cleantauri
if exist "%TAURI_CLEAN%" powershell -NoProfile -ExecutionPolicy Bypass -File "%TAURI_CLEAN%"
pause
goto menu

:uninstall
if exist "%RUNTIMES%\uv" rd /s /q "%RUNTIMES%\uv"
if exist "%RUNTIMES%\.uv-cache" rd /s /q "%RUNTIMES%\.uv-cache"
if exist "%APP_DIR%\server\.venv" rd /s /q "%APP_DIR%\server\.venv"
echo [INFO] Runtime cleanup done.
pause
goto menu

:done
endlocal
exit /b 0
