@echo off
setlocal EnableDelayedExpansion

REM ============================================================================
REM XREPORT Test Runner
REM Automated E2E test execution for Windows
REM ============================================================================

echo.
echo ============================================================
echo  XREPORT Test Runner
echo ============================================================
echo.

REM Store the script directory
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "XREPORT_DIR=%SCRIPT_DIR%.."
set "PYTHON_EXE=%PROJECT_ROOT%\\XREPORT\\resources\\runtimes\\python\\python.exe"
set "VENV_PYTHON=%PROJECT_ROOT%\\.venv\\Scripts\\python.exe"

REM Check for Python (prefer uv-created .venv, then embedded runtime)
if exist "%VENV_PYTHON%" (
    set "PYTHON_CMD=%VENV_PYTHON%"
) else if exist "%PYTHON_EXE%" (
    set "PYTHON_CMD=%PYTHON_EXE%"
) else (
    where python >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Python not found in PATH. Please install Python 3.14+.
        exit /b 1
    )
    set "PYTHON_CMD=python"
)

REM Check for pytest
"%PYTHON_CMD%" -c "import pytest" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing test dependencies...
    "%PYTHON_CMD%" -m pip install -e "%PROJECT_ROOT%[test]"
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install test dependencies.
        exit /b 1
    )
)

REM Check for playwright
"%PYTHON_CMD%" -c "import playwright" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing Playwright...
    "%PYTHON_CMD%" -m pip install pytest-playwright
)

REM Install Playwright browsers if needed
"%PYTHON_CMD%" -m playwright install chromium >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing Playwright browsers...
    "%PYTHON_CMD%" -m playwright install
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install Playwright browsers.
        exit /b 1
    )
)

echo.
echo [INFO] Prerequisites verified.
echo.

REM Check if servers are already running
set "BACKEND_RUNNING=0"
set "FRONTEND_RUNNING=0"

curl -s http://127.0.0.1:8000/docs >nul 2>&1
if %ERRORLEVEL% equ 0 set "BACKEND_RUNNING=1"

curl -s http://127.0.0.1:7861 >nul 2>&1
if %ERRORLEVEL% equ 0 set "FRONTEND_RUNNING=1"

REM Start servers if not running
set "STARTED_BACKEND=0"
set "STARTED_FRONTEND=0"

if "%BACKEND_RUNNING%"=="0" (
    echo [INFO] Starting backend server...
    start /B cmd /c "cd /d %PROJECT_ROOT% && "%PYTHON_CMD%" -m uvicorn XREPORT.server.app:app --host 127.0.0.1 --port 8000" >nul 2>&1
    set "STARTED_BACKEND=1"
    timeout /t 3 /nobreak >nul
)

if "%FRONTEND_RUNNING%"=="0" (
    echo [INFO] Starting frontend server...
    start /B cmd /c "cd /d %XREPORT_DIR%\client && npm run preview" >nul 2>&1
    set "STARTED_FRONTEND=1"
    timeout /t 3 /nobreak >nul
)

REM Wait for servers to be ready
echo [INFO] Waiting for servers to be ready...
set "ATTEMPTS=0"
:wait_loop
if %ATTEMPTS% geq 30 (
    echo [ERROR] Servers failed to start within 30 seconds.
    goto cleanup
)

curl -s http://127.0.0.1:8000/docs >nul 2>&1
if %ERRORLEVEL% neq 0 (
    set /a ATTEMPTS+=1
    timeout /t 1 /nobreak >nul
    goto wait_loop
)

curl -s http://127.0.0.1:7861 >nul 2>&1
if %ERRORLEVEL% neq 0 (
    set /a ATTEMPTS+=1
    timeout /t 1 /nobreak >nul
    goto wait_loop
)

echo [INFO] Servers are ready.
echo.

REM Run tests
echo ============================================================
echo  Running Tests
echo ============================================================
echo.

cd /d "%PROJECT_ROOT%"
"%PYTHON_CMD%" -m pytest tests -v --tb=short %*
set "TEST_RESULT=%ERRORLEVEL%"

echo.
echo ============================================================
if %TEST_RESULT% equ 0 (
    echo  All tests PASSED
) else (
    echo  Some tests FAILED
)
echo ============================================================
echo.

:cleanup
REM Cleanup: Stop servers we started
if "%STARTED_BACKEND%"=="1" (
    echo [INFO] Stopping backend server...
    for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
        taskkill /F /PID %%a >nul 2>&1
    )
)

if "%STARTED_FRONTEND%"=="1" (
    echo [INFO] Stopping frontend server...
    for /f "tokens=5" %%a in ('netstat -aon ^| findstr :7861 ^| findstr LISTENING') do (
        taskkill /F /PID %%a >nul 2>&1
    )
)

exit /b %TEST_RESULT%
