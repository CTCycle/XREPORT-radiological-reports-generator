@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM == XREPORT E2E Test Runner
REM == Automatically starts the application, runs tests, and cleans up.
REM ============================================================================

set "tests_folder=%~dp0"
set "root_folder=%tests_folder%..\"
set "project_folder=%root_folder%XREPORT\"
set "runtimes_dir=%project_folder%resources\runtimes"
set "settings_dir=%project_folder%settings"

set "python_dir=%runtimes_dir%\python"
set "python_exe=%python_dir%\python.exe"
set "uv_dir=%runtimes_dir%\uv"
set "uv_exe=%uv_dir%\uv.exe"
set "nodejs_dir=%runtimes_dir%\nodejs"
set "node_exe=%nodejs_dir%\node.exe"
set "npm_cmd=%nodejs_dir%\npm.cmd"

set "DOTENV=%settings_dir%\.env"
set "FRONTEND_DIR=%project_folder%client"
set "FRONTEND_DIST=%FRONTEND_DIR%\dist"
set "UVICORN_MODULE=XREPORT.server.app:app"

title XREPORT Test Runner
echo.
echo ============================================================================
echo    XREPORT E2E Test Runner
echo ============================================================================
echo.

REM ============================================================================
REM == Check prerequisites
REM ============================================================================
if not exist "%python_exe%" (
    echo [ERROR] Python not found. Please run XREPORT\start_on_windows.bat first.
    goto error
)
if not exist "%uv_exe%" (
    echo [ERROR] uv not found. Please run XREPORT\start_on_windows.bat first.
    goto error
)
if not exist "%node_exe%" (
    echo [ERROR] Node.js not found. Please run XREPORT\start_on_windows.bat first.
    goto error
)
if not exist "%npm_cmd%" (
    echo [ERROR] npm not found. Please run XREPORT\start_on_windows.bat first.
    goto error
)

echo [OK] All prerequisites found.

REM ============================================================================
REM == Load environment variables
REM ============================================================================
set "FASTAPI_HOST=127.0.0.1"
set "FASTAPI_PORT=8000"
set "UI_HOST=127.0.0.1"
set "UI_PORT=7861"

if exist "%DOTENV%" (
    for /f "usebackq tokens=* delims=" %%L in ("%DOTENV%") do (
        set "line=%%L"
        if not "!line!"=="" if "!line:~0,1!" NEQ "#" if "!line:~0,1!" NEQ ";" (
            for /f "tokens=1* delims==" %%K in ("!line!") do (
                set "k=%%K"
                set "v=%%L"
                if defined v (
                    if "!v:~0,1!"=="\"" set "v=!v:~1,-1!"
                    if "!v:~0,1!"=="'" set "v=!v:~1,-1!"
                )
                set "!k!=!v!"
            )
        )
    )
)

REM ============================================================================
REM == Force portable runtimes (avoid global Python/npm)
REM ============================================================================
set "PATH=%python_dir%;%nodejs_dir%;%PATH%"
set "PYTHONHOME=%python_dir%"
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "VIRTUAL_ENV="
set "__PYVENV_LAUNCHER__="
set "PYTHON=%python_exe%"
set "npm_config_python=%python_exe%"

REM ============================================================================
REM == Configure pytest / Playwright options
REM ============================================================================
if not defined E2E_HEADLESS set "E2E_HEADLESS=true"
if not defined E2E_BROWSER set "E2E_BROWSER=chromium"
if not defined E2E_SLOWMO set "E2E_SLOWMO=0"
if not defined E2E_PWDEBUG set "E2E_PWDEBUG=0"

set "PYTEST_ARGS=tests -v --tb=short"
if defined E2E_BROWSER set "PYTEST_ARGS=!PYTEST_ARGS! --browser !E2E_BROWSER!"
if /i "!E2E_HEADLESS!"=="false" set "PYTEST_ARGS=!PYTEST_ARGS! --headed"
if /i "!E2E_HEADLESS!"=="0" set "PYTEST_ARGS=!PYTEST_ARGS! --headed"
if not "!E2E_SLOWMO!"=="0" set "PYTEST_ARGS=!PYTEST_ARGS! --slowmo !E2E_SLOWMO!"
if /i "!E2E_PWDEBUG!"=="1" set "PWDEBUG=1" & set "PYTEST_ARGS=!PYTEST_ARGS! --headed"
if /i "!E2E_PWDEBUG!"=="true" set "PWDEBUG=1" & set "PYTEST_ARGS=!PYTEST_ARGS! --headed"

REM ============================================================================
REM == Install Playwright browsers if needed
REM ============================================================================
echo [STEP 1/4] Checking Playwright browsers...
"%uv_exe%" run --python "%python_exe%" python -m playwright install chromium >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Installing Playwright browsers...
    "%uv_exe%" run --python "%python_exe%" python -m playwright install
)
echo [OK] Playwright browsers ready.

REM ============================================================================
REM == Start backend
REM ============================================================================
echo [STEP 2/4] Starting backend server...
call :kill_port %FASTAPI_PORT%
start "" /b "%uv_exe%" run --python "%python_exe%" python -m uvicorn %UVICORN_MODULE% --host %FASTAPI_HOST% --port %FASTAPI_PORT% --log-level warning

REM Wait for backend to be ready
echo [INFO] Waiting for backend to start...
timeout /t 5 /nobreak >nul

REM ============================================================================
REM == Start frontend
REM ============================================================================
echo [STEP 3/4] Starting frontend server...

if not exist "%FRONTEND_DIR%\node_modules" (
    echo [INFO] Installing frontend dependencies...
    pushd "%FRONTEND_DIR%" >nul
    call "%npm_cmd%" install
    popd >nul
)

if not exist "%FRONTEND_DIST%" (
    echo [INFO] Building frontend...
    pushd "%FRONTEND_DIR%" >nul
    call "%npm_cmd%" run build
    popd >nul
)

call :kill_port %UI_PORT%
pushd "%FRONTEND_DIR%" >nul
start "" /b "%npm_cmd%" run preview -- --host %UI_HOST% --port %UI_PORT% --strictPort
popd >nul

REM Wait for frontend to be ready
echo [INFO] Waiting for frontend to start...
timeout /t 3 /nobreak >nul

REM ============================================================================
REM == Run tests
REM ============================================================================
echo [STEP 4/4] Running E2E tests...
echo.
echo ============================================================================

pushd "%root_folder%" >nul
"%uv_exe%" run --python "%python_exe%" python -m pytest %PYTEST_ARGS%
set "test_result=%ERRORLEVEL%"
popd >nul

echo.
echo ============================================================================

REM ============================================================================
REM == Cleanup: Stop servers
REM ============================================================================
echo [CLEANUP] Stopping servers...
call :kill_port %FASTAPI_PORT%
call :kill_port %UI_PORT%

if %test_result% EQU 0 (
    echo [SUCCESS] All tests passed!
    endlocal & exit /b 0
) else (
    echo [FAILED] Some tests failed. Exit code: %test_result%
    pause
    endlocal & exit /b %test_result%
)

REM ============================================================================
REM == Error
REM ============================================================================
:error
echo.
echo !!! An error occurred. !!!
pause
endlocal & exit /b 1

REM ============================================================================
REM == Kill process on port
REM ============================================================================
:kill_port
set "target_port=%~1"
if not defined target_port goto :eof
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R ":%target_port%"') do (
    taskkill /PID %%P /F >nul 2>&1
)
goto :eof
