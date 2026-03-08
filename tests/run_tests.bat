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
set "XREPORT_DIR=%PROJECT_ROOT%\\XREPORT"
set "PYTHON_EXE=%PROJECT_ROOT%\\XREPORT\\resources\\runtimes\\python\\python.exe"
set "VENV_PYTHON=%PROJECT_ROOT%\\.venv\\Scripts\\python.exe"
set "NODEJS_DIR=%PROJECT_ROOT%\\XREPORT\\resources\\runtimes\\nodejs"
set "NPM_CMD=%NODEJS_DIR%\\npm.cmd"
set "FRONTEND_DIR=%XREPORT_DIR%\\client"
set "FRONTEND_DIST=%FRONTEND_DIR%\\dist"
set "FRONTEND_LOCKFILE=%FRONTEND_DIR%\\package-lock.json"
set "DOTENV=%XREPORT_DIR%\\settings\\.env"
set "OPTIONAL_DEPENDENCIES=false"
set "FASTAPI_HOST=127.0.0.1"
set "FASTAPI_PORT=8000"
set "UI_HOST=127.0.0.1"
set "UI_PORT=7861"
set "TEST_RESULT=1"

REM Load env values from .env if present
if exist "%DOTENV%" (
    for /f "usebackq tokens=* delims=" %%A in ("%DOTENV%") do (
        set "line=%%A"
        if not "!line!"=="" if "!line:~0,1!" NEQ "#" if "!line:~0,1!" NEQ ";" (
            for /f "tokens=1,* delims==" %%K in ("!line!") do (
                set "key=%%K"
                set "value=%%L"
                if defined value (
                    for /f "tokens=* delims= " %%Q in ("!value!") do set "value=%%Q"
                    set "value=!value:"=!"
                    if "!value:~0,1!"=="'" if "!value:~-1!"=="'" set "value=!value:~1,-1!"
                )
                if /i "!key!"=="OPTIONAL_DEPENDENCIES" set "OPTIONAL_DEPENDENCIES=!value!"
                if /i "!key!"=="FASTAPI_HOST" set "FASTAPI_HOST=!value!"
                if /i "!key!"=="FASTAPI_PORT" set "FASTAPI_PORT=!value!"
                if /i "!key!"=="UI_HOST" set "UI_HOST=!value!"
                if /i "!key!"=="UI_PORT" set "UI_PORT=!value!"
            )
        )
    )
)

set "FASTAPI_URL_HOST=%FASTAPI_HOST%"
set "UI_URL_HOST=%UI_HOST%"
if /i "%FASTAPI_URL_HOST%"=="0.0.0.0" set "FASTAPI_URL_HOST=127.0.0.1"
if /i "%UI_URL_HOST%"=="0.0.0.0" set "UI_URL_HOST=127.0.0.1"

set "APP_TEST_BACKEND_URL=http://%FASTAPI_URL_HOST%:%FASTAPI_PORT%"
set "APP_TEST_FRONTEND_URL=http://%UI_URL_HOST%:%UI_PORT%"

REM Check for Python (require uv-created .venv)
if exist "%VENV_PYTHON%" (
    set "PYTHON_CMD=%VENV_PYTHON%"
) else (
    echo [ERROR] .venv not found at "%VENV_PYTHON%".
    echo [ERROR] Run XREPORT\\start_on_windows.bat to create the environment.
    exit /b 1
)

REM Check for npm (prefer embedded runtime)
if exist "%NPM_CMD%" (
    set "NPM_RUN=%NPM_CMD%"
) else (
    where npm >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] npm not found in PATH. Please install Node.js or run start_on_windows.bat.
        exit /b 1
    )
    set "NPM_RUN=npm"
)

REM Check for pytest/playwright in the existing .venv only (no installs here)
if /i "%OPTIONAL_DEPENDENCIES%"=="true" (
    "%PYTHON_CMD%" -c "import pytest" >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] pytest not installed in .venv.
        echo [ERROR] Set OPTIONAL_DEPENDENCIES=true and run XREPORT\\start_on_windows.bat.
        exit /b 1
    )

    "%PYTHON_CMD%" -c "import playwright" >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] playwright not installed in .venv.
        echo [ERROR] Set OPTIONAL_DEPENDENCIES=true and run XREPORT\\start_on_windows.bat.
        exit /b 1
    )

    "%PYTHON_CMD%" -c "import psutil" >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] psutil not installed in .venv.
        echo [ERROR] Set OPTIONAL_DEPENDENCIES=true and run XREPORT\\start_on_windows.bat.
        exit /b 1
    )
)

echo.
echo [INFO] Prerequisites verified.
echo [INFO] Backend URL: %APP_TEST_BACKEND_URL%
echo [INFO] Frontend URL: %APP_TEST_FRONTEND_URL%
echo.

REM Check if servers are already running
set "BACKEND_RUNNING=0"
set "FRONTEND_RUNNING=0"

curl -s --max-time 2 %APP_TEST_BACKEND_URL%/docs >nul 2>&1
if %ERRORLEVEL% equ 0 set "BACKEND_RUNNING=1"

curl -s --max-time 2 %APP_TEST_FRONTEND_URL% >nul 2>&1
if %ERRORLEVEL% equ 0 set "FRONTEND_RUNNING=1"

REM Start servers if not running
set "STARTED_BACKEND=0"
set "STARTED_FRONTEND=0"

if "%BACKEND_RUNNING%"=="0" (
    echo [INFO] Starting backend server...
    start "" /B /D "%PROJECT_ROOT%" "%PYTHON_CMD%" -m uvicorn XREPORT.server.app:app --host %FASTAPI_HOST% --port %FASTAPI_PORT%
    set "STARTED_BACKEND=1"
    timeout /t 3 /nobreak >nul
)

if "%FRONTEND_RUNNING%"=="0" (
    if not exist "%FRONTEND_DIR%\\node_modules" (
        echo [INFO] Installing frontend dependencies...
        pushd "%FRONTEND_DIR%" >nul
        if exist "%FRONTEND_LOCKFILE%" (
            call "%NPM_RUN%" ci
        ) else (
            call "%NPM_RUN%" install
        )
        set "npm_ec=%ERRORLEVEL%"
        popd >nul
        if not "%npm_ec%"=="0" (
            echo [ERROR] Frontend dependency install failed with code %npm_ec%.
            exit /b 1
        )
    )

    if not exist "%FRONTEND_DIST%" (
        echo [INFO] Building frontend...
        pushd "%FRONTEND_DIR%" >nul
        call "%NPM_RUN%" run build
        set "npm_build_ec=%ERRORLEVEL%"
        popd >nul
        if not "%npm_build_ec%"=="0" (
            echo [ERROR] Frontend build failed with code %npm_build_ec%.
            exit /b 1
        )
    )

    echo [INFO] Starting frontend server...
    start "" /B /D "%PROJECT_ROOT%" "%PYTHON_CMD%" "%PROJECT_ROOT%\\tests\\spaserver.py" --directory "%FRONTEND_DIST%" --host %UI_HOST% --port %UI_PORT%
    set "STARTED_FRONTEND=1"
    timeout /t 3 /nobreak >nul
)

REM Wait for servers to be ready
echo [INFO] Waiting for servers to be ready...
set "ATTEMPTS=0"
:wait_loop
if %ATTEMPTS% geq 30 (
    echo [ERROR] Servers failed to start within 30 seconds.
    set "TEST_RESULT=1"
    goto cleanup
)

curl -s --max-time 2 %APP_TEST_BACKEND_URL%/docs >nul 2>&1
if %ERRORLEVEL% neq 0 (
    set /a ATTEMPTS+=1
    timeout /t 1 /nobreak >nul
    goto wait_loop
)

curl -s --max-time 2 %APP_TEST_FRONTEND_URL% >nul 2>&1
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
"%PYTHON_CMD%" -m pytest "%PROJECT_ROOT%\\tests" -v --tb=short %*
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
    for /f "tokens=5" %%a in ('netstat -aon ^| findstr :%FASTAPI_PORT% ^| findstr LISTENING') do (
        taskkill /F /PID %%a >nul 2>&1
    )
)

if "%STARTED_FRONTEND%"=="1" (
    echo [INFO] Stopping frontend server...
    for /f "tokens=5" %%a in ('netstat -aon ^| findstr :%UI_PORT% ^| findstr LISTENING') do (
        taskkill /F /PID %%a >nul 2>&1
    )
)

exit /b %TEST_RESULT%

