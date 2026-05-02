@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "RUNTIMES=%ROOT%runtimes"
set "APP_DIR=%ROOT%app"
set "SERVER_DIR=%APP_DIR%\server"
set "CLIENT_DIR=%APP_DIR%\client"
set "PYTHONPATH=%APP_DIR%"
set "SETTINGS_DIR=%ROOT%settings"
set "PYPROJECT=%SERVER_DIR%\pyproject.toml"
set "PYTHON_EXE=%RUNTIMES%\python\python.exe"
set "UV_EXE=%RUNTIMES%\uv\uv.exe"
set "NODE_EXE=%RUNTIMES%\nodejs\node.exe"
set "NPM_CMD=%RUNTIMES%\nodejs\npm.cmd"
set "VENV_DIR=%SERVER_DIR%\.venv"
set "UV_PROJECT_ENVIRONMENT=%VENV_DIR%"
set "UV_CACHE_DIR=%RUNTIMES%\.uv-cache"
set "DOTENV=%SETTINGS_DIR%\.env"
set "FRONTEND_DIST=%CLIENT_DIR%\dist"
set "BACKEND_BOOT_LOG=%APP_DIR%\resources\logs\backend_boot.log"
set "UVICORN_MODULE=server.app:app"

set "FASTAPI_HOST=127.0.0.1"
set "FASTAPI_PORT=8000"
set "UI_HOST=127.0.0.1"
set "UI_PORT=8001"
set "RELOAD=false"
set "OPTIONAL_DEPENDENCIES=false"

if exist "%DOTENV%" (
  for /f "usebackq tokens=* delims=" %%L in ("%DOTENV%") do (
    set "line=%%L"
    if not "!line!"=="" if "!line:~0,1!" NEQ "#" if "!line:~0,1!" NEQ ";" (
      for /f "tokens=1,* delims==" %%A in ("!line!") do set "%%A=%%B"
    )
  )
)

if not exist "%PYPROJECT%" (
  echo [FATAL] Missing pyproject: %PYPROJECT%
  exit /b 1
)

if not exist "%UV_EXE%" (
  echo [FATAL] Missing uv runtime: %UV_EXE%
  exit /b 1
)
if not exist "%PYTHON_EXE%" (
  echo [FATAL] Missing python runtime: %PYTHON_EXE%
  exit /b 1
)
if not exist "%NPM_CMD%" (
  echo [FATAL] Missing npm runtime: %NPM_CMD%
  exit /b 1
)

echo [STEP 1/5] Installing dependencies with uv from pyproject.toml
pushd "%SERVER_DIR%" >nul
set "UV_EXTRAS="
if /i "%OPTIONAL_DEPENDENCIES%"=="true" set "UV_EXTRAS=--all-extras"
"%UV_EXE%" sync --python "%PYTHON_EXE%" %UV_EXTRAS%
if errorlevel 1 (
  "%UV_EXE%" sync %UV_EXTRAS%
  if errorlevel 1 (
    popd >nul
    echo [FATAL] uv sync failed.
    exit /b 1
  )
)
popd >nul

if exist "%UV_CACHE_DIR%" rd /s /q "%UV_CACHE_DIR%" >nul 2>&1

echo [STEP 2/5] Installing frontend dependencies
if not exist "%CLIENT_DIR%\node_modules" (
  pushd "%CLIENT_DIR%" >nul
  if exist "package-lock.json" (
    call "%NPM_CMD%" ci
  ) else (
    call "%NPM_CMD%" install
  )
  if errorlevel 1 (
    popd >nul
    echo [FATAL] Frontend dependency install failed.
    exit /b 1
  )
  popd >nul
)

echo [STEP 3/5] Building frontend
if not exist "%FRONTEND_DIST%" (
  pushd "%CLIENT_DIR%" >nul
  call "%NPM_CMD%" run build
  if errorlevel 1 (
    popd >nul
    echo [FATAL] Frontend build failed.
    exit /b 1
  )
  popd >nul
)

echo [STEP 4/5] Pruning uv cache
if exist "%UV_CACHE_DIR%" rd /s /q "%UV_CACHE_DIR%" >nul 2>&1

echo [STEP 5/5] Launching backend and frontend
if exist "%BACKEND_BOOT_LOG%" del /q "%BACKEND_BOOT_LOG%" >nul 2>&1
pushd "%SERVER_DIR%" >nul
start "" /b cmd /c "set ""PYTHONPATH=%APP_DIR%"" && cd /d ""%SERVER_DIR%"" && ""%UV_EXE%"" run --no-sync --python ""%PYTHON_EXE%"" python -m uvicorn %UVICORN_MODULE% --host %FASTAPI_HOST% --port %FASTAPI_PORT% --log-level info > ""%BACKEND_BOOT_LOG%"" 2>&1"
popd >nul

pushd "%CLIENT_DIR%" >nul
start "" /b "%NPM_CMD%" run preview -- --host %UI_HOST% --port %UI_PORT% --strictPort
popd >nul

start "" "http://%UI_HOST%:%UI_PORT%"
echo [SUCCESS] Backend and frontend correctly launched
endlocal
exit /b 0






