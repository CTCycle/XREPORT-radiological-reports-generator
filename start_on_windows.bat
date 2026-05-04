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
set "NPM_CMD=%RUNTIMES%\nodejs\npm.cmd"
set "UV_CACHE_DIR=%RUNTIMES%\.uv-cache"
set "DOTENV=%SETTINGS_DIR%\.env"
set "FRONTEND_DIST=%CLIENT_DIR%\dist"
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

if not exist "%PYPROJECT%" ( echo [FATAL] Missing pyproject: %PYPROJECT% & exit /b 1 )
if not exist "%UV_EXE%" ( echo [FATAL] Missing uv runtime: %UV_EXE% & exit /b 1 )
if not exist "%PYTHON_EXE%" ( echo [FATAL] Missing python runtime: %PYTHON_EXE% & exit /b 1 )
if not exist "%NPM_CMD%" ( echo [FATAL] Missing npm runtime: %NPM_CMD% & exit /b 1 )

echo [STEP 1/5] Installing dependencies with uv from pyproject.toml
pushd "%SERVER_DIR%" >nul
set "UV_EXTRAS="
if /i "%OPTIONAL_DEPENDENCIES%"=="true" set "UV_EXTRAS=--all-extras"
"%UV_EXE%" sync --python "%PYTHON_EXE%" %UV_EXTRAS%
if errorlevel 1 (
  "%UV_EXE%" sync %UV_EXTRAS%
  if errorlevel 1 ( popd >nul & echo [FATAL] uv sync failed. & exit /b 1 )
)
popd >nul

echo [STEP 2/5] Installing frontend dependencies
if not exist "%CLIENT_DIR%\node_modules" (
  pushd "%CLIENT_DIR%" >nul
  if exist "package-lock.json" ( call "%NPM_CMD%" ci ) else ( call "%NPM_CMD%" install )
  if errorlevel 1 ( popd >nul & echo [FATAL] Frontend dependency install failed. & exit /b 1 )
  popd >nul
)

echo [STEP 3/5] Building frontend
if not exist "%FRONTEND_DIST%" (
  pushd "%CLIENT_DIR%" >nul
  call "%NPM_CMD%" run build
  if errorlevel 1 ( popd >nul & echo [FATAL] Frontend build failed. & exit /b 1 )
  popd >nul
)

echo [STEP 4/5] Pruning uv cache
if exist "%UV_CACHE_DIR%" rd /s /q "%UV_CACHE_DIR%" >nul 2>&1

echo [STEP 5/5] Launching backend and frontend
call :kill_port %FASTAPI_PORT%
pushd "%SERVER_DIR%" >nul
start "" /b "%UV_EXE%" run --no-sync --python "%PYTHON_EXE%" python -m uvicorn %UVICORN_MODULE% --host %FASTAPI_HOST% --port %FASTAPI_PORT% --log-level info
popd >nul

set "BACKEND_BASE_URL=http://%FASTAPI_HOST%:%FASTAPI_PORT%"
echo [WAIT] Waiting for backend readiness at %BACKEND_BASE_URL%...
for /L %%i in (1,1,60) do (
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$base='%BACKEND_BASE_URL%'; $paths=@('/api/health','/health','/docs','/'); foreach ($p in $paths) { try { $r = Invoke-WebRequest -UseBasicParsing -Uri ($base + $p) -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300) { exit 0 } } catch {} }; exit 1" >nul 2>&1
  if !errorlevel! equ 0 goto :backend_ready
  timeout /t 1 /nobreak >nul 2>&1
)
echo [FATAL] Backend did not become ready at %BACKEND_BASE_URL%.
exit /b 1

:backend_ready
pushd "%CLIENT_DIR%" >nul
call :kill_port %UI_PORT%
start "" /b "%NPM_CMD%" run preview -- --host %UI_HOST% --port %UI_PORT% --strictPort
popd >nul

start "" "http://%UI_HOST%:%UI_PORT%"
echo [SUCCESS] Backend and frontend correctly launched
endlocal
exit /b 0

:kill_port
set "target_port=%~1"
if not defined target_port goto :eof
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R ":%target_port%"') do taskkill /PID %%P /F >nul 2>&1
goto :eof
