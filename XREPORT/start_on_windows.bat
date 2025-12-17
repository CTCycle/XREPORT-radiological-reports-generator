@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM == Configuration
REM ============================================================================
set "project_folder=%~dp0"
set "root_folder=%project_folder%..\"
set "runtimes_dir=%project_folder%resources\runtimes"
set "settings_dir=%project_folder%settings"
set "python_dir=%runtimes_dir%\python"
set "python_exe=%python_dir%\python.exe"
set "python_pth_file=%python_dir%\python312._pth"
set "env_marker=%python_dir%\.is_installed"

set "uv_dir=%runtimes_dir%\uv"
set "uv_exe=%uv_dir%\uv.exe"
set "uv_zip_path=%uv_dir%\uv.zip"
set "UV_CACHE_DIR=%runtimes_dir%\uv_cache"

set "py_version=3.12.10"
set "python_zip_filename=python-%py_version%-embed-amd64.zip"
set "python_zip_url=https://www.python.org/ftp/python/%py_version%/%python_zip_filename%"
set "python_zip_path=%python_dir%\%python_zip_filename%"

set "UV_CHANNEL=latest"
set "UV_ZIP_AMD=https://github.com/astral-sh/uv/releases/%UV_CHANNEL%/download/uv-x86_64-pc-windows-msvc.zip"
set "UV_ZIP_ARM=https://github.com/astral-sh/uv/releases/%UV_CHANNEL%/download/uv-aarch64-pc-windows-msvc.zip"

set "nodejs_version=22.12.0"
set "nodejs_dir=%runtimes_dir%\nodejs"
set "nodejs_zip_filename=node-v%nodejs_version%-win-x64.zip"
set "nodejs_zip_url=https://nodejs.org/dist/v%nodejs_version%/%nodejs_zip_filename%"
set "nodejs_zip_path=%nodejs_dir%\%nodejs_zip_filename%"
set "node_exe=%nodejs_dir%\node.exe"
set "npm_cmd=%nodejs_dir%\npm.cmd"
set "env_marker_node=%nodejs_dir%\.is_installed"

set "pyproject=%root_folder%pyproject.toml"
set "UVICORN_MODULE=APP.server.app:app"
set "FRONTEND_DIR=%project_folder%client"
set "FRONTEND_DIST=%FRONTEND_DIR%\dist"

set "DOTENV=%settings_dir%\.env"
set "TMPDL=%TEMP%\app_dl.ps1"
set "TMPEXP=%TEMP%\app_expand.ps1"
set "TMPTXT=%TEMP%\app_txt.ps1"
set "TMPFIND=%TEMP%\app_find_uv.ps1"
set "TMPVER=%TEMP%\app_pyver.ps1"
set "TMPFINDNODE=%TEMP%\app_find_node.ps1"

set "UV_LINK_MODE=copy"

title APP Launcher
echo.

set "NPM_CMD=%npm_cmd%"
set "NODE_CMD=%node_exe%"

REM ============================================================================
REM == Prepare helper PowerShell scripts
REM ============================================================================
echo $ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri $args[0] -OutFile $args[1] > "%TMPDL%"
echo $ErrorActionPreference='Stop'; Expand-Archive -LiteralPath $args[0] -DestinationPath $args[1] -Force > "%TMPEXP%"
echo $ErrorActionPreference='Stop'; (Get-Content -LiteralPath $args[0]) -replace '#import site','import site' ^| Set-Content -LiteralPath $args[0] > "%TMPTXT%"
echo $ErrorActionPreference='Stop'; (Get-ChildItem -LiteralPath $args[0] -Recurse -Filter 'uv.exe' ^| Select-Object -First 1).FullName > "%TMPFIND%"
echo $ErrorActionPreference='Stop'; ^& $args[0] -c "import platform;print(platform.python_version())" > "%TMPVER%"
echo $ErrorActionPreference='Stop'; (Get-ChildItem -LiteralPath $args[0] -Recurse -Filter 'node.exe' ^| Select-Object -First 1).FullName > "%TMPFINDNODE%"

REM ============================================================================
REM == Step 1: Ensure Python (embeddable)
REM ============================================================================
echo [STEP 1/5] Setting up Python (embeddable) locally
if not exist "%python_dir%" md "%python_dir%" >nul 2>&1

if not exist "%python_exe%" (
  echo [DL] %python_zip_url%
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPDL%" "%python_zip_url%" "%python_zip_path%" || goto error
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPEXP%" "%python_zip_path%" "%python_dir%" || goto error
  del /q "%python_zip_path%" >nul 2>&1
)

if exist "%python_pth_file%" (
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPTXT%" "%python_pth_file%" || goto error
)

for /f "delims=" %%V in ('powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPVER%" "%python_exe%"') do set "found_py=%%V"
echo [OK] Python ready: !found_py!

REM ============================================================================
REM == Step 2: Ensure uv (portable)
REM ============================================================================
echo [STEP 2/5] Installing uv (portable)
if not exist "%uv_dir%" md "%uv_dir%" >nul 2>&1

set "uv_zip_url=%UV_ZIP_AMD%"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "uv_zip_url=%UV_ZIP_ARM%"

if not exist "%uv_exe%" (
  echo [DL] %uv_zip_url%
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPDL%" "%uv_zip_url%" "%uv_zip_path%" || goto error
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPEXP%" "%uv_zip_path%" "%uv_dir%" || goto error
  del /q "%uv_zip_path%" >nul 2>&1

  for /f "delims=" %%F in ('powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPFIND%" "%uv_dir%"') do set "found_uv=%%F"
  if not defined found_uv (
    echo [FATAL] uv.exe not found after extraction.
    goto error
  )
  if /i not "%found_uv%"=="%uv_exe%" copy /y "%found_uv%" "%uv_exe%" >nul
)

"%uv_exe%" --version >nul 2>&1 && for /f "delims=" %%V in ('"%uv_exe%" --version') do echo %%V

REM ============================================================================
REM == Step 3: Ensure Node.js (portable)
REM ============================================================================
echo [STEP 3/5] Installing Node.js (portable)
if not exist "%nodejs_dir%" md "%nodejs_dir%" >nul 2>&1

if not exist "%node_exe%" (
  echo [DL] %nodejs_zip_url%
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPDL%" "%nodejs_zip_url%" "%nodejs_zip_path%" || goto error
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPEXP%" "%nodejs_zip_path%" "%nodejs_dir%" || goto error
  del /q "%nodejs_zip_path%" >nul 2>&1

  for /f "delims=" %%F in ('powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPFINDNODE%" "%nodejs_dir%"') do set "found_node=%%F"
  if not defined found_node (
    echo [FATAL] node.exe not found after extraction.
    goto error
  )

  if /i not "%found_node%"=="%node_exe%" (
    for %%D in ("!found_node!") do set "node_parent=%%~dpD"
    xcopy /s /e /i /q "!node_parent!*" "%nodejs_dir%" >nul
  )
)

if exist "%node_exe%" (
  for /f "delims=" %%V in ('"%node_exe%" --version') do echo [OK] Node.js ready: %%V
  set "NPM_CMD=%npm_cmd%"
  set "NODE_CMD=%node_exe%"
  set "PATH=%nodejs_dir%;%PATH%"

) else (
  echo [FATAL] node.exe not found at expected location.
  goto error
)

REM ============================================================================
REM == Step 4: Install deps via uv
REM ============================================================================
echo [STEP 4/5] Installing dependencies with uv from pyproject.toml
if not exist "%pyproject%" (
  echo [FATAL] Missing pyproject: "%pyproject%"
  goto error
)

pushd "%root_folder%" >nul
"%uv_exe%" sync --python "%python_exe%"
set "sync_ec=%ERRORLEVEL%"
if not "%sync_ec%"=="0" (
  echo [WARN] uv sync with embeddable Python failed, code %sync_ec%. Falling back to uv-managed Python
  "%uv_exe%" sync
  set "sync_ec=%ERRORLEVEL%"
)
popd >nul
if not "%sync_ec%"=="0" (
  echo [FATAL] uv sync failed with code %sync_ec%.
  goto error
)

> "%env_marker%" echo setup_completed
echo [SUCCESS] Environment setup complete.

REM ============================================================================
REM == Step 5: Prune uv cache
REM ============================================================================
echo [STEP 5/5] Pruning uv cache
if exist "%UV_CACHE_DIR%" rd /s /q "%UV_CACHE_DIR%" || echo [WARN] Could not delete cache dir quickly.

REM ============================================================================
REM == Load env overrides
REM ============================================================================
:load_env
set "FASTAPI_HOST=127.0.0.1"
set "FASTAPI_PORT=8000"
set "UI_HOST=127.0.0.1"
set "UI_PORT=7861"
set "RELOAD=false"

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
) else (
  echo [INFO] No .env overrides found at "%DOTENV%". Using defaults.
)

echo [INFO] FASTAPI_HOST=!FASTAPI_HOST! FASTAPI_PORT=!FASTAPI_PORT! UI_HOST=!UI_HOST! UI_PORT=!UI_PORT! RELOAD=!RELOAD!
set "UI_URL=http://!UI_HOST!:!UI_PORT!"
set "RELOAD_FLAG="
if /i "!RELOAD!"=="true" set "RELOAD_FLAG=--reload"

REM ============================================================================
REM Start backend and frontend
REM ============================================================================
if not exist "%python_exe%" (
  echo [FATAL] python.exe not found at "%python_exe%"
  goto error
)

echo [RUN] Launching backend via uvicorn (!UVICORN_MODULE!)
call :kill_port %FASTAPI_PORT%
start "" /b "%uv_exe%" run --python "%python_exe%" python -m uvicorn %UVICORN_MODULE% --host %FASTAPI_HOST% --port %FASTAPI_PORT% %RELOAD_FLAG% --log-level info

if not exist "%FRONTEND_DIR%\node_modules" (
  echo [STEP] Installing frontend dependencies...
  pushd "%FRONTEND_DIR%" >nul
  call "%NPM_CMD%" install
  set "npm_ec=!ERRORLEVEL!"
  popd >nul
  if not "!npm_ec!"=="0" (
    echo [FATAL] npm install failed with code !npm_ec!.
    goto error
  )
)

if not exist "%FRONTEND_DIST%" (
  echo [STEP] Building frontend
  pushd "%FRONTEND_DIR%" >nul
  call "%NPM_CMD%" run build
  set "npm_build_ec=!ERRORLEVEL!"
  popd >nul
  if not "!npm_build_ec!"=="0" (
    echo [FATAL] Frontend build failed with code !npm_build_ec!.
    goto error
  )
) else (
  echo [INFO] Frontend build already present at "%FRONTEND_DIST%".
)

echo [RUN] Launching frontend
pushd "%FRONTEND_DIR%" >nul
call :kill_port %UI_PORT%
start "" /b "%NPM_CMD%" run preview -- --host %UI_HOST% --port %UI_PORT% --strictPort
popd >nul

start "" "%UI_URL%"
echo [SUCCESS] Backend and frontend correctly launched
goto cleanup

REM ============================================================================
REM Cleanup temp helpers
REM ============================================================================
:cleanup
del /q "%TMPDL%" "%TMPEXP%" "%TMPTXT%" "%TMPFIND%" "%TMPVER%" "%TMPFINDNODE%" >nul 2>&1
endlocal & exit /b 0

REM ============================================================================
REM == Error
REM ============================================================================
:error
echo.
echo !!! An error occurred during execution. !!!
pause
del /q "%TMPDL%" "%TMPEXP%" "%TMPTXT%" "%TMPFIND%" "%TMPVER%" "%TMPFINDNODE%" >nul 2>&1
endlocal & exit /b 1

:kill_port
set "target_port=%~1"
if not defined target_port goto :eof
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R ":!target_port!"') do (
  taskkill /PID %%P /F >nul 2>&1
)
goto :eof
