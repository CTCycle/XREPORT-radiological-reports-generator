@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM == Configuration
REM ============================================================================
set "project_folder=%~dp0"
set "root_folder=%project_folder%..\"
set "runtimes_dir=%root_folder%runtimes"
set "settings_dir=%project_folder%settings"
set "python_dir=%runtimes_dir%\python"
set "python_exe=%python_dir%\python.exe"
set "python_pth_file=%python_dir%\python314._pth"
set "env_marker=%python_dir%\.is_installed"

set "uv_dir=%runtimes_dir%\uv"
set "uv_exe=%uv_dir%\uv.exe"
set "uv_zip_path=%uv_dir%\uv.zip"
set "UV_CACHE_DIR=%runtimes_dir%\.uv-cache"
set "venv_dir=%runtimes_dir%\.venv"
set "UV_PROJECT_ENVIRONMENT=%venv_dir%"
set "runtime_uv_lock=%root_folder%runtimes\uv.lock"
set "uv_lock_file=%root_folder%uv.lock"

set "py_version=3.14.2"
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
set "UVICORN_MODULE=XREPORT.server.app:app"
set "FRONTEND_DIR=%project_folder%client"
set "FRONTEND_DIST=%FRONTEND_DIR%\dist"
set "FRONTEND_LOCKFILE=%FRONTEND_DIR%\package-lock.json"
set "BACKEND_BOOT_LOG=%project_folder%resources\logs\backend_boot.log"

set "DOTENV=%settings_dir%\.env"
set "TMPDL=%TEMP%\app_dl.ps1"
set "TMPEXP=%TEMP%\app_expand.ps1"
set "TMPTXT=%TEMP%\app_txt.ps1"
set "TMPFIND=%TEMP%\app_find_uv.ps1"
set "TMPVER=%TEMP%\app_pyver.ps1"

set "UV_LINK_MODE=copy"

title XREPORT Launcher
echo.

set "NPM_CMD=%npm_cmd%"
set "NODE_CMD=%node_exe%"

REM ============================================================================
REM == Ensure runtime directories exist
REM ============================================================================
if not exist "%runtimes_dir%" md "%runtimes_dir%" >nul 2>&1
if not exist "%python_dir%" md "%python_dir%" >nul 2>&1
if not exist "%nodejs_dir%" md "%nodejs_dir%" >nul 2>&1

REM ============================================================================
REM == Prepare helper PowerShell scripts
REM ============================================================================
echo $ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri $args[0] -OutFile $args[1] > "%TMPDL%"
echo $ErrorActionPreference='Stop'; Expand-Archive -LiteralPath $args[0] -DestinationPath $args[1] -Force > "%TMPEXP%"
echo $ErrorActionPreference='Stop'; (Get-Content -LiteralPath $args[0]) -replace '#import site','import site' ^| Set-Content -LiteralPath $args[0] > "%TMPTXT%"
echo $ErrorActionPreference='Stop'; (Get-ChildItem -LiteralPath $args[0] -Recurse -Filter 'uv.exe' ^| Select-Object -First 1).FullName > "%TMPFIND%"
echo $ErrorActionPreference='Stop'; ^& $args[0] -c "import platform;print(platform.python_version())" > "%TMPVER%"

REM ============================================================================
REM == Step 1: Ensure Python (embeddable)
REM ============================================================================
echo [STEP 1/5] Setting up Python (embeddable) locally

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

if not exist "%node_exe%" (
  echo [DL] %nodejs_zip_url%
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPDL%" "%nodejs_zip_url%" "%nodejs_zip_path%" || goto error
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPEXP%" "%nodejs_zip_path%" "%nodejs_dir%" || goto error
  del /q "%nodejs_zip_path%" >nul 2>&1
)

set "node_archive_dir=%nodejs_dir%\node-v%nodejs_version%-win-x64"
if exist "%node_archive_dir%\node.exe" (
  call :promote_node_runtime "%node_archive_dir%"
  if errorlevel 1 goto error
)

if exist "%node_exe%" (
  for /f "delims=" %%V in ('"%node_exe%" --version') do echo [OK] Node.js ready: %%V
  set "NPM_CMD=%npm_cmd%"
  set "NODE_CMD=%node_exe%"
  set "PATH=%nodejs_dir%;%PATH%"

) else (
  echo [FATAL] node.exe not found in "%nodejs_dir%".
  echo [INFO] Expected file: "%node_exe%"
  goto error
)

REM ============================================================================
REM == Load env overrides (needed before dependency install)
REM ============================================================================
:load_env
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
      for /f "tokens=1,* delims==" %%A in ("!line!") do (
        set "k=%%A"
        set "v=%%B"
        if defined v (
          for /f "tokens=* delims= " %%Q in ("!v!") do set "v=%%Q"
          set "v=!v:"=!"
          if "!v:~0,1!"=="'" if "!v:~-1!"=="'" set "v=!v:~1,-1!"
        )
        set "!k!=!v!"
      )
    )
  )
) else (
  echo [INFO] No .env overrides found at "%DOTENV%". Using defaults.
)

set "INSTALL_EXTRAS=false"
if /i "!OPTIONAL_DEPENDENCIES!"=="true" set "INSTALL_EXTRAS=true"

echo [INFO] FASTAPI_HOST=!FASTAPI_HOST! FASTAPI_PORT=!FASTAPI_PORT! UI_HOST=!UI_HOST! UI_PORT=!UI_PORT! RELOAD=!RELOAD!
set "UI_URL=http://!UI_HOST!:!UI_PORT!"
set "RELOAD_FLAG="
if /i "!RELOAD!"=="true" set "RELOAD_FLAG=--reload"

REM Ensure the embeddable runtime is used (avoid picking up Conda/other Python DLLs)
set "PYTHONHOME=%python_dir%"
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"

REM ============================================================================
REM == Step 4: Install deps via uv
REM ============================================================================
echo [STEP 4/5] Installing dependencies with uv from pyproject.toml
if not exist "%pyproject%" (
  echo [FATAL] Missing pyproject: "%pyproject%"
  goto error
)

pushd "%root_folder%" >nul
if exist "%runtime_uv_lock%" (
  echo [INFO] Using runtime lockfile from "%runtime_uv_lock%".
  copy /y "%runtime_uv_lock%" "%uv_lock_file%" >nul
  if errorlevel 1 (
    echo [WARN] Failed to copy runtime lockfile to "%uv_lock_file%". Continuing with existing root lockfile.
  ) else (
    echo [INFO] Runtime lockfile staged at "%uv_lock_file%".
  )
) else (
  echo [INFO] Runtime lockfile not found at "%runtime_uv_lock%". uv sync will use the current root lockfile state.
)
set "uv_extras_flag="
if /i "%INSTALL_EXTRAS%"=="true" set "uv_extras_flag=--all-extras"
"%uv_exe%" sync --python "%python_exe%" %uv_extras_flag%
set "sync_ec=%ERRORLEVEL%"
if not "%sync_ec%"=="0" (
  echo [WARN] uv sync with embeddable Python failed, code %sync_ec%. Falling back to uv-managed Python
  "%uv_exe%" sync %uv_extras_flag%
  set "sync_ec=%ERRORLEVEL%"
)
if "%sync_ec%"=="0" (
  if exist "%uv_lock_file%" (
    copy /y "%uv_lock_file%" "%runtime_uv_lock%" >nul
    if errorlevel 1 (
      echo [WARN] uv sync succeeded but failed to update runtime lockfile at "%runtime_uv_lock%".
    ) else (
      echo [INFO] Updated runtime lockfile at "%runtime_uv_lock%".
    )
  ) else (
    echo [WARN] uv sync succeeded but "%uv_lock_file%" is missing, so runtime lockfile was not updated.
  )
)
popd >nul
if exist "%uv_lock_file%" (
  del /q "%uv_lock_file%" >nul 2>&1
  if exist "%uv_lock_file%" (
    echo [WARN] Could not remove temporary root lockfile "%uv_lock_file%".
  ) else (
    echo [INFO] Removed temporary root lockfile "%uv_lock_file%".
  )
)
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

if not exist "%FRONTEND_DIR%\node_modules" (
  echo [STEP] Installing frontend dependencies...
  pushd "%FRONTEND_DIR%" >nul
  if exist "%FRONTEND_LOCKFILE%" (
    call "%NPM_CMD%" ci
  ) else (
    call "%NPM_CMD%" install
  )
  set "npm_ec=!ERRORLEVEL!"
  popd >nul
  if not "!npm_ec!"=="0" (
    echo [FATAL] Frontend dependency install failed with code !npm_ec!.
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

REM ============================================================================
REM Start backend and frontend
REM ============================================================================
if not exist "%python_exe%" (
  echo [FATAL] python.exe not found at "%python_exe%"
  goto error
)

echo [RUN] Launching backend via uvicorn (!UVICORN_MODULE!)
call :kill_port !FASTAPI_PORT!
call :wait_port_free !FASTAPI_PORT! 10
if errorlevel 1 (
  echo [FATAL] Port !FASTAPI_PORT! is still busy after cleanup. Backend cannot start.
  goto error
)
if exist "%BACKEND_BOOT_LOG%" del /q "%BACKEND_BOOT_LOG%" >nul 2>&1
start "" /b "%uv_exe%" run --no-sync --python "%python_exe%" python -m uvicorn %UVICORN_MODULE% --host !FASTAPI_HOST! --port !FASTAPI_PORT! !RELOAD_FLAG! --log-level info > "%BACKEND_BOOT_LOG%" 2>&1

REM ============================================================================
REM Wait for backend
REM ============================================================================
set "BACKEND_BASE_URL=http://!FASTAPI_HOST!:!FASTAPI_PORT!"
echo [WAIT] Waiting for backend readiness at !BACKEND_BASE_URL!...
for /L %%i in (1,1,60) do (
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$base='!BACKEND_BASE_URL!'; $paths=@('/api/health','/health','/docs','/'); foreach ($p in $paths) { try { $r = Invoke-WebRequest -UseBasicParsing -Uri ($base + $p) -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300) { exit 0 } } catch {} }; exit 1" >nul 2>&1
  if !errorlevel! equ 0 goto :backend_ready_check
  timeout /t 1 /nobreak >nul 2>&1
)
echo [FATAL] Backend did not become ready at !BACKEND_BASE_URL! (checked /api/health, /health, /docs, /).
goto error
:backend_ready_check

echo [RUN] Launching frontendpushd "%FRONTEND_DIR%" >nul
call :kill_port !UI_PORT!
call :wait_port_free !UI_PORT! 10
if errorlevel 1 (
  popd >nul
  echo [FATAL] Port !UI_PORT! is still busy after cleanup. Frontend cannot start.
  goto error
)
start "" /b "%NPM_CMD%" run preview -- --host !UI_HOST! --port !UI_PORT! --strictPort
popd >nul

start "" "%UI_URL%"
echo [SUCCESS] Backend and frontend correctly launched
goto cleanup

:promote_node_runtime
set "node_source_dir=%~1"
if not defined node_source_dir exit /b 1
for %%D in ("%~1") do set "node_source_dir=%%~fD"
if /i "!node_source_dir!"=="%nodejs_dir%" exit /b 0

robocopy "!node_source_dir!" "%nodejs_dir%" /MOVE /E /R:2 /W:1 /NFL /NDL /NJH /NJS /NC /NS >nul
set "node_move_ec=!ERRORLEVEL!"
if !node_move_ec! geq 8 (
  echo [FATAL] Failed to flatten portable Node.js runtime from "!node_source_dir!".
  exit /b !node_move_ec!
)

if exist "!node_source_dir!" rd /s /q "!node_source_dir!" >nul 2>&1
exit /b 0

REM ============================================================================
REM Cleanup temp helpers
REM ============================================================================
:cleanup
if exist "%uv_lock_file%" del /q "%uv_lock_file%" >nul 2>&1
del /q "%TMPDL%" "%TMPEXP%" "%TMPTXT%" "%TMPFIND%" "%TMPVER%" >nul 2>&1
endlocal & exit /b 0

REM ============================================================================
REM == Error
REM ============================================================================
:error
echo.
echo !!! An error occurred during execution. !!!
pause
del /q "%TMPDL%" "%TMPEXP%" "%TMPTXT%" "%TMPFIND%" "%TMPVER%" >nul 2>&1
endlocal & exit /b 1

:kill_port
set "target_port=%~1"
if not defined target_port goto :eof
for /f "usebackq delims=" %%P in (`powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$port=%target_port%; Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique"`) do (
  if not "%%P"=="0" taskkill /PID %%P /F >nul 2>&1
)
goto :eof

:wait_port_free
set "target_port=%~1"
set "wait_seconds=%~2"
if not defined wait_seconds set "wait_seconds=10"
for /L %%W in (1,1,!wait_seconds!) do (
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$port=%target_port%; $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue; if ($null -eq $conn) { exit 0 } else { exit 1 }" >nul 2>&1
  if !errorlevel! equ 0 exit /b 0
  timeout /t 1 /nobreak >nul
)
exit /b 1
