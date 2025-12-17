@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM == Config
REM ============================================================================
set "project_folder=%~dp0"
set "root_folder=%project_folder%..\" 
set "setup_dir=%project_folder%setup"
set "python_dir=%setup_dir%\python"
set "python_exe=%python_dir%\python.exe"
set "python_pth_file=%python_dir%\python312._pth"
set "env_marker=%python_dir%\.is_installed"

set "uv_dir=%setup_dir%\uv"
set "uv_exe=%uv_dir%\uv.exe"
set "uv_zip_path=%uv_dir%\uv.zip"
set "UV_CACHE_DIR=%setup_dir%\uv_cache"

set "py_version=3.12.10"
set "python_zip_filename=python-%py_version%-embed-amd64.zip"
set "python_zip_url=https://www.python.org/ftp/python/%py_version%/%python_zip_filename%"
set "python_zip_path=%python_dir%\%python_zip_filename%"

REM uv release URLs
set "UV_CHANNEL=latest"
set "UV_ZIP_AMD=https://github.com/astral-sh/uv/releases/%UV_CHANNEL%/download/uv-x86_64-pc-windows-msvc.zip"
set "UV_ZIP_ARM=https://github.com/astral-sh/uv/releases/%UV_CHANNEL%/download/uv-aarch64-pc-windows-msvc.zip"

REM pyproject + app
set "pyproject=%root_folder%pyproject.toml"
set "app_script=%project_folder%app\app.py"

REM Temp PS script locations (no spaces)
set "TMPDL=%TEMP%\app_dl.ps1"
set "TMPEXP=%TEMP%\app_expand.ps1"
set "TMPTXT=%TEMP%\app_txt.ps1"
set "TMPFIND=%TEMP%\app_find_uv.ps1"
set "TMPVER=%TEMP%\app_pyver.ps1"

REM Prefer copy instead of hardlinks to avoid warnings/perf surprises on Windows
set "UV_LINK_MODE=copy"

title App bootstrap (Python + uv)
echo.

REM ============================================================================
REM == Prepare helper PowerShell scripts (argument-driven; no nested quoting)
REM ============================================================================
> "%TMPDL%"  echo $ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri $args[0] -OutFile $args[1]
> "%TMPEXP%" echo $ErrorActionPreference='Stop'; Expand-Archive -LiteralPath $args[0] -DestinationPath $args[1] -Force
> "%TMPTXT%" echo $ErrorActionPreference='Stop'; (Get-Content -LiteralPath $args[0]) -replace '#import site','import site' ^| Set-Content -LiteralPath $args[0]
> "%TMPFIND%" echo $ErrorActionPreference='Stop'; (Get-ChildItem -LiteralPath $args[0] -Recurse -Filter 'uv.exe' ^| Select-Object -First 1).FullName
> "%TMPVER%" echo $ErrorActionPreference='Stop'; ^& $args[0] -c "import platform;print(platform.python_version())"

REM ============================================================================
REM == Fast path
REM ============================================================================
if exist "%env_marker%" if exist "%python_exe%" if exist "%uv_exe%" goto run_app

REM ============================================================================
REM == Step 1: Ensure Python (embeddable)
REM ============================================================================
echo [STEP 1/4] Setting up Python (embeddable) locally
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
echo [STEP 2/4] Installing uv (portable)
if not exist "%uv_dir%" md "%uv_dir%" >nul 2>&1

REM Safe default to AMD64; override only if ARM64
set "uv_zip_url=%UV_ZIP_AMD%"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "uv_zip_url=%UV_ZIP_ARM%"

if not exist "%uv_exe%" (
  echo [DL] %uv_zip_url%
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPDL%" "%uv_zip_url%" "%uv_zip_path%" || goto error
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPEXP%" "%uv_zip_path%" "%uv_dir%" || goto error
  del /q "%uv_zip_path%" >nul 2>&1

  for /f "delims=" %%F in ('powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPFIND%" "%uv_dir%"') do set "found_uv=%%F"
  if not defined found_uv echo [FATAL] uv.exe not found after extraction.& goto error
  if /i not "%found_uv%"=="%uv_exe%" copy /y "%found_uv%" "%uv_exe%" >nul
)

"%uv_exe%" --version >nul 2>&1 && for /f "delims=" %%V in ('"%uv_exe%" --version') do echo %%V

REM ============================================================================
REM == Step 3: Install deps via uv
REM ============================================================================
echo [STEP 3/4] Installing dependencies with uv from pyproject.toml
if not exist "%pyproject%" echo [FATAL] Missing pyproject: "%pyproject%"& goto error

pushd "%root_folder%" >nul
"%uv_exe%" sync --python "%python_exe%"
set "sync_ec=%ERRORLEVEL%"
if not "%sync_ec%"=="0" (
  echo [WARN] uv sync with embeddable Python failed, code %sync_ec%. Falling back to uv-managed Python
  "%uv_exe%" sync
  set "sync_ec=%ERRORLEVEL%"
)
popd >nul
if not "%sync_ec%"=="0" echo [FATAL] uv sync failed with code %sync_ec%.& goto error

> "%env_marker%" echo setup_completed
echo [SUCCESS] Environment setup complete.

REM ============================================================================
REM == Step 4: Prune uv cache
REM ============================================================================
echo [STEP 4/4] Pruning uv cache
if exist "%UV_CACHE_DIR%" rd /s /q "%UV_CACHE_DIR%" || echo [WARN] Could not delete cache dir quickly.

REM ============================================================================
REM == Run app (through uv)
REM ============================================================================
:run_app
echo [RUN] Launching application
if not exist "%app_script%" echo [FATAL] Application script not found: "%app_script%"& goto error

pushd "%root_folder%" >nul
"%uv_exe%" run --python "%python_exe%" python "%app_script%"
set "run_ec=%ERRORLEVEL%"
popd >nul

if "%run_ec%"=="0" (
  echo [SUCCESS] Application launched successfully.
  goto cleanup
) else (
  echo [FATAL] Application exited with code %run_ec%.
  goto error
)

REM ============================================================================
REM == Cleanup temp helpers
REM ============================================================================
:cleanup
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