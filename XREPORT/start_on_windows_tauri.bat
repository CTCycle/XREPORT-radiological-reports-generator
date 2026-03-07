@echo off
setlocal enabledelayedexpansion

if /i "%~1"=="--backend" goto backend_mode

set "project_folder=%~dp0"
set "client_dir=%project_folder%client"
set "tauri_dir=%client_dir%\src-tauri"
set "settings_dir=%project_folder%settings"
set "env_file=%settings_dir%\.env"
set "env_template=%settings_dir%\.env.local.tauri.example"

set "release_exe=%tauri_dir%\target\release\xreport-desktop.exe"
set "release_exe_alt=%tauri_dir%\target\release\XREPORT Desktop.exe"
set "debug_exe=%tauri_dir%\target\debug\xreport-desktop.exe"
set "debug_exe_alt=%tauri_dir%\target\debug\XREPORT Desktop.exe"
set "launcher_marker=%tauri_dir%\target\.xreport_tauri_single_launcher_v1"

echo [TAURI] One-click launcher

if not exist "%env_file%" if exist "%env_template%" (
  copy /Y "%env_template%" "%env_file%" >nul
  echo [INFO] Created "%env_file%" from local Tauri template.
)

if exist "%launcher_marker%" (
  call :launch_if_exists "%release_exe%" && goto launcher_success
  call :launch_if_exists "%release_exe_alt%" && goto launcher_success
  call :launch_if_exists "%debug_exe%" && goto launcher_success
  call :launch_if_exists "%debug_exe_alt%" && goto launcher_success
  echo [INFO] Marker found but executable missing. Rebuilding...
) else (
  echo [INFO] First run after launcher update. Building desktop executable...
)

where cargo >nul 2>&1
if errorlevel 1 (
  echo [FATAL] Rust/Cargo not found. Install Rust first: https://rustup.rs/
  goto launcher_error
)

set "npm_cmd="
where npm >nul 2>&1
if not errorlevel 1 set "npm_cmd=npm"
if not defined npm_cmd if exist "%project_folder%resources\runtimes\nodejs\npm.cmd" (
  set "npm_cmd=%project_folder%resources\runtimes\nodejs\npm.cmd"
)
if not defined npm_cmd (
  echo [FATAL] npm not found. Install Node.js or run XREPORT\start_on_windows.bat once.
  goto launcher_error
)

if not exist "%client_dir%\package.json" (
  echo [FATAL] Missing client package.json at "%client_dir%"
  goto launcher_error
)

pushd "%client_dir%" >nul
if exist "package-lock.json" (
  call "%npm_cmd%" ci
) else (
  call "%npm_cmd%" install
)
if errorlevel 1 (
  popd >nul
  echo [FATAL] npm dependency installation failed.
  goto launcher_error
)

call "%npm_cmd%" run tauri:build
if errorlevel 1 (
  popd >nul
  echo [FATAL] Tauri build failed.
  goto launcher_error
)
popd >nul
> "%launcher_marker%" echo single_launcher_v1

call :launch_if_exists "%release_exe%" && goto launcher_success
call :launch_if_exists "%release_exe_alt%" && goto launcher_success
call :launch_if_exists "%debug_exe%" && goto launcher_success
call :launch_if_exists "%debug_exe_alt%" && goto launcher_success

echo [FATAL] Build completed but no Tauri executable was found.
goto launcher_error

:launch_if_exists
set "candidate=%~1"
if not exist "%candidate%" exit /b 1
echo [RUN] Launching "%candidate%"
start "" "%candidate%"
exit /b 0

:launcher_success
echo [OK] Desktop app started.
endlocal & exit /b 0

:launcher_error
echo.
echo Press any key to close this launcher...
pause >nul
endlocal & exit /b 1
:backend_mode
set "project_folder=%~dp0"
set "root_folder=%project_folder%..\"
set "runtimes_dir=%project_folder%resources\runtimes"
set "settings_dir=%project_folder%settings"
set "python_dir=%runtimes_dir%\python"
set "python_exe=%python_dir%\python.exe"
set "python_pth_file=%python_dir%\python314._pth"

set "uv_dir=%runtimes_dir%\uv"
set "uv_exe=%uv_dir%\uv.exe"
set "uv_zip_path=%uv_dir%\uv.zip"
set "UV_CACHE_DIR=%runtimes_dir%\uv_cache"

set "py_version=3.14.2"
set "python_zip_filename=python-%py_version%-embed-amd64.zip"
set "python_zip_url=https://www.python.org/ftp/python/%py_version%/%python_zip_filename%"
set "python_zip_path=%python_dir%\%python_zip_filename%"

set "UV_CHANNEL=latest"
set "UV_ZIP_AMD=https://github.com/astral-sh/uv/releases/%UV_CHANNEL%/download/uv-x86_64-pc-windows-msvc.zip"
set "UV_ZIP_ARM=https://github.com/astral-sh/uv/releases/%UV_CHANNEL%/download/uv-aarch64-pc-windows-msvc.zip"

set "pyproject=%root_folder%pyproject.toml"
set "UVICORN_MODULE=XREPORT.server.app:app"
set "DOTENV=%settings_dir%\.env"

set "TMPDL=%TEMP%\app_tauri_dl.ps1"
set "TMPEXP=%TEMP%\app_tauri_expand.ps1"
set "TMPTXT=%TEMP%\app_tauri_txt.ps1"
set "TMPFIND=%TEMP%\app_tauri_find_uv.ps1"
set "TMPVER=%TEMP%\app_tauri_pyver.ps1"

set "UV_LINK_MODE=copy"

if not exist "%runtimes_dir%" md "%runtimes_dir%" >nul 2>&1
if not exist "%python_dir%" md "%python_dir%" >nul 2>&1
if not exist "%uv_dir%" md "%uv_dir%" >nul 2>&1

echo $ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri $args[0] -OutFile $args[1] > "%TMPDL%"
echo $ErrorActionPreference='Stop'; Expand-Archive -LiteralPath $args[0] -DestinationPath $args[1] -Force > "%TMPEXP%"
echo $ErrorActionPreference='Stop'; (Get-Content -LiteralPath $args[0]) -replace '#import site','import site' ^| Set-Content -LiteralPath $args[0] > "%TMPTXT%"
echo $ErrorActionPreference='Stop'; (Get-ChildItem -LiteralPath $args[0] -Recurse -Filter 'uv.exe' ^| Select-Object -First 1).FullName > "%TMPFIND%"
echo $ErrorActionPreference='Stop'; ^& $args[0] -c "import platform;print(platform.python_version())" > "%TMPVER%"

echo [TAURI 1/3] Setting up Python (embeddable)

if not exist "%python_exe%" (
  echo [DL] %python_zip_url%
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPDL%" "%python_zip_url%" "%python_zip_path%" || goto backend_error
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPEXP%" "%python_zip_path%" "%python_dir%" || goto backend_error
  del /q "%python_zip_path%" >nul 2>&1
)

if exist "%python_pth_file%" (
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPTXT%" "%python_pth_file%" || goto backend_error
)

for /f "delims=" %%V in ('powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPVER%" "%python_exe%"') do set "found_py=%%V"
echo [OK] Python ready: !found_py!

echo [TAURI 2/3] Installing uv (portable)

set "uv_zip_url=%UV_ZIP_AMD%"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "uv_zip_url=%UV_ZIP_ARM%"

if not exist "%uv_exe%" (
  echo [DL] %uv_zip_url%
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPDL%" "%uv_zip_url%" "%uv_zip_path%" || goto backend_error
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPEXP%" "%uv_zip_path%" "%uv_dir%" || goto backend_error
  del /q "%uv_zip_path%" >nul 2>&1

  for /f "delims=" %%F in ('powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPFIND%" "%uv_dir%"') do set "found_uv=%%F"
  if not defined found_uv (
    echo [FATAL] uv.exe not found after extraction.
    goto backend_error
  )
  if /i not "%found_uv%"=="%uv_exe%" copy /y "%found_uv%" "%uv_exe%" >nul
)

"%uv_exe%" --version >nul 2>&1 && for /f "delims=" %%V in ('"%uv_exe%" --version') do echo %%V

set "FASTAPI_HOST=127.0.0.1"
set "FASTAPI_PORT=8000"
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
  echo [INFO] No .env file found at "%DOTENV%". Using defaults.
)

set "INSTALL_EXTRAS=false"
if /i "!OPTIONAL_DEPENDENCIES!"=="true" set "INSTALL_EXTRAS=true"

set "RELOAD_FLAG="
if /i "!RELOAD!"=="true" set "RELOAD_FLAG=--reload"

echo [INFO] FASTAPI_HOST=!FASTAPI_HOST! FASTAPI_PORT=!FASTAPI_PORT! RELOAD=!RELOAD!

set "PYTHONHOME=%python_dir%"
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "XREPORT_TAURI_MODE=true"

echo [TAURI 3/3] Installing dependencies and starting backend
if not exist "%pyproject%" (
  echo [FATAL] Missing pyproject: "%pyproject%"
  goto backend_error
)

pushd "%root_folder%" >nul
set "uv_extras_flag="
if /i "%INSTALL_EXTRAS%"=="true" set "uv_extras_flag=--all-extras"
set "use_uv_managed_python=false"
"%uv_exe%" sync --python "%python_exe%" %uv_extras_flag%
set "sync_ec=%ERRORLEVEL%"
if not "%sync_ec%"=="0" (
  echo [WARN] uv sync with embeddable Python failed, code %sync_ec%. Falling back to uv-managed Python
  "%uv_exe%" sync %uv_extras_flag%
  set "sync_ec=%ERRORLEVEL%"
  if "%sync_ec%"=="0" set "use_uv_managed_python=true"
)
if not "%sync_ec%"=="0" (
  popd >nul
  echo [FATAL] uv sync failed with code %sync_ec%.
  goto backend_error
)

echo [RUN] Launching backend via uvicorn (%UVICORN_MODULE%)
if /i "%use_uv_managed_python%"=="true" (
  "%uv_exe%" run python -m uvicorn %UVICORN_MODULE% --host !FASTAPI_HOST! --port !FASTAPI_PORT! !RELOAD_FLAG! --log-level info
) else (
  "%uv_exe%" run --python "%python_exe%" python -m uvicorn %UVICORN_MODULE% --host !FASTAPI_HOST! --port !FASTAPI_PORT! !RELOAD_FLAG! --log-level info
)
set "backend_ec=%ERRORLEVEL%"
popd >nul

if exist "%UV_CACHE_DIR%" rd /s /q "%UV_CACHE_DIR%" >nul 2>&1
if "%backend_ec%"=="0" goto backend_cleanup

echo [ERROR] Backend exited with code %backend_ec%.
goto backend_error

:backend_cleanup
del /q "%TMPDL%" "%TMPEXP%" "%TMPTXT%" "%TMPFIND%" "%TMPVER%" >nul 2>&1
endlocal & exit /b 0

:backend_error
del /q "%TMPDL%" "%TMPEXP%" "%TMPTXT%" "%TMPFIND%" "%TMPVER%" >nul 2>&1
endlocal & exit /b 1

