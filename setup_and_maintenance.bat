@echo off
setlocal EnableDelayedExpansion

for %%I in ("%~dp0.") do set "repo_root=%%~fI"
set "app_dir=%repo_root%\app"
set "server_dir=%app_dir%\server"
set "client_dir=%app_dir%\client"
set "scripts_dir=%app_dir%\scripts"
set "tests_dir=%app_dir%\tests"
set "log_dir=%app_dir%\resources\logs"
set "settings_dir=%repo_root%\settings"
set "runtimes_dir=%repo_root%\runtimes"
set "python_dir=%runtimes_dir%\python"
set "uv_dir=%runtimes_dir%\uv"
set "nodejs_dir=%runtimes_dir%\nodejs"

set "python_exe=%python_dir%\python.exe"
set "python_pth_file=%python_dir%\python314._pth"
set "uv_exe=%uv_dir%\uv.exe"
set "node_exe=%nodejs_dir%\node.exe"
set "npm_cmd=%nodejs_dir%\npm.cmd"
set "init_db_script=%scripts_dir%\initialize_database.py"
set "tauri_clean_script=%repo_root%\release\tauri\scripts\clean-tauri-build.ps1"
set "pyproject=%server_dir%\pyproject.toml"
set "frontend_dist=%client_dir%\dist"
set "frontend_lockfile=%client_dir%\package-lock.json"
set "uv_cache_dir=%runtimes_dir%\.uv-cache"
set "dotenv=%settings_dir%\.env"
set "PYTHONPATH=%app_dir%"

set "py_version=3.14.2"
set "python_zip_filename=python-%py_version%-embed-amd64.zip"
set "python_zip_url=https://www.python.org/ftp/python/%py_version%/%python_zip_filename%"
set "python_zip_path=%python_dir%\%python_zip_filename%"

set "UV_CHANNEL=latest"
set "UV_ZIP_AMD=https://github.com/astral-sh/uv/releases/%UV_CHANNEL%/download/uv-x86_64-pc-windows-msvc.zip"
set "UV_ZIP_ARM=https://github.com/astral-sh/uv/releases/%UV_CHANNEL%/download/uv-aarch64-pc-windows-msvc.zip"
set "uv_zip_path=%uv_dir%\uv.zip"

set "nodejs_version=22.12.0"
set "nodejs_zip_filename=node-v%nodejs_version%-win-x64.zip"
set "nodejs_zip_url=https://nodejs.org/dist/v%nodejs_version%/%nodejs_zip_filename%"
set "nodejs_zip_path=%nodejs_dir%\%nodejs_zip_filename%"

set "TMPDL=%TEMP%\setup_dl.ps1"
set "TMPEXP=%TEMP%\setup_expand.ps1"
set "TMPTXT=%TEMP%\setup_txt.ps1"
set "TMPFIND=%TEMP%\setup_find_uv.ps1"
set "TMPVER=%TEMP%\setup_pyver.ps1"

:menu
cls
echo ==========================================================================
echo                         Setup and Maintenance
echo ==========================================================================
echo 1. Install application
echo 2. Initialize database
echo 3. Uninstall application
echo 4. Remove desktop packages
echo 5. Run test suite
echo 6. Remove logs
echo 7. Exit
echo.
set /p sub_choice="Select an option (1-7): "
set "sub_choice=%sub_choice: =%"

if "%sub_choice%"=="1" goto :install_app
if "%sub_choice%"=="2" goto :run_init_db
if "%sub_choice%"=="3" goto :uninstall
if "%sub_choice%"=="4" goto :remove_desktop
if "%sub_choice%"=="5" goto :run_tests
if "%sub_choice%"=="6" goto :remove_logs
if "%sub_choice%"=="7" goto :exit

echo [ERROR] Invalid option.
pause
goto :menu

:install_app
if not exist "%runtimes_dir%" md "%runtimes_dir%" >nul 2>&1
if not exist "%python_dir%" md "%python_dir%" >nul 2>&1
if not exist "%uv_dir%" md "%uv_dir%" >nul 2>&1
if not exist "%nodejs_dir%" md "%nodejs_dir%" >nul 2>&1

echo $ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri $args[0] -OutFile $args[1] > "%TMPDL%"
echo $ErrorActionPreference='Stop'; Expand-Archive -LiteralPath $args[0] -DestinationPath $args[1] -Force > "%TMPEXP%"
echo $ErrorActionPreference='Stop'; (Get-Content -LiteralPath $args[0]) -replace '#import site','import site' ^| Set-Content -LiteralPath $args[0] > "%TMPTXT%"
echo $ErrorActionPreference='Stop'; (Get-ChildItem -LiteralPath $args[0] -Recurse -Filter 'uv.exe' ^| Select-Object -First 1).FullName > "%TMPFIND%"
echo $ErrorActionPreference='Stop'; ^& $args[0] -c "import platform;print(platform.python_version())" > "%TMPVER%"

echo [STEP 1/5] Setting up Python (embeddable) locally
if not exist "%python_exe%" (
  echo [DL] %python_zip_url%
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPDL%" "%python_zip_url%" "%python_zip_path%" || goto install_error
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPEXP%" "%python_zip_path%" "%python_dir%" || goto install_error
  del /q "%python_zip_path%" >nul 2>&1
)
if exist "%python_pth_file%" (
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPTXT%" "%python_pth_file%" || goto install_error
)
for /f "delims=" %%V in ('powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPVER%" "%python_exe%"') do set "found_py=%%V"
echo [OK] Python ready: !found_py!

echo [STEP 2/5] Installing uv (portable)
set "uv_zip_url=%UV_ZIP_AMD%"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "uv_zip_url=%UV_ZIP_ARM%"
if not exist "%uv_exe%" (
  echo [DL] %uv_zip_url%
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPDL%" "%uv_zip_url%" "%uv_zip_path%" || goto install_error
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPEXP%" "%uv_zip_path%" "%uv_dir%" || goto install_error
  del /q "%uv_zip_path%" >nul 2>&1
  for /f "delims=" %%F in ('powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPFIND%" "%uv_dir%"') do set "found_uv=%%F"
  if not defined found_uv (
    echo [FATAL] uv.exe not found after extraction.
    goto install_error
  )
  if /i not "!found_uv!"=="%uv_exe%" copy /y "!found_uv!" "%uv_exe%" >nul
)
"%uv_exe%" --version >nul 2>&1 && for /f "delims=" %%V in ('"%uv_exe%" --version') do echo %%V

echo [STEP 3/5] Installing Node.js (portable)
if not exist "%node_exe%" (
  echo [DL] %nodejs_zip_url%
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPDL%" "%nodejs_zip_url%" "%nodejs_zip_path%" || goto install_error
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TMPEXP%" "%nodejs_zip_path%" "%nodejs_dir%" || goto install_error
  del /q "%nodejs_zip_path%" >nul 2>&1
)
set "node_archive_dir=%nodejs_dir%\node-v%nodejs_version%-win-x64"
if exist "%node_archive_dir%\node.exe" (
  call :promote_node_runtime "%node_archive_dir%"
  if errorlevel 1 goto install_error
)
if not exist "%node_exe%" (
  echo [FATAL] node.exe not found in "%nodejs_dir%".
  goto install_error
)
if not exist "%npm_cmd%" (
  echo [FATAL] npm.cmd not found in "%nodejs_dir%".
  goto install_error
)
for /f "delims=" %%V in ('"%node_exe%" --version') do echo [OK] Node.js ready: %%V

set "OPTIONAL_DEPENDENCIES=false"
if exist "%dotenv%" (
  for /f "usebackq tokens=* delims=" %%L in ("%dotenv%") do (
    set "line=%%L"
    if not "!line!"=="" if "!line:~0,1!" NEQ "#" if "!line:~0,1!" NEQ ";" (
      for /f "tokens=1,* delims==" %%A in ("!line!") do (
        set "k=%%A"
        set "v=%%B"
        if /i "!k!"=="OPTIONAL_DEPENDENCIES" set "OPTIONAL_DEPENDENCIES=!v!"
      )
    )
  )
)

set "PYTHONHOME=%python_dir%"
set "PYTHONNOUSERSITE=1"

echo [STEP 4/5] Installing dependencies with uv from pyproject.toml
if not exist "%pyproject%" (
  echo [FATAL] Missing pyproject: "%pyproject%"
  goto install_error
)
set "uv_extras_flag="
if /i "%OPTIONAL_DEPENDENCIES%"=="true" set "uv_extras_flag=--all-extras"
pushd "%server_dir%" >nul
"%uv_exe%" sync --python "%python_exe%" %uv_extras_flag%
set "sync_ec=%ERRORLEVEL%"
if not "%sync_ec%"=="0" (
  "%uv_exe%" sync %uv_extras_flag%
  set "sync_ec=%ERRORLEVEL%"
)
popd >nul
if not "%sync_ec%"=="0" (
  echo [FATAL] uv sync failed with code %sync_ec%.
  goto install_error
)

echo [STEP 5/5] Pruning uv cache
if exist "%uv_cache_dir%" rd /s /q "%uv_cache_dir%" >nul 2>&1

if not exist "%client_dir%\node_modules" (
  echo [STEP] Installing frontend dependencies...
  pushd "%client_dir%" >nul
  if exist "%frontend_lockfile%" (
    call "%npm_cmd%" ci
  ) else (
    call "%npm_cmd%" install
  )
  set "npm_ec=%ERRORLEVEL%"
  popd >nul
  if not "!npm_ec!"=="0" (
    echo [FATAL] Frontend dependency install failed with code !npm_ec!.
    goto install_error
  )
)

if not exist "%frontend_dist%" (
  echo [STEP] Building frontend
  pushd "%client_dir%" >nul
  call "%npm_cmd%" run build
  set "npm_build_ec=%ERRORLEVEL%"
  popd >nul
  if not "!npm_build_ec!"=="0" (
    echo [FATAL] Frontend build failed with code !npm_build_ec!.
    goto install_error
  )
) else (
  echo [INFO] Frontend build already present at "%frontend_dist%".
)

echo [SUCCESS] Installation completed. Application not launched.
goto install_cleanup_ok

:install_error
echo [ERROR] Installation failed.

:install_cleanup_ok
del /q "%TMPDL%" "%TMPEXP%" "%TMPTXT%" "%TMPFIND%" "%TMPVER%" >nul 2>&1
pause
goto :menu

:run_init_db
if not exist "%init_db_script%" (
  echo [ERROR] Missing database script: "%init_db_script%".
  pause
  goto :menu
)
if not exist "%uv_exe%" (
  echo [ERROR] Missing uv runtime: "%uv_exe%".
  pause
  goto :menu
)
if not exist "%python_exe%" (
  echo [ERROR] Missing python runtime: "%python_exe%".
  pause
  goto :menu
)
if not exist "%server_dir%\pyproject.toml" (
  echo [ERROR] Missing server project file: "%server_dir%\pyproject.toml".
  pause
  goto :menu
)

pushd "%server_dir%" >nul
"%uv_exe%" run --project "%server_dir%" --python "%python_exe%" python "%init_db_script%"
set "run_ec=%ERRORLEVEL%"
popd >nul
if "%run_ec%"=="0" (
  echo [SUCCESS] Database initialization completed.
) else (
  echo [ERROR] Database initialization failed with exit code %run_ec%.
)
pause
goto :menu

:uninstall
echo --------------------------------------------------------------------------
echo [UNINSTALL] Starting uninstall procedure...

echo [UNINSTALL] Removing lockfiles...
if exist "%repo_root%\app\server\uv.lock" del /q "%repo_root%\app\server\uv.lock"
if exist "%repo_root%\uv.lock" del /q "%repo_root%\uv.lock"

echo [UNINSTALL] Removing runtime directories...
if exist "%runtimes_dir%\uv" rd /s /q "%runtimes_dir%\uv"
if exist "%runtimes_dir%\.uv-cache" rd /s /q "%runtimes_dir%\.uv-cache"
if exist "%runtimes_dir%\uv_cache" rd /s /q "%runtimes_dir%\uv_cache"
if exist "%runtimes_dir%\python" rd /s /q "%runtimes_dir%\python"
if exist "%runtimes_dir%\nodejs" rd /s /q "%runtimes_dir%\nodejs"

echo [UNINSTALL] Removing setup runtime artifacts...
if exist "%repo_root%\setup\uv" rd /s /q "%repo_root%\setup\uv"
if exist "%repo_root%\setup\uv_cache" rd /s /q "%repo_root%\setup\uv_cache"
if exist "%repo_root%\setup\python" (
  for /f "delims=" %%F in ('dir /b "%repo_root%\setup\python"') do (
    if /I not "%%F"==".gitkeep" (
      if exist "%repo_root%\setup\python\%%F\" (
        rd /s /q "%repo_root%\setup\python\%%F"
      ) else (
        del /q "%repo_root%\setup\python\%%F"
      )
    )
  )
)

echo [UNINSTALL] Removing virtual environments...
if exist "%repo_root%\app\server\.venv" rd /s /q "%repo_root%\app\server\.venv"
if exist "%repo_root%\.venv" rd /s /q "%repo_root%\.venv"

echo [UNINSTALL] Removing frontend artifacts...
if exist "%client_dir%\node_modules" rd /s /q "%client_dir%\node_modules"
if exist "%client_dir%\.angular" rd /s /q "%client_dir%\.angular"
if exist "%client_dir%\dist" rd /s /q "%client_dir%\dist"
if exist "%client_dir%\package-lock.json" del /q "%client_dir%\package-lock.json"

echo [UNINSTALL] Completed.
pause
goto :menu

:remove_desktop
if exist "%tauri_clean_script%" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%tauri_clean_script%"
) else (
  if exist "%client_dir%\src-tauri\target\release" rd /s /q "%client_dir%\src-tauri\target\release"
  if exist "%client_dir%\src-tauri\target" rd /s /q "%client_dir%\src-tauri\target"
  if exist "%repo_root%\release\windows" rd /s /q "%repo_root%\release\windows"
  for /r "%repo_root%" %%F in (*.exe) do del /q "%%F"
)
echo [SUCCESS] Desktop package cleanup completed.
pause
goto :menu

:run_tests
set "test_script=%tests_dir%\run_tests.bat"
if not exist "%test_script%" (
  echo [ERROR] Missing test script: "%test_script%".
  pause
  goto :menu
)

echo [RUN] Executing test suite: "%test_script%"
pushd "%tests_dir%" >nul
call "%test_script%"
set "test_ec=%ERRORLEVEL%"
popd >nul
if "%test_ec%"=="0" (
  echo [SUCCESS] Test suite completed.
) else (
  echo [ERROR] Test suite failed with exit code %test_ec%.
)
pause
goto :menu

:remove_logs
if not exist "%log_dir%\" (
  echo [INFO] Log directory not found at "%log_dir%".
  pause
  goto :menu
)
if exist "%log_dir%\*.log" (
  del /q "%log_dir%\*.log"
  echo [SUCCESS] Log files deleted.
) else (
  echo [INFO] No log files found.
)
pause
goto :menu

:promote_node_runtime
set "node_source_dir=%~1"
if not defined node_source_dir exit /b 1
for %%D in ("%~1") do set "node_source_dir=%%~fD"
if /i "%node_source_dir%"=="%nodejs_dir%" exit /b 0

robocopy "%node_source_dir%" "%nodejs_dir%" /MOVE /E /R:2 /W:1 /NFL /NDL /NJH /NJS /NC /NS >nul
set "node_move_ec=%ERRORLEVEL%"
if %node_move_ec% geq 8 (
  echo [FATAL] Failed to flatten portable Node.js runtime from "%node_source_dir%".
  exit /b %node_move_ec%
)

if exist "%node_source_dir%" rd /s /q "%node_source_dir%" >nul 2>&1
exit /b 0

:exit
endlocal

