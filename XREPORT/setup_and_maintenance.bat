@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM == Configuration: define project and tool paths
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

set "pyproject=%root_folder%pyproject.toml"
set "update_script=%project_folder%tools\update_project.py"
set "log_path=%project_folder%resources\logs"
set "uv_lock=%runtimes_dir%\uv.lock"
set "venv_dir=%runtimes_dir%\.venv"
set "UV_PROJECT_ENVIRONMENT=%venv_dir%"
set "client_dir=%project_folder%client"
set "nodejs_dir=%runtimes_dir%\nodejs"
set "server_dir=%project_folder%server"
set "scripts_dir=%project_folder%\scripts"
set "init_db_script=%scripts_dir%\initialize_database.py"
set "tauri_clean_script=%root_folder%release\tauri\scripts\clean-tauri-build.ps1"
set "tauri_release_target=%client_dir%\src-tauri\target\release"
set "tauri_export_dir=%root_folder%release\windows"


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Show setup menu
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:setup_menu
cls
echo ==========================================================================
echo                         Setup and Maintenance
echo ==========================================================================
echo 1. Initialize database
echo 2. Remove logs
echo 3. Clean desktop build artifacts
echo 4. Uninstall app
echo 5. Exit
echo.
set /p sub_choice="Select an option (1-5): "

if "%sub_choice%"=="1" goto :run_init_db
if "%sub_choice%"=="2" goto :logs
if "%sub_choice%"=="3" goto :clean_desktop_build
if "%sub_choice%"=="4" goto :uninstall
if "%sub_choice%"=="5" goto :exit
echo Invalid option, try again.
pause
goto :setup_menu

:logs
if not exist "%log_path%" (
  echo [INFO] Log directory not found at "%log_path%".
  pause
  goto :setup_menu
)
if exist "%log_path%\*.log" (
  del /q "%log_path%\*.log"
  if "%ERRORLEVEL%"=="0" (
    echo [SUCCESS] Log files deleted.
  ) else (
    echo [WARN] Some log files could not be deleted.
  )
) else (
  echo [INFO] No log files found.
)
pause
goto :setup_menu

:uninstall
echo --------------------------------------------------------------------------
echo This operation will remove runtime-local uv artifacts, caches, lockfile,
echo virtual environment, local Python files in runtimes, and the portable
echo Node.js installation.
echo.
set /p confirm="Type YES to continue: "
if /i not "%confirm%"=="YES" (
  echo [INFO] Uninstall cancelled.
  pause
  goto :setup_menu
)
if exist "%uv_lock%" (
  del /q "%uv_lock%"
  echo [INFO] Removed "%uv_lock%".
) else (
  echo [INFO] No runtime lockfile found to remove at "%uv_lock%".
)
if exist "%uv_dir%" (
  rd /s /q "%uv_dir%"
  echo [INFO] Removed uv directory "%uv_dir%".
) else (
  echo [INFO] No uv directory found to remove.
)
if exist "%UV_CACHE_DIR%" (
  rd /s /q "%UV_CACHE_DIR%"
  echo [INFO] Removed uv cache "%UV_CACHE_DIR%".
) else (
  echo [INFO] No uv cache directory found to remove at "%UV_CACHE_DIR%".
)
:run_server_script_end
pause
exit /b !run_script_ec!

:exit
endlocal
