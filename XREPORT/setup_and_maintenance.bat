@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM == Configuration: define project and tool paths
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

set "pyproject=%root_folder%pyproject.toml"
set "update_script=%project_folder%tools\update_project.py"
set "log_path=%project_folder%resources\logs"
set "uv_lock=%root_folder%uv.lock"
set "venv_dir=%root_folder%.venv"
set "client_dir=%project_folder%client"
set "nodejs_dir=%runtimes_dir%\nodejs"


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Show setup menu
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:setup_menu
cls
echo ==========================================================================
echo                         Setup and Maintenance
echo ==========================================================================
echo 1. Remove logs
echo 2. Uninstall app
echo 3. Exit
echo.
set /p sub_choice="Select an option (1-5): "

if "%sub_choice%"=="3" goto :logs
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
echo This operation will remove uv artifacts, caches, local Python files in
echo resources\runtimes, the portable Node.js installation, and the .venv
echo directory. The embedded Python folder will be cleaned but the folder
echo structure will be preserved.
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
  echo [INFO] No uv.lock file found to remove.
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
  echo [INFO] No uv cache directory found to remove.
)
if exist "%python_dir%" (
  for /f "delims=" %%F in ('dir /b "%python_dir%"') do (
    if /i not "%%F"==".gitkeep" (
      if exist "%python_dir%\%%F\" (
        rd /s /q "%python_dir%\%%F"
      ) else (
        del /q "%python_dir%\%%F"
      )
    )
  )
  echo [INFO] Cleaned python directory contents.
) else (
  echo [INFO] Python directory "%python_dir%" not found.
)
if exist "%venv_dir%" (
  rd /s /q "%venv_dir%"
  echo [INFO] Removed virtual environment "%venv_dir%".
) else (
  echo [INFO] No .venv directory found to remove.
)
if exist "%client_dir%\node_modules" (
  rd /s /q "%client_dir%\node_modules"
  echo [INFO] Removed frontend node_modules at "%client_dir%\node_modules".
) else (
  echo [INFO] No frontend node_modules directory found to remove.
)
if exist "%nodejs_dir%" (
  rd /s /q "%nodejs_dir%"
  echo [INFO] Removed portable Node.js directory "%nodejs_dir%".
) else (
  echo [INFO] No portable Node.js directory found to remove.
)
if exist "%client_dir%\dist" (
  rd /s /q "%client_dir%\dist"
  echo [INFO] Removed frontend build directory "%client_dir%\dist".
) else (
  echo [INFO] No frontend build directory found to remove.
)
if exist "%client_dir%\package-lock.json" (
  del /q "%client_dir%\package-lock.json"
  echo [INFO] Removed frontend package-lock.json at "%client_dir%\package-lock.json".
) else (
  echo [INFO] No frontend package-lock.json found to remove.
)
echo [SUCCESS] Uninstall completed.
pause
goto :setup_menu

:exit
endlocal
