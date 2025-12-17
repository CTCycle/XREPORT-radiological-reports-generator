@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM == Configuration: define project and tool paths
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

set "pyproject=%root_folder%pyproject.toml"
set "update_script=%setup_dir%\scripts\update_project.py"
set "log_path=%project_folder%resources\logs"
set "uv_lock=%root_folder%uv.lock"
set "venv_dir=%root_folder%.venv"


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Show setup menu
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:setup_menu
cls
echo ==========================================================================
echo                         Setup and Maintenance
echo ==========================================================================
echo 1. Enable root path imports
echo 2. Update project
echo 3. Remove logs
echo 4. Uninstall app
echo 5. Exit
echo.
set /p sub_choice="Select an option (1-5): "

if "%sub_choice%"=="1" goto :eggs
if "%sub_choice%"=="2" goto :update
if "%sub_choice%"=="3" goto :logs
if "%sub_choice%"=="4" goto :uninstall
if "%sub_choice%"=="5" goto :exit
echo Invalid option, try again.
pause
goto :setup_menu

:eggs
if not exist "%uv_exe%" (
  echo [ERROR] uv executable not found at "%uv_exe%".
  echo Please run start_on_windows.bat first.
  pause
  goto :setup_menu
)
if not exist "%pyproject%" (
  echo [ERROR] Missing pyproject: "%pyproject%".
  pause
  goto :setup_menu
)
pushd "%root_folder%" >nul
"%uv_exe%" pip install --python "%python_exe%" -e .
set "uv_ec=%ERRORLEVEL%"
popd >nul
if not "%uv_ec%"=="0" (
  echo [ERROR] Failed to enable root imports. uv exited with code %uv_ec%.
) else (
  echo [SUCCESS] Root path imports enabled using uv.
)
pause
goto :setup_menu

:update
if not exist "%uv_exe%" (
  echo [ERROR] uv executable not found at "%uv_exe%".
  echo Please run start_on_windows.bat first.
  pause
  goto :setup_menu
)
if not exist "%update_script%" (
  echo [ERROR] Update script not found at "%update_script%".
  pause
  goto :setup_menu
)
"%uv_exe%" run --python "%python_exe%" python "%update_script%"
set "update_ec=%ERRORLEVEL%"
if not "%update_ec%"=="0" (
  echo [ERROR] Project update failed with code %update_ec%.
) else (
  echo [SUCCESS] Project update completed.
)
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
echo This operation will remove uv artifacts, caches, local Python files,
echo and the .venv directory. The embedded python folder will be cleaned but
echo the folder structure will be preserved.
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
if exist "%uv_cache_dir%" (
  rd /s /q "%uv_cache_dir%"
  echo [INFO] Removed uv cache "%uv_cache_dir%".
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
echo [SUCCESS] Uninstall completed.
pause
goto :setup_menu

:exit
endlocal
