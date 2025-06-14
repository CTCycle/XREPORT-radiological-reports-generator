@echo off
setlocal enabledelayedexpansion

for /f "delims=" %%i in ("%~dp0..") do set "project_folder=%%~fi"
set "env_name=XREPORT"
set "project_name=XREPORT"
set "setup_path=%project_folder%\setup"
set "env_path=%setup_path%\environment\%env_name%"
set "conda_path=%setup_path%\miniconda"
set "app_path=%project_folder%\%project_name%"

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Show setup menu
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:setup_menu
cls
echo ==========================================================================
echo                         Setup  and Maintenance                          
echo ==========================================================================
echo 1. Run installation
echo 2. Enable root path imports
echo 3. Update project
echo 4. Remove logs
echo 5. Exit
echo.
set /p sub_choice="Select an option (1-5): "

if "%sub_choice%"=="1" goto :start_installation
if "%sub_choice%"=="2" goto :eggs
if "%sub_choice%"=="3" goto :update
if "%sub_choice%"=="4" goto :logs
if "%sub_choice%"=="5" goto :exit
echo Invalid option, try again.
goto :setup_menu

:start_installation
call "%setup_path%\install_on_windows.bat"
goto :setup_menu

:eggs
call conda activate "%env_path%" && cd "%project_folder%" && pip install -e . --use-pep517
pause
goto :setup_menu

:update
cd "%project_folder%"
call git pull
if errorlevel 1 (
    echo Error: Git pull failed.
    pause
    goto :setup_menu
)
pause
goto :setup_menu

:logs
cd "%app_path%\resources\logs" 
if not exist *.log (
    echo No log files found.
    pause
    goto :setup_menu
)
del *.log /q
echo Log files deleted.
pause
goto :setup_menu

:exit
endlocal