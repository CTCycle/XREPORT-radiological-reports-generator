@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM == Configuration: define project and Python paths
REM ============================================================================
set "project_folder=%~dp0"
set "root_folder=%project_folder%..\"
set "python_dir=%project_folder%setup\python"
set "python_exe=%python_dir%\python.exe"
set "pip_exe=%python_dir%\Scripts\pip.exe"
set "app_script=%project_folder%app\app.py"
set "requirements_path=%project_folder%setup\requirements.txt"
set "log_path=%project_folder%resources\logs"


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Show setup menu
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:setup_menu
cls
echo ==========================================================================
echo                         Setup  and Maintenance                          
echo ==========================================================================
echo 1. Enable root path imports
echo 2. Update project
echo 3. Remove logs
echo 4. Exit
echo.
set /p sub_choice="Select an option (1-4): "

if "%sub_choice%"=="1" goto :eggs
if "%sub_choice%"=="2" goto :update
if "%sub_choice%"=="3" goto :logs
if "%sub_choice%"=="4" goto :exit
echo Invalid option, try again.
goto :setup_menu

:eggs
pushd "%root_folder%"
"%pip_exe%" install -e . --use-pep517 || popd
popd
goto :setup_menu

:update
echo Updating project... 
"%python_exe%" "%project_folder%setup\scripts\update_project.py"
pause
goto :setup_menu

:logs
pushd %log_path%
if not exist *.log (
    echo No log files found.
    popd
    pause
    goto :setup_menu
)

del *.log /q || popd
echo Log files deleted 
popd 
pause
goto :setup_menu


:exit
endlocal