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
:: Check if conda is installed
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:check_conda
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Anaconda/Miniconda is not installed. Installing Miniconda..   
    cd /d "%conda_path%"        
    if not exist Miniconda3-latest-Windows-x86_64.exe (
        echo Downloading Miniconda 64-bit installer..
        powershell -Command "Invoke-WebRequest -Uri https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -OutFile Miniconda3-latest-Windows-x86_64.exe"
    )    
    echo Installing Miniconda to %conda_path%
    start /wait "" Miniconda3-latest-Windows-x86_64.exe ^
        /InstallationType=JustMe ^
        /RegisterPython=0 ^
        /AddToPath=0 ^
        /S ^
        /D=%conda_path%    
    
    call "%conda_path%\Scripts\activate.bat" "%conda_path%"
    echo Miniconda installation is complete.    
    goto :check_environment

) else (
    echo Anaconda/Miniconda already installed.   
    goto :check_environment
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Check if the environment exists when not using a custom environment
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:check_environment
if exist "%env_path%" (    
    echo Python environment '%env_name%' detected.
    goto :conda_activation

) else (
    echo Running first-time installation for %env_name%. 
    echo Please wait until completion and do not close this window!
    echo Depending on your internet connection, this may take a while..
    call "%setup_path%\install_on_windows.bat"
    goto :conda_activation
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Precheck for conda source 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:conda_activation
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (   
    call "%conda_path%\Scripts\activate.bat" "%conda_path%"      
    goto :main_menu
) 

call conda activate "%env_path%" && python "%app_path%\commons\app\main.py"