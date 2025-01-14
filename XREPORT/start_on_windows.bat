@echo off
setlocal enabledelayedexpansion

set "env_name=XREPORT"
set "project_name=XREPORT"

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Check if conda is installed
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:check_conda
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Anaconda/Miniconda is not installed. Installing Miniconda...   
    cd /d "%~dp0"        
    if not exist Miniconda3-latest-Windows-x86_64.exe (
        echo Downloading Miniconda 64-bit installer...
        powershell -Command "Invoke-WebRequest -Uri https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -OutFile Miniconda3-latest-Windows-x86_64.exe"
    )    
    echo Installing Miniconda to %USERPROFILE%\Miniconda3
    start /wait "" Miniconda3-latest-Windows-x86_64.exe ^
        /InstallationType=JustMe ^
        /RegisterPython=0 ^
        /AddToPath=0 ^
        /S ^
        /D=%~dp0setup\miniconda    
    
    call "%~dp0..\setup\miniconda\Scripts\activate.bat" "%~dp0..\setup\miniconda"
    echo Miniconda installation is complete.    
    goto :initial_check

) else (
    echo Anaconda/Miniconda already installed. Checking python environment...    
    goto :initial_check
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Check if the environment exists when not using a custom environment
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:initial_check   
cd /d "%~dp0\.."

:check_environment
set "env_path=.setup\environment\%env_name%"

if exist ".setup\environment\%env_name%\" (    
    echo Python environment '%env_name%' detected.
    goto :cudacheck

) else (
    echo Running first-time installation for %env_name%. 
    echo Please wait until completion and do not close this window!
    echo Depending on your internet connection, this may take a while...
    call ".\setup\install_on_windows.bat"
    goto :cudacheck
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Check if NVIDIA GPU is available using nvidia-smi
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:cudacheck
nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    echo NVIDIA GPU detected. Checking CUDA version...
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
) else (
    echo No NVIDIA GPU detected or NVIDIA drivers are not installed.
)
goto :main_menu


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Precheck for conda source 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:conda_activation
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (   
    call "%~dp0..\setup\miniconda\Scripts\activate.bat" "%~dp0..\setup\miniconda"       
    goto :main_menu
) 

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Show main menu
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:main_menu
echo.
echo =======================================
echo                 XREPORT
echo =======================================
echo 1. Dataset analysis
echo 2. Preprocess dataset
echo 3. Model training and evaluation
echo 4. Generate radiological reports
echo 5. Setup and Maintenance
echo 6. Exit
echo.
set /p choice="Select an option (1-6): "

if "%choice%"=="1" goto :datanalysis
if "%choice%"=="2" goto :processing
if "%choice%"=="3" goto :ML_menu
if "%choice%"=="4" goto :inference
if "%choice%"=="5" goto :setup_menu
if "%choice%"=="6" goto exit

echo Invalid option, try again.
pause
goto :main_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run data analysis
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:datanalysis
cls
start cmd /k "call conda activate --prefix %env_path% && jupyter notebook .\validation\dataset_validation.ipynb"
goto :main_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run preprocessing
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:processing
cls
call conda activate --prefix %env_path% && python .\preprocessing\dataset_preprocessing.py
goto :main_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run model inference
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:inference
cls
call conda activate --prefix %env_path% && python .\inference\report_generator.py
goto :main_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Start machine learning menu
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:ML_menu
cls
echo =======================================
echo              XREPORT ML
echo =======================================
echo 1. Train from scratch
echo 2. Train from checkpoint
echo 3. Evaluate model performances
echo 4. Back to main menu
echo.
set /p sub_choice="Select an option (1-4): "

if "%sub_choice%"=="1" goto :train_fs
if "%sub_choice%"=="2" goto :train_ckpt
if "%sub_choice%"=="3" goto :modeleval
if "%sub_choice%"=="4" goto :main_menu
echo Invalid option, try again.
pause
goto :ML_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run model training from scratch
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:train_fs
cls
call conda activate --prefix %env_path% && python .\training\model_training.py
pause
goto :ML_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run model training from checkpoint
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:train_ckpt
cls
call conda activate --prefix %env_path% && python .\training\train_from_checkpoint.py
goto :ML_menu


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run model evaluation
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:modeleval
cls
start cmd /k "call conda activate --prefix %env_path% && jupyter notebook .\validation\model_evaluation.ipynb"
goto :ML_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Show setup menu
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:setup_menu
cls
echo =======================================
echo         Setup and Maintenance
echo =======================================
echo 1. Install project in editable mode
echo 2. Remove logs
echo 3. Back to main menu
echo.
set /p sub_choice="Select an option (1-3): "

if "%sub_choice%"=="1" goto :eggs
if "%sub_choice%"=="2" goto :logs
if "%sub_choice%"=="3" goto :main_menu
echo Invalid option, try again.
pause
goto :setup_menu

:eggs
call conda activate --prefix %env_path% && cd .. && pip install -e . --use-pep517 && cd %project_name%
pause
goto :setup_menu

:logs
cd /d "%~dp0..\%project_name%\resources\logs"
del *.log /q
cd ..\..
pause
goto :setup_menu