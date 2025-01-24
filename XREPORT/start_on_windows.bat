@echo off
setlocal enabledelayedexpansion

for /f "delims=" %%i in ("%~dp0..") do set "project_folder=%%~fi"
set "env_name=XREPORT"
set "project_name=XREPORT"
set "env_path=%project_folder%\setup\environment\%env_name%"
set "app_path=%project_folder%\%project_name%"
set "conda_path=%project_folder%\setup\miniconda"
set "setup_path=%project_folder%\setup"

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Check if conda is installed
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:check_conda
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Anaconda/Miniconda is not installed. Installing Miniconda...   
    cd /d "%conda_path%"        
    if not exist Miniconda3-latest-Windows-x86_64.exe (
        echo Downloading Miniconda 64-bit installer...
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
    goto :cudacheck

) else (
    echo Running first-time installation for %env_name%. 
    echo Please wait until completion and do not close this window!
    echo Depending on your internet connection, this may take a while...
    call "%setup_path%\install_on_windows.bat"
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
    call "%conda_path%\Scripts\activate.bat" "%conda_path%"      
    goto :main_menu
) 

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Show main menu
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:main_menu
echo.
echo ==========================================================================
echo        XREPORT: generate radiological reports with machine learning
echo ==========================================================================
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
goto :main_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run data analysis
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:datanalysis
cls
start cmd /k "call conda activate "%env_path%" && jupyter notebook "%app_path%"\validation\dataset_validation.ipynb"
goto :main_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run preprocessing
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:processing
cls
start cmd /k "call conda activate "%env_path%" && python "%app_path%"\database\build_processed_dataset.py"
goto :main_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run model inference
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:inference
cls
start cmd /k "call conda activate "%env_path%" && python "%app_path%"\inference\report_generator.py"
goto :main_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Start machine learning menu
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:ML_menu
cls
echo ==========================================================================
echo                               XREPORT ML
echo ==========================================================================
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
goto :ML_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run model training from scratch
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:train_fs
cls
start cmd /k "call conda activate "%env_path%" && python "%app_path%"\training\model_training.py"
pause
goto :ML_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run model training from checkpoint
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:train_ckpt
cls
start cmd /k "call conda activate "%env_path%" && python "%app_path%"\training\train_from_checkpoint.py"
goto :ML_menu


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run model evaluation
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:modeleval
cls
start cmd /k "call conda activate "%env_path%" && jupyter notebook "%app_path%"\validation\model_evaluation.ipynb"
goto :ML_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Show setup menu
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:setup_menu
cls
echo ==========================================================================
echo                         Setup  and Maintenance                          
echo ==========================================================================
echo 1. Install project in editable mode
echo 2. Update project
echo 3. Remove logs
echo 4. Back to main menu
echo.
set /p sub_choice="Select an option (1-4): "

if "%sub_choice%"=="1" goto :eggs
if "%sub_choice%"=="2" goto :update
if "%sub_choice%"=="3" goto :logs
if "%sub_choice%"=="4" goto :main_menu
echo Invalid option, try again.
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