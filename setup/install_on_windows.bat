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
:: Precheck for conda source 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:conda_activation
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (   
    call "%conda_path%\Scripts\activate.bat" "%conda_path%"     
    goto :check_env
)  

:: [CHECK CUSTOM ENVIRONMENTS] 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Check if the Python environment is available or else install it
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:check_env
call conda activate %env_path% 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python v3.12 environment "%env_name%" is being created
    call conda create --prefix "%env_path%" python=3.12 -y
    call conda activate "%env_path%"
)
goto :check_git

:: [INSTALL GIT] 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Install git using conda
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:check_git
echo.
echo Checking git installation
git --version >nul 2>&1
if errorlevel 1 (
    echo Git not found. Installing git using conda..
    call conda install -y git
) else (
    echo Git is already installed.
)
goto :dependencies

:: [INSTALL DEPENDENCIES] 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Install dependencies to python environment
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:dependencies
echo.
echo Install python libraries and packages
call pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
call pip install tensorflow==2.18.1
call pip install keras==3.8.0 transformers==4.45.2 scikit-learn==1.6.1 opencv-python==4.11.0.86
call pip install matplotlib==3.10.0 numpy==2.0.2 pandas==2.2.3 tqdm==4.67.1 

:: [INSTALL TRITON] 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Install dependencies to python environment
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
echo Installing triton from windows wheel
cd triton
call cd  "%setup_path%\triton" && pip install triton-3.2.0-cp312-cp312-win_amd64.whl
cd .

:: [INSTALLATION OF PYDOT/PYDOTPLUS]
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Install pydot/pydotplus for graphic model visualization
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
echo Installing pydot and pydotplus..
call conda install pydot -y
call conda install pydotplus -y

:: [INSTALL PROJECT IN EDITABLE MODE] 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Install project in developer mode
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
echo Install utils packages in editable mode
call cd "%project_folder%" && pip install -e . --use-pep517

:: [CLEAN CACHE] 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Clean packages cache
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
echo.
echo Cleaning conda and pip cache 
call conda clean --all -y
call pip cache purge

:: [SHOW LIST OF INSTALLED DEPENDENCIES]
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Show installed dependencies
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
echo.
echo List of installed dependencies:
call conda list
echo.
echo Installation complete. You can now run '%env_name%' on this system!
pause