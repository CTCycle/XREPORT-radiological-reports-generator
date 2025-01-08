@echo off
cd /d "%~dp0"

set "env_name=XREPORT"
set "project_name=XREPORT"
set "env_path=.\environment\%env_name%"

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Precheck for conda source 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:conda_activation
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (   
    call "%~dp0miniconda\Scripts\activate.bat" "%~dp0miniconda"       
    goto :main_menu
) 

:: [CHECK CUSTOM ENVIRONMENTS] 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Check if FEXT environment is available or use custom environment
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
call conda activate "%env_path%" 2>nul
if %ERRORLEVEL% neq 0 (
    echo %env_name% is being created (python version = 3.11)
    call conda create --prefix "%env_path%" python=3.11 -y
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
    echo Git not found. Installing git using conda...
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
call pip install torch==2.5.0+cu124 torchvision==0.20.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
call pip install https://storage.googleapis.com/tensorflow/versions/2.18.0/tensorflow-2.18.0-cp311-cp311-win_amd64.whl
call pip install keras==3.7.0 transformers==4.45.2 scikit-learn==1.6.0 opencv-python==4.10.0.84
call pip install matplotlib==3.9.2 numpy==2.1.0 pandas==2.2.3 tqdm==4.66.4 matplotlib==3.9.0
call pip install jupyter==1.1.1

:: [INSTALL TRITON] 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Install dependencies to python environment
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
echo Installing triton from windows wheel
cd triton
call pip install triton-3.1.0-cp311-cp311-win_amd64.whl
cd ..

:: [INSTALLATION OF PYDOT/PYDOTPLUS]
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Install pydot/pydotplus for graphic model visualization
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
echo Installing pydot and pydotplus...
call conda install pydot -y
call conda install pydotplus -y

:: [INSTALL PROJECT IN EDITABLE MODE] 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Install project in developer mode
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
echo Install utils packages in editable mode
call cd .. && pip install -e . --use-pep517 && cd %project_name%

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
echo Installation complete. You can now run %env_name% on this system!
pause