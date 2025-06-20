@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM == Configuration: define project and Python paths
REM ============================================================================
set "project_folder=%~dp0"
set "python_dir=%project_folder%app\python"
set "python_exe=%python_dir%\python.exe"
set "pip_exe=%python_dir%\Scripts\pip.exe"
set "app_script=%project_folder%app\src\main.py"
set "requirements_path=%project_folder%app\requirements.txt"
set "git_dir=%project_folder%app\git"
set "git_exe=%git_dir%\cmd\git.exe"
set "triton_path=%project_folder%app\triton\triton-3.2.0-cp312-cp312-win_amd64.whl"

REM ============================================================================
REM == 0. Skip full setup if environment already present
REM ============================================================================
if exist "%python_exe%" if exist "%pip_exe%" if exist "%git_exe%" (
    echo [INFO] Environment already installed. Skipping setup.
    goto :run_app
)

REM ============================================================================
REM == 1. Full environment setup
REM ============================================================================
echo [STEP 1/4] Setting up Python environment...

REM --- Dynamic variables for Python distribution
set "python_version=3.12.10"
set "python_major_version=3"
set "python_minor_version=12"
set "python_zip_filename=python-%python_version%-embed-amd64.zip"
set "python_zip_url=https://www.python.org/ftp/python/%python_version%/%python_zip_filename%"
set "python_zip_path=%python_dir%\%python_zip_filename%"
set "python_pth_file=%python_dir%\python%python_major_version%%python_minor_version%._pth"
set "get_pip_url=https://bootstrap.pypa.io/get-pip.py"
set "get_pip_path=%python_dir%\get-pip.py"
set "root_folder=%project_folder%..\"
set "get_pip_url=https://bootstrap.pypa.io/get-pip.py"
set "get_pip_path=%python_dir%\get-pip.py"

REM Create Python directory
mkdir "%python_dir%" 2>nul

REM Download and extract embeddable Python
powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%python_zip_url%', '%python_zip_path%')" || goto :error
powershell -Command "Expand-Archive -Path '%python_zip_path%' -DestinationPath '%python_dir%' -Force" || goto :error
del "%python_zip_path%"

REM Bootstrap pip
powershell -Command "(Get-Content '%python_pth_file%') -replace '#import site','import site' | Set-Content '%python_pth_file%'" || goto :error
powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%get_pip_url%', '%get_pip_path%')" || goto :error
"%python_exe%" "%get_pip_path%" || goto :error
del "%get_pip_path%"

REM Install dependencies
echo [STEP 2/4] Installing dependencies...
if not exist "%requirements_path%" (
    echo [FATAL] requirements.txt not found.
    goto :error
)

echo [INFO] Upgrading pip package
"%pip_exe%" install --upgrade pip >nul 2>&1

"%pip_exe%" install --no-warn-script-location -r "%requirements_path%" || goto :error

echo [INFO] Installing setuptools
"%pip_exe%" install --no-warn-script-location setuptools wheel || goto :error

echo [INFO] Installing triton
"%pip_exe_path%" install "%triton_path%" || goto :error

pushd "%root_folder%"
"%pip_exe%" install -e . --use-pep517 || (popd & goto :error)
popd

"%pip_exe%" cache purge || goto :error

echo [SUCCESS] Environment setup complete.

REM ============================================================================
REM == 3. Install git
REM ============================================================================
REM Install git
echo [STEP 3/4] Installing portable Git
    
REM Bootstrap Git
set "git_zip_url=https://github.com/git-for-windows/git/releases/download/v2.45.1.windows.1/PortableGit-2.45.1-64-bit.7z.exe"
set "git_zip_path=%git_dir%\git_portable.7z.exe"
mkdir "%git_dir%" 2>nul
powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%git_zip_url%', '%git_zip_path%')" || goto :error
"%git_zip_path%" -o"%git_dir%" -y
del "%git_zip_path%"

echo [SUCCESS] Git has been installed locally.


REM ============================================================================
REM == 2. Run the application
REM ============================================================================
:run_app
echo [STEP 4/4] Running application...
if not exist "%app_script%" (
    echo [FATAL] Application script not found: "%app_script%"
    goto :error
)

"%python_exe%" "%app_script%" || goto :error
echo [SUCCESS] Application launched successfully.

endlocal
exit /b 0

REM ============================================================================
REM == Error handling
REM ============================================================================
:error
echo.
echo !!! An error occurred during execution. !!!
pause
endlocal
exit /b 1
