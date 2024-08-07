@echo off
rem Use this script to create a new environment called "XREPORT"

echo STEP 1: Creation of XREPORT environment
call conda create -n XREPORT python=3.11 -y
if errorlevel 1 (
    echo Failed to create the environment XREPORT
    goto :eof
)

rem If present, activate the environment
call conda activate XREPORT

rem Install additional packages with pip
echo STEP 2: Install python libraries and packages
call --extra-index-url https://download.pytorch.org/whl/cu121
call pip install torch==2.4.0+cu121
call pip install torchvision==0.19.0+cu121
call pip install tensorflow-cpu==2.17.0 keras==3.4.1 transformers==4.43.3
call pip install numpy==1.26.4 pandas==2.2.2 openpyxl==3.1.5 tqdm==4.66.4 
call pip install scikit-learn==1.2.2 matplotlib==3.9.0 opencv-python==4.10.0.84
if errorlevel 1 (
    echo Failed to install Python libraries.
    goto :eof
)

rem Install additional tools
echo STEP 3: Install additional libraries
call conda install pydot -y
call conda install pydotplus -y
if errorlevel 1 (
    echo Failed to install Graphviz or Pydot.
    goto :eof
)

@echo off
rem install packages in editable mode
echo STEP 4: Install utils packages in editable mode
call cd .. && pip install -e .
if errorlevel 1 (
    echo Failed to install the package in editable mode
    goto :eof
)

rem Clean cache
echo Cleaning conda and pip cache 
call conda clean -all -y
call pip cache purge


rem Print the list of dependencies installed in the environment
echo List of installed dependencies
call conda list

set/p<nul =Press any key to exit... & pause>nul
