@echo off
rem Use this script to create a new environment called "XREPORT"

call conda activate XREPORT && cd .. && pip install -e . --use-pep517
if errorlevel 1 (
    echo Failed to install the package in editable mode
    goto :eof
)

