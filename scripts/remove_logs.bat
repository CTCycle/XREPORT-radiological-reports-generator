@echo off
REM Change directory to the resources/logs folder 
cd /d "%~dp0..\XREPORT\resources\logs"

REM Remove all .log files in the resources/logs folder
del *.log /q

