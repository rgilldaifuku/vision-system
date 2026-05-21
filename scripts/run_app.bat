@echo off
cd /d "%~dp0.."

echo Starting AI Detection System...
echo Current folder:
cd
echo.

echo Checking Python
py --version


echo.
echo Launching app...
py -m app.main

echo.
echo App closed.
pause