@echo off
rem Double-click this to open the dashboard.
rem It starts the server if it is not already running, then opens your browser.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch.ps1" %*
if errorlevel 1 pause
