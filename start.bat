@echo off
cd /d "%~dp0"
title WiiRemoteAssignments
py app.py %*
pause
