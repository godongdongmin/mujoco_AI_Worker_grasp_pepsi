@echo off
cd /d "%~dp0"
wsl.exe --distribution Ubuntu-24.04 --cd "%CD%" --exec bash ros2/run.sh --gui %*
set "ROS_EXIT=%ERRORLEVEL%"
if not "%ROS_EXIT%"=="0" pause
exit /b %ROS_EXIT%
