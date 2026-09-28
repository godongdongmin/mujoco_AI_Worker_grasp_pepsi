@echo off
cd /d "%~dp0"
wsl.exe --distribution Ubuntu-24.04 --cd "%CD%" --exec bash ros2/graph.sh %*
set "GRAPH_EXIT=%ERRORLEVEL%"
if not "%GRAPH_EXIT%"=="0" pause
exit /b %GRAPH_EXIT%
