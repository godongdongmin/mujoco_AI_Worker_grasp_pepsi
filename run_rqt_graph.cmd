@echo off
title ROS 2 - rqt_graph
cd /d "%~dp0"
echo Starting rqt_graph in Ubuntu-24.04...
echo A separate "Node Graph" window will open. This console normally stays quiet.
echo Run run_ros2.cmd as well to see the MuJoCo node.
echo.
wsl.exe --distribution Ubuntu-24.04 --cd "%CD%" --exec bash ros2/graph.sh %*
set "GRAPH_EXIT=%ERRORLEVEL%"
echo rqt_graph closed. Exit code: %GRAPH_EXIT%
if not "%GRAPH_EXIT%"=="0" pause
exit /b %GRAPH_EXIT%
