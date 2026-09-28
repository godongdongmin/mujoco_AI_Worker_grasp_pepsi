#!/usr/bin/env bash
set -eo pipefail
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
    echo 'ROS 2 Jazzy is not installed. Run ros2/setup_jazzy.sh first.' >&2
    exit 1
fi
# rqt uses Ubuntu's Qt/Python packages, separate from the MuJoCo venv.
source /opt/ros/jazzy/setup.bash
if ! ros2 pkg prefix rqt_graph >/dev/null 2>&1; then
    echo 'Install: sudo apt-get install ros-jazzy-rqt-graph' >&2
    exit 1
fi
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
exec ros2 run rqt_graph rqt_graph "$@"
