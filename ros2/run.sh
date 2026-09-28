#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ! -f /opt/ros/jazzy/setup.bash || ! -x .venv_ros2/bin/python ]]; then
    echo 'First run: bash ros2/setup_jazzy.sh' >&2
    exit 1
fi
source /opt/ros/jazzy/setup.bash
source .venv_ros2/bin/activate
export LANG=C.UTF-8
exec python run_ros2.py "$@"
