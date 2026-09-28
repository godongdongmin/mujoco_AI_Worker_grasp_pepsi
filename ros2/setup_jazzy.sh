#!/usr/bin/env bash
# Ubuntu 24.04. System packages and user venv can also be installed separately.
set -euo pipefail
mode="${1:---all}"
case "$mode" in --all|--system-only|--env-only) ;; *) echo 'Use --all, --system-only or --env-only' >&2; exit 1 ;; esac
source /etc/os-release
if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != 24.04 ]]; then
    echo 'This setup requires Ubuntu 24.04.' >&2
    exit 1
fi
if [[ "$EUID" == 0 && "$mode" != --system-only ]]; then
    echo 'Run the project environment setup as your Ubuntu user.' >&2
    exit 1
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
privilege=()
if [[ "$EUID" != 0 ]]; then privilege=(sudo); fi
if [[ "$mode" != --env-only ]]; then
"${privilege[@]}" apt-get update
"${privilege[@]}" apt-get install -y curl ca-certificates software-properties-common python3-venv git
"${privilege[@]}" add-apt-repository -y universe
# Official ROS repository configuration package, following Jazzy's install guide.
if ! dpkg-query -W -f='${Status}' ros2-apt-source 2>/dev/null | grep -q 'install ok installed'; then
    release="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest |
        python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')"
    if [[ ! "$release" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo 'Unexpected ros2-apt-source release tag.' >&2
        exit 1
    fi
    mkdir -p outputs
    curl -fL "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${release}/ros2-apt-source_${release}.noble_all.deb" \
        -o outputs/ros2-apt-source.deb
    "${privilege[@]}" dpkg -i outputs/ros2-apt-source.deb
fi
"${privilege[@]}" apt-get update
"${privilege[@]}" apt-get install -y ros-jazzy-ros-base ros-jazzy-tf2-ros ros-jazzy-tf2-msgs \
    libgl1 libegl1 libglfw3 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1
fi
if [[ "$mode" == --system-only ]]; then exit 0; fi
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
    echo 'Install system dependencies first with --system-only.' >&2
    exit 1
fi
model_repo=vendor/robotis_mujoco_menagerie
model_commit=d8344c0dbe7a00208d0301111523dde65efc174a
if [[ ! -d "$model_repo" ]]; then
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie.git "$model_repo"
    git -C "$model_repo" fetch --depth 1 origin "$model_commit"
    git -C "$model_repo" checkout --detach "$model_commit"
    git -C "$model_repo" sparse-checkout set robotis_ffw
elif [[ "$(git -C "$model_repo" rev-parse HEAD)" != "$model_commit" ]]; then
    echo 'Existing vendor model has a different revision; left unchanged.' >&2
    exit 1
fi
if [[ ! -x .venv_ros2/bin/python ]]; then
    /usr/bin/python3 -m venv --system-site-packages .venv_ros2
fi
set +u
source /opt/ros/jazzy/setup.bash
source .venv_ros2/bin/activate
set -u
python -m pip install -r requirements.txt
python -c 'import rclpy, tf2_ros, mujoco; print("ROS 2 / MuJoCo imports OK")'
python validate_ros2_adapter.py
echo 'Ready: bash ros2/run.sh --gui'
