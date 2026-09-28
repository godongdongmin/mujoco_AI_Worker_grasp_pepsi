#!/usr/bin/env bash
# Run inside Ubuntu 24.04 as a normal user. Installs ROS 2 and a separate venv.
set -euo pipefail
source /etc/os-release
if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != 24.04 ]]; then
    echo 'This setup requires Ubuntu 24.04.' >&2
    exit 1
fi
if [[ "$EUID" == 0 ]]; then
    echo 'Run as your Ubuntu user; sudo is used only for system packages.' >&2
    exit 1
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
sudo apt-get update
sudo apt-get install -y curl ca-certificates software-properties-common python3-venv git
sudo add-apt-repository -y universe
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
    sudo dpkg -i outputs/ros2-apt-source.deb
fi
sudo apt-get update
sudo apt-get install -y ros-jazzy-ros-base ros-jazzy-tf2-ros ros-jazzy-tf2-msgs \
    libgl1 libegl1 libglfw3 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1
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
