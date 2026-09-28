# ROS 2 연동

- 대상: WSL2 / Ubuntu 24.04 / ROS 2 Jazzy / Python 3.12.
- `main`의 Windows 앱은 유지. 이 브랜치는 동일한 `Session`을 사용하는 별도 ROS 실행 파일을 추가합니다.
- 구현: 상태 발행, 손·관절 목표 수신, 초기화·일시정지·현재 자세 유지 서비스.
- 검증(2026-09-28): 어댑터 6개 테스트, 실제 ROS 토픽·서비스·TF·clock, WSLg GUI 및 79개 제어 위젯 통과.
- 이 PC의 GUI 측정: NVIDIA RTX 5060, 평균 제어 99.2 Hz / 약 29 fps. 하드 실시간 보장은 아닙니다.
- ros2_control, MoveIt, 전체 로봇 URDF/TF, 궤적 Action, 충돌 회피는 포함하지 않습니다.

## 설치

1. Windows 관리자 PowerShell: `wsl --install -d Ubuntu-24.04 --no-launch`.
2. 재부팅을 요구하면 재부팅 후 `wsl -d Ubuntu-24.04`로 시작하여 Ubuntu 사용자·암호를 설정합니다. 배포판이 없다면 1번 명령을 다시 실행합니다.
3. Ubuntu 터미널에서 저장소를 준비하고 설치합니다. Linux 파일시스템 사용을 권장합니다.

```bash
mkdir -p ~/projects
cd ~/projects
git clone --branch feature/ros2-integration https://github.com/godongdongmin/mujoco_AI_Worker_grasp_pepsi.git
cd mujoco_AI_Worker_grasp_pepsi
bash ros2/setup_jazzy.sh
bash ros2/run.sh --gui
```

Windows의 현재 프로젝트 폴더를 WSL에서 열어도 됩니다. Linux용 `.venv_ros2`와 Windows용 `.venv`는 분리되어 있습니다. GUI 없이 실행하려면 `--gui`를 생략합니다. `--duration 10`은 벽시계 기준 10초 후 종료합니다.

이 PC의 설치 완료 후에는 Windows에서 **`run_ros2.cmd`**로 ROS 연동 GUI를 실행할 수 있습니다. `run_ik.cmd`는 기존 Windows 앱을 실행합니다. 관리자 설치와 사용자 환경 구성이 분리된 경우 `setup_jazzy.sh --system-only` / `--env-only`를 사용할 수 있습니다.

## 노드 연결도

- Windows: **`run_rqt_graph.cmd`**. Ubuntu: `bash ros2/graph.sh`.
- 명령창은 시작 안내 이후 조용한 것이 정상이며, 별도 **Node Graph** 창에서 확인합니다.
- `run_ros2.cmd`도 실행해야 `ai_worker_bridge`가 표시됩니다. 나중에 실행했다면 연결도에서 새로고침합니다.
- 토픽까지 보려면 상단을 **Nodes/Topics (all)**로 설정하고 **Dead sinks / Leaf topics** 숨기기를 해제합니다. 현재 외부 구독자가 없는 상태 토픽도 표시됩니다.
- 별도 제어 노드는 추가하지 않습니다. MuJoCo·IK·GUI는 현재 `ai_worker_bridge` 내부에 있습니다.

창이 검거나 제목에 `[WARN:COPY MODE]`가 있으면 WSLg 표시 오류일 수 있습니다. WSL 작업을 저장·종료한 뒤 Windows 터미널에서 `wsl --shutdown`을 실행하고 두 실행 파일을 다시 시작합니다. 이 명령은 실행 중인 모든 WSL 프로그램을 종료합니다. [관련 WSLg 오류](https://github.com/microsoft/openvmm/issues/4274)

## 인터페이스

기본 namespace 기준입니다. 위치 m, 회전 rad, ROS quaternion 순서 **x,y,z,w**.

| 방향 | 이름 | 타입 / 내용 |
|---|---|---|
| 발행 | `/joint_states` | `sensor_msgs/msg/JointState`: 실제 63개 관절 위치·속도. effort 미제공 |
| 발행 | `/ai_worker/{left,right}/pose` | `geometry_msgs/msg/PoseStamped`: 실제 손 기준점의 world 자세 |
| 발행 | `/tf` | world → ai_worker_left_hand / ai_worker_right_hand 두 변환 |
| 발행 | `/clock` | 초기 안착 이후 시뮬레이션 시간. reset에도 역행하지 않음 |
| 수신 | `/ai_worker/{left,right}/target_pose` | `PoseStamped`: frame_id는 `world`, quaternion은 단위 크기 |
| 수신 | `/ai_worker/joint_targets` | `JointState`: name / position만 사용. 일부 관절 지정 가능 |
| 수신 | `/ai_worker/{left,right}/{thumb,grasp}` | `std_msgs/msg/Float64`: 0–1 프리셋 값 |
| 서비스 | `/ai_worker/reset` | `std_srvs/srv/Trigger`: 로봇·캔 초기화, pause 상태 유지 |
| 서비스 | `/ai_worker/hold` | `Trigger`: 양팔의 실제 현재 자세를 IK 목표로 지정 |
| 서비스 | `/ai_worker/pause` | `std_srvs/srv/SetBool`: true 정지 / false 재개 |

- 100 Hz 제어 / 500 Hz 물리, 상태 발행 목표 50 Hz. PC 성능에 따라 느려질 수 있습니다.
- ROS 콜백·GUI·물리는 같은 스레드에서 실행합니다. 타이머는 steady clock을 사용하므로 pause 중에도 서비스를 받습니다.
- GUI 지연 시 고정 시간 간격으로 최대 100 ms를 따라잡습니다. 그 이상의 지연은 누적하지 않고 보고서에 기록합니다.
- 목표는 도착 시 적용하고 유지합니다. timestamp 예약 실행·자동 만료는 없습니다. GUI와 ROS 입력은 같은 목표를 변경합니다.
- 관절 명령은 이름·범위·유한값을 전체 검사한 후 적용합니다. 해당 팔은 JOINT 모드로, 손 자세 명령을 받으면 IK 모드로 바뀝니다.
- 명령 QoS는 Reliable / Volatile / KeepLast(1). 무효한 명령은 거부하고 로그를 남깁니다.
- 하나의 ROS domain에 이 시뮬레이터와 `/clock` 발행자를 하나씩만 실행합니다. 소비 노드는 필요시 `use_sim_time:=true`로 설정합니다.
- ROS 연결이 끊겨도 마지막 위치 목표를 유지하는 **시뮬레이션용** 인터페이스입니다.

## 조작 예시

새 Ubuntu 터미널에서 `source /opt/ros/jazzy/setup.bash` 후 실행합니다.

```bash
ros2 topic echo /joint_states --once
ros2 topic pub --once /ai_worker/left/grasp std_msgs/msg/Float64 '{data: 0.4}'
ros2 service call /ai_worker/pause std_srvs/srv/SetBool '{data: true}'
ros2 service call /ai_worker/hold std_srvs/srv/Trigger '{}'
ros2 service call /ai_worker/reset std_srvs/srv/Trigger '{}'
ros2 service call /ai_worker/pause std_srvs/srv/SetBool '{data: false}'
```

## 검증

저장소 루트에서 실행합니다. 실제 ROS 검증은 별도 domain에서 자체 시뮬레이터와 테스트 노드를 실행합니다.

```bash
source /opt/ros/jazzy/setup.bash
source .venv_ros2/bin/activate
python validate_ros2_adapter.py
ROS_DOMAIN_ID=73 python validate_ros2_live.py
bash ros2/run.sh --gui --duration 10
```

`--ui-screenshot outputs/ros2_workspace.png`로 GUI를 저장할 수 있습니다. 실행 보고서 `outputs/ros2_run.json`에는 제어 횟수·시뮬레이션 시간·렌더링 프레임·최종 상태가 기록됩니다. 두 파일 모두 Git에서 제외합니다.

WSLg GUI에서는 X11/D3D12를 사용하고, NVIDIA GPU가 있으면 우선 선택합니다. 그림자·반사는 성능을 위해 끄며 충돌·물리는 그대로입니다. GPU 선택은 `MESA_D3D12_DEFAULT_ADAPTER_NAME`으로 변경할 수 있습니다. [WSLg GPU 설정](https://github.com/microsoft/wslg/wiki/GPU-selection-in-WSLg)

공식 설치 문서: [WSL](https://learn.microsoft.com/en-us/windows/wsl/install), [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).
