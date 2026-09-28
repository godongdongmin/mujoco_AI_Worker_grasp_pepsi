# MuJoCo AI Worker — Pepsi workspace

ROBOTIS FFW-SH5 양팔·다지 손을 조작하는 고정 베이스 시뮬레이터.

`main`: 기존 Windows MuJoCo 앱. `feature/ros2-integration`: [ROS 2 연동](ros2/README.md) 개발 브랜치.

## 주요 기능

- Qt 통합 창: 중앙 3D 화면, 좌우 손 제어, 하단 63개 관절 탭
- 양손 위치·자세 IK, Thumb / Grasp 슬라이더
- 탁자와 물리 접촉이 적용된 펩시 캔 2개
- 위치 **m**, 회전 **rad**, 손 프리셋 **%**
- 제어 100 Hz / 물리 500 Hz 기준. 실제 속도는 PC 성능에 따라 달라짐

## 설치·실행

Windows x64 · Python 3.12 · Git · OpenGL 지원 GPU가 필요합니다.

```powershell
git clone https://github.com/godongdongmin/mujoco_AI_Worker_grasp_pepsi.git
cd mujoco_AI_Worker_grasp_pepsi
.\setup.ps1
.\run_ik.cmd
```

`setup.ps1`은 `.venv`에 패키지를 설치하고, 고정된 공식 모델 버전을 `vendor/`에 내려받은 뒤 검증합니다. 두 폴더는 Git에 포함하지 않습니다. 자동 동작 데모는 `run_ik_demo.cmd`로 실행합니다.

## 조작

| 항목 | 기능 |
|---|---|
| 좌우 X / Y / Z · Roll / Pitch / Yaw | 월드 기준 절대 목표 입력. 숫자 직접 입력 가능 |
| Thumb | 0–60%: 직각으로 펼침, 60–100%: C자 굽힘 |
| Grasp | 검지·중지·약지 제어. 새끼손가락은 Joints에서 별도 조절 |
| Use current pose / Hold both | 실제 현재 자세를 새 목표로 지정 |
| Reset robot / F11 | 로봇·캔·목표 초기화. Thumb 60%, Grasp 0% |
| Pause / F12 · Demo / F10 | 일시정지 · 자동 데모 전환 |
| Left / Right / Joints | 패널 표시·숨기기. 경계 드래그로 크기 조절 |
| 마우스 왼쪽 / 오른쪽 드래그 · 휠 | 카메라 회전 / 이동 · 줌 |

중앙 화면에서 F8은 좌우 손 선택, F9는 위치/회전 전환입니다. ↑/↓는 X, ←/→는 Y, Insert/Delete는 Z를 조절합니다(한 번에 0.01 m 또는 0.05 rad).

## 검증·출력

```powershell
.\.venv\Scripts\python.exe validate_ik.py
.\.venv\Scripts\python.exe validate_ik_panel.py --integrated
.\.venv\Scripts\python.exe run_ik.py --headless --duration 5
```

검증 결과·실행 보고서는 `outputs/`에 자동 생성되며 Git에서는 제외합니다. GUI 캡처는 `--duration 5 --ui-screenshot outputs/workspace.png`, 장면 이미지는 `--render outputs/scene.png`로 저장합니다.

## 범위·출처

- 자동 파지·들어올리기, 충돌 회피 경로 계획, 실물 로봇 제어는 미구현입니다.
- IK는 관절 **명령 변화율**을 제한합니다. 실물 속도·토크 보장을 뜻하지 않습니다.
- [ROBOTIS 공식 모델](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie): Apache-2.0. 버전과 런타임 보정은 [SOURCES.json](SOURCES.json)에 기록합니다.
- [펩시 로고 출처](assets/pepsi/SOURCE.md): 상표·이미지 권리는 해당 소유자에게 있습니다.
