"""Check all 63 command routes, joint/IK ownership, and independent hand presets."""
import json

import glfw
import numpy as np

from ai_worker import ROOT, ensure_finite
from ik_control import SIDES
from ik_panel import PoseSliders
from run_ik import Session


def validate():
    session = Session()
    controls, model, data = session.joint_controls, session.model, session.data
    assert len(controls.targets) == 63
    assert len(set(controls.joints)) == 63
    original = controls.targets.copy()
    # Exercise every route with a small reachable change, without assuming all
    # contact-constrained joints can perfectly achieve their commanded angle.
    requested = {}
    for index in range(model.nu):
        goal = np.clip(original[index] + (-.015 if controls.sliding[index] else .025), *controls.limits[index])
        controls.set_joint(index, goal)
        requested[index] = goal
    assert controls.arm_modes == {'l': 'JOINT', 'r': 'JOINT'}
    for _ in range(250):
        session.tick()
    for index, goal in requested.items():
        assert np.isclose(controls.targets[index], goal)
        if not controls.wheels[index]:
            assert np.isclose(data.ctrl[index], goal), (index, data.ctrl[index], goal)
    wheel_errors = {model.actuator(i).name: float(abs(data.qpos[controls.qadr[i]] - controls.targets[i])) for i in np.flatnonzero(controls.wheels)}
    assert max(wheel_errors.values()) < .02, wheel_errors
    # Switching back to IK only transfers the requested arm; other arm stays manual.
    sliders = PoseSliders(session)
    sliders.set_value('l', 0, sliders.values['l'][0] + .005)
    assert controls.arm_modes == {'l': 'IK', 'r': 'JOINT'}
    right_goal = controls.targets[session.controller.actuators['r']].copy()
    for _ in range(200):
        session.tick()
    assert np.array_equal(right_goal, data.ctrl[session.controller.actuators['r']])
    # Start from neutral to test thumb / three-finger groups and pinky exclusion.
    session.key(glfw.KEY_F11)
    grasp_motion = {}
    for side in SIDES:
        thumb = [model.actuator(f'finger_{side}_joint{i}').id for i in range(1, 5)]
        fingers = [model.actuator(f'finger_{side}_joint{i}').id for i in range(5, 17)]
        pinky = [model.actuator(f'finger_{side}_joint{i}').id for i in range(17, 21)]
        pinky_goal = np.array([0, .25, .20, .15])
        for index, value in zip(pinky, pinky_goal):
            controls.set_joint(index, value)
        assert not controls.grasp_custom[side]['grasp']
        before = controls.targets.copy()
        controls.set_grasp(side, 'thumb', .5)
        assert np.array_equal(controls.targets[fingers], before[fingers])
        other = 'r' if side == 'l' else 'l'
        other_acts = [model.actuator(f'finger_{other}_joint{i}').id for i in range(1, 21)]
        assert np.array_equal(controls.targets[other_acts], before[other_acts])
        thumb_goal = controls.targets[thumb].copy()
        controls.set_grasp(side, 'grasp', .6)
        assert np.array_equal(controls.targets[thumb], thumb_goal)
        assert np.array_equal(controls.targets[pinky], pinky_goal)
        q0 = data.qpos.copy()
        for _ in range(200):
            session.tick()
        finger_delta = float(np.max(np.abs(data.qpos[controls.qadr[fingers]] - q0[controls.qadr[fingers]])))
        assert finger_delta > .1, (side, finger_delta)
        assert controls.arm_modes == {'l': 'IK', 'r': 'IK'}
        assert np.array_equal(data.ctrl[pinky], pinky_goal)
        grasp_motion[side] = {'max_three_finger_motion_rad': finger_delta,
                              'manual_pinky_target_preserved': True}
        # Direct finger edits mark that synergy custom; preset restores ownership.
        controls.set_joint(fingers[1], .1)
        assert controls.grasp_custom[side]['grasp']
        controls.set_grasp(side, 'grasp', 0)
        assert not controls.grasp_custom[side]['grasp']
        for value in (1, .5, 0):
            controls.set_grasp(side, 'grasp', value)
            assert np.array_equal(controls.targets[pinky], pinky_goal)
    ensure_finite(data)
    # Range clamping, reset and finite-input rejection.
    index = model.actuator('head_joint2').id
    controls.set_joint(index, 100)
    assert controls.targets[index] == controls.limits[index, 1]
    for setter in [lambda: controls.set_joint(index, np.nan), lambda: controls.set_grasp('l', 'thumb', np.inf)]:
        try:
            setter()
            raise AssertionError('Invalid input accepted')
        except ValueError:
            pass
    session.key(glfw.KEY_F11)
    assert controls.arm_modes == {'l': 'IK', 'r': 'IK'}
    assert not any(controls.wheel_enabled)
    return {'status': 'PASS', 'joint_channels': 63, 'all_joint_targets_persist': True,
            'wheel_position_errors_rad': wheel_errors, 'grasp_motion': grasp_motion,
            'independent_arm_modes': 'PASS', 'hand_group_independence': 'PASS',
            'range_reset_invalid_input': 'PASS', 'warnings': data.warning.number.tolist()}


if __name__ == '__main__':
    report = validate()
    path = ROOT / 'outputs' / 'joint_control_validation.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
