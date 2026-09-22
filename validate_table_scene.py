"""Validate the L-arm hammer pose, table, dynamic cans and fixed-base approach."""
import json

import glfw
import mujoco
import numpy as np

from ai_worker import ROOT, ensure_finite
from ik_control import INITIAL_THUMB_VALUE, SIDES, hammer_hand_rotation, load_ik_model, site_pose
from run_ik import Session
from table_scene import CAN_HEIGHT, CAN_MASS, CAN_POSITIONS, TABLE_TOP


def validate():
    model, data = load_ik_model()
    for side in SIDES:
        elbow = data.qpos[model.joint(f'arm_{side}_joint4').qposadr[0]]
        assert np.isclose(elbow, -np.pi / 2)
        assert np.allclose(site_pose(data, model.site(f'ik_palm_{side}').id).rotation, hammer_hand_rotation(side), atol=1e-12)
    session = Session()
    model, data = session.model, session.data
    for _ in range(1000):
        session.tick()
    cans = []
    for number, xy in enumerate(CAN_POSITIONS, start=1):
        body = model.body(f'pepsi_can_{number}')
        state = data.body(f'pepsi_can_{number}')
        axis_z = state.xmat.reshape(3, 3)[:, 2]
        tilt = float(np.rad2deg(np.arccos(np.clip(axis_z[2], -1, 1))))
        height_error = float(abs(state.xpos[2] - CAN_HEIGHT/2 - TABLE_TOP))
        assert np.linalg.norm(state.xpos[:2] - xy) < 0.001
        assert height_error < 0.001 and tilt < 0.1
        assert np.isclose(body.mass[0], CAN_MASS)
        assert model.joint(f'pepsi_can_{number}_free').type[0] == mujoco.mjtJoint.mjJNT_FREE
        cans.append({'name': body.name, 'position_m': state.xpos.tolist(), 'tilt_deg': tilt,
                     'bottom_to_table_error_m': height_error, 'mass_kg': float(body.mass[0])})
    errors = session.controller.errors(data)
    assert max(v['orientation_error_deg'] for v in errors.values()) < .1
    initial_hands = {}
    for side in SIDES:
        angles = np.array([data.qpos[model.joint(f'finger_{side}_joint{i}').qposadr[0]]
                           for i in range(1, 21)])
        assert np.max(np.abs(angles[4:])) < np.deg2rad(1), (side, angles)
        assert np.max(np.abs(angles[[0, 2, 3]])) < np.deg2rad(1)
        assert abs(abs(angles[1]) - np.pi / 2) < np.deg2rad(1)
        finger_axes = [data.body(f'finger_{side}_link{i}').xmat.reshape(3, 3)[:, 2]
                       for i in (8, 12, 16, 20)]
        thumb_axis = data.body(f'finger_{side}_link4').xmat.reshape(3, 3)[:, 1]
        assert min(float(a @ b) for a in finger_axes for b in finger_axes) > np.cos(np.deg2rad(1))
        angle = float(np.rad2deg(np.arccos(np.clip(abs(thumb_axis @ finger_axes[0]), 0, 1))))
        assert angle > 89, (side, angle)
        initial_hands[side] = {'four_finger_max_joint_angle_deg': float(np.rad2deg(abs(angles[4:]).max())),
                               'thumb_to_finger_angle_deg': angle,
                               'thumb_slider_percent': session.joint_controls.grasp_values[side]['thumb'] * 100}
    for side, target in session.controller.targets.items():
        assert np.array_equal(target.rotation, hammer_hand_rotation(side))
    # A free can must respond to an applied perturbation, not be a fixed prop.
    body_id = model.body('pepsi_can_1').id
    before = data.body('pepsi_can_1').xpos.copy()
    data.xfrc_applied[body_id, 0] = 3.0
    for _ in range(30):
        session.tick()
    data.xfrc_applied[body_id] = 0
    displacement = float(np.linalg.norm(data.body('pepsi_can_1').xpos - before))
    assert displacement > .005, displacement
    session.key(glfw.KEY_F11)
    assert np.array_equal(data.qpos, session.initial_qpos)
    assert np.array_equal(data.qvel, np.zeros(model.nv))
    for side in SIDES:
        assert np.array_equal(session.controller.targets[side].rotation, hammer_hand_rotation(side))
        assert session.joint_controls.grasp_values[side] == {'thumb': INITIAL_THUMB_VALUE, 'grasp': 0}
    # Exercise a smooth, open-hand approach beside both cans. This verifies
    # reachability without table contact; it is not a grasp/lift success test.
    starts = {s: p.position.copy() for s, p in session.controller.targets.items()}
    goals = {s: np.array([CAN_POSITIONS[i][0] - .05,
                         CAN_POSITIONS[i][1] + (.085 if s == 'l' else -.085),
                         TABLE_TOP + .12]) for i, s in enumerate(SIDES)}
    wheel_ids = [j for j in range(model.njnt) if 'wheel' in model.joint(j).name]
    wheel_qadr = [int(model.jnt_qposadr[j]) for j in wheel_ids]
    initial_wheels = data.qpos[wheel_qadr].copy()
    assert 'floating_base' not in [model.joint(j).name for j in range(model.njnt)]
    table_body = model.body('work_table').id
    can_bodies = {model.body(f'pepsi_can_{i}').id for i in (1, 2)}
    for tick in range(600):
        alpha = .5 - .5 * np.cos(np.pi * min((tick + 1) / 400, 1))
        for side in SIDES:
            session.controller.targets[side].position[:] = starts[side] + alpha * (goals[side] - starts[side])
        session.tick()
        for contact in data.contact:
            bodies = {int(model.geom_bodyid[g]) for g in contact.geom}
            if table_body in bodies:
                assert bodies - {table_body} <= can_bodies, 'Robot touched table during approach'
    approach_errors = session.controller.errors(data)
    assert max(v['position_error_mm'] for v in approach_errors.values()) < 1
    assert max(v['orientation_error_deg'] for v in approach_errors.values()) < .1
    assert np.max(np.abs(data.qpos[wheel_qadr] - initial_wheels)) < 1e-8
    ensure_finite(data)
    return {'status': 'PASS', 'initial_elbow_command_deg': -90,
            'initial_target_rpy_deg': {'l': [-90, 0, -90], 'r': [90, 0, 90]}, 'table_height_m': TABLE_TOP,
            'initial_hand_shapes': initial_hands,
            'cans_after_10_seconds': cans, 'hand_errors_after_10_seconds': errors,
            'can_push_displacement_m': displacement, 'reset_restores_scene': True,
            'robot_control_channels': model.nu, 'scene_free_joints': 2,
            'approach_targets_m': {s: p.tolist() for s, p in goals.items()},
            'approach_errors': approach_errors, 'approach_robot_table_contact': False,
            'base_fixed_and_wheels_unchanged': True,
            'grasp_and_lift_tested': False, 'warnings': data.warning.number.tolist()}


if __name__ == '__main__':
    report = validate()
    output = ROOT / 'outputs' / 'table_scene_validation.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
