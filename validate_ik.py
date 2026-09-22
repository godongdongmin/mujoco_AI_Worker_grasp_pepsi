"""Numerical Jacobian checks, reachable/unreachable IK, and physics tracking."""
import json
import time

import glfw
import mujoco
import numpy as np

from ai_worker import ROOT, ensure_finite
from ik_control import DualArmIK, Pose, SIDES, axis_rotation, load_ik_model, rotation_vector, site_pose
from run_ik import Session


def check_jacobians():
    model, data = load_ik_model()
    controller = DualArmIK(model, data)
    result = {}
    for side in SIDES:
        site = controller.sites[side]
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site)
        original = data.qpos.copy()
        columns = []
        eps = 1e-6
        for address in controller.qadr[side]:
            data.qpos[:] = original
            data.qpos[address] += eps
            mujoco.mj_forward(model, data)
            plus = site_pose(data, site)
            data.qpos[address] -= 2 * eps
            mujoco.mj_forward(model, data)
            minus = site_pose(data, site)
            columns.append(np.r_[(plus.position - minus.position) / (2 * eps),
                                  rotation_vector(plus.rotation, minus.rotation) / (2 * eps)])
        numerical = np.array(columns).T
        analytical = np.vstack((jp[:, controller.dofs[side]], jr[:, controller.dofs[side]]))
        error = float(np.max(np.abs(numerical - analytical)))
        assert error < 1e-6, (side, error)
        result[side] = error
        data.qpos[:] = original
        mujoco.mj_forward(model, data)
    # The log error must remain valid at 180 degrees, not collapse to zero.
    assert abs(np.linalg.norm(rotation_vector(axis_rotation(0, np.pi), np.eye(3))) - np.pi) < 1e-10
    return result


def check_kinematic_cases():
    model, data = load_ik_model()
    rng = np.random.default_rng(22)
    initial = data.qpos.copy()
    generated = mujoco.MjData(model)
    samples = []
    for case in range(12):
        data.qpos[:] = initial
        mujoco.mj_forward(model, data)
        controller = DualArmIK(model, data)
        generated.qpos[:] = initial
        for side in SIDES:
            delta = rng.uniform(-1, 1, 7) * np.array([0.2, 0.12, 0.15, 0.25, 0.15, 0.2, 0.15])
            # Sample only valid joint angles, including when the new initial
            # wrist pose is near a joint limit.
            generated.qpos[controller.qadr[side]] = np.clip(
                generated.qpos[controller.qadr[side]] + delta,
                controller.bounds[side][:, 0], controller.bounds[side][:, 1])
        mujoco.mj_forward(model, generated)
        controller.targets = {s: site_pose(generated, controller.sites[s]) for s in SIDES}
        for tick in range(250):
            controller.solve(data)
            for side in SIDES:
                data.qpos[controller.qadr[side]] = controller.command[side]
            mujoco.mj_forward(model, data)
            if all(v['converged'] for v in controller.last_result.values()):
                break
        error = controller.errors(data)
        for values in error.values():
            assert values['position_error_mm'] < 1.0, (case, error)
            assert values['orientation_error_deg'] < 0.5, (case, error)
        samples.append({'case': case, 'ticks': tick + 1, 'errors': error})
    return samples


def check_physics():
    session = Session()
    controller, data = session.controller, session.data
    initial_base = data.body('base_link').xpos.copy()
    non_arm = np.ones(session.model.nu, dtype=bool)
    for side in SIDES:
        non_arm[controller.actuators[side]] = False
    original_non_arm = data.ctrl[non_arm].copy()
    # Static target changes both translation and orientation on both hands.
    for side, sign in [('l', 1), ('r', -1)]:
        target = controller.targets[side]
        target.position += [0.03, sign * 0.025, 0.04]
        target.rotation = axis_rotation(0, sign * np.deg2rad(10)) @ axis_rotation(2, np.deg2rad(8)) @ target.rotation
    for _ in range(400):
        session.tick()
        assert np.array_equal(data.ctrl[non_arm], original_non_arm)
    static = controller.errors(data)
    for error in static.values():
        assert error['position_error_mm'] < 2.0, static
        assert error['orientation_error_deg'] < 0.5, static
    # Return to reset state, then test independent, moving Cartesian targets.
    session.key(glfw.KEY_F11)
    session.demo = True
    session.demo_start = data.time
    start_sample = len(session.error_samples)
    for _ in range(800):
        session.tick()
    dynamic = {s: {key: max(e[s][key] for e in session.error_samples[start_sample:])
                   for key in static[s]} for s in SIDES}
    for error in dynamic.values():
        assert error['position_error_mm'] < 12.0, dynamic
        assert error['orientation_error_deg'] < 1.5, dynamic
    assert session.peak_speed <= controller.max_joint_speed + 1e-9
    assert np.array_equal(initial_base, data.body('base_link').xpos)
    ensure_finite(data)
    return {'static_after_4s': static, 'dynamic_8s_peak': dynamic,
            'peak_command_speed_rad_s': session.peak_speed,
            'ik_ms_mean': float(np.mean(session.solve_times) * 1000),
            'ik_ms_p95': float(np.percentile(session.solve_times, 95) * 1000),
            'base_fixed': True, 'non_arm_commands_unchanged': True,
            'warnings': data.warning.number.tolist()}


def check_edge_cases():
    session = Session()
    controller, data = session.controller, session.data
    controller.targets['l'].position += [5, 0, 2]
    for _ in range(200):
        session.tick()
        for side in SIDES:
            assert np.all(controller.command[side] >= controller.bounds[side][:, 0])
            assert np.all(controller.command[side] <= controller.bounds[side][:, 1])
    residual = controller.errors(data)
    assert residual['l']['position_error_mm'] > 1000
    assert not controller.last_result['l']['converged']
    assert session.peak_speed <= 1 + 1e-9
    assert residual['r']['position_error_mm'] < 2, residual
    # Invalid targets must be rejected before any command changes.
    previous = data.ctrl.copy()
    controller.targets['l'].position[0] = np.nan
    try:
        controller.solve(data)
        raise AssertionError('NaN target was not rejected')
    except ValueError:
        pass
    assert np.array_equal(data.ctrl, previous)
    # Near-straight arms: damped solve remains finite with limited commands.
    session.key(glfw.KEY_F11)
    for side in SIDES:
        data.qpos[controller.qadr[side]] = 0
    mujoco.mj_forward(session.model, data)
    controller.reset(data)
    for side in SIDES:
        controller.targets[side].position += [0.01, 0, 0.01]
    controller.solve(data)
    assert np.isfinite(data.ctrl).all()
    return {'unreachable_target_residual': residual, 'bounds_and_slew': 'PASS',
            'other_hand_hold': 'PASS', 'nan_rejection': 'PASS', 'near_singular': 'PASS'}


def check_keyboard():
    session = Session()
    controller = session.controller
    left = controller.targets['l'].copy()
    right = controller.targets['r'].copy()
    session.key(glfw.KEY_F8)
    session.key(glfw.KEY_UP)
    session.key(glfw.KEY_F9)
    session.key(glfw.KEY_UP)
    assert np.array_equal(controller.targets['l'].position, left.position)
    assert np.isclose(controller.targets['r'].position[0] - right.position[0], 0.01)
    assert np.isclose(np.linalg.norm(rotation_vector(controller.targets['r'].rotation, right.rotation)), .05)
    session.key(glfw.KEY_F12)
    assert session.paused
    session.key(glfw.KEY_F12)
    assert not session.paused
    session.key(glfw.KEY_F10)
    assert session.demo
    session.key(glfw.KEY_HOME)
    assert not session.demo
    session.key(glfw.KEY_F11)
    return 'PASS'


def check_table_release():
    """Regression: a large upward goal must recover after pressing on the table."""
    from ik_panel import PoseSliders
    reports = []
    for grasp in (0, .5):
        session = Session()
        model, data, controller = session.model, session.data, session.controller
        sliders = PoseSliders(session)
        base_start = data.body('base_link').xpos.copy()
        for side, sign in [('l', 1), ('r', -1)]:
            sliders.set_value(side, 0, .49)
            sliders.set_value(side, 1, sign * .345)
            session.joint_controls.set_grasp(side, 'thumb', 1 if grasp else 0)
            session.joint_controls.set_grasp(side, 'grasp', grasp)
        pressed = False
        # Parallel initial fingers clear the table lower than the old splayed
        # pose. Press to .74 m so this remains an actual contact-release test.
        for z in (.84, .79, .74, .90):
            for side in SIDES:
                sliders.set_value(side, 2, z)
            for _ in range(400):
                session.tick()
                for side in SIDES:
                    assert np.all(controller.command[side] >= controller.bounds[side][:, 0] - 1e-12)
                    assert np.all(controller.command[side] <= controller.bounds[side][:, 1] + 1e-12)
            if z == .74:
                pressed = any('work_table' in [model.body(model.geom_bodyid[g]).name for g in contact.geom]
                              and any(model.body(model.geom_bodyid[g]).name.startswith(('finger_', 'hx5_', 'arm_'))
                                      for g in contact.geom) for contact in data.contact)
        assert pressed, 'Test must actually contact the table before lifting'
        errors = controller.errors(data)
        for side in SIDES:
            assert errors[side]['position_error_mm'] < 2, (grasp, side, errors)
            assert errors[side]['orientation_error_deg'] < .5, errors
            assert controller.last_result[side]['converged']
        assert session.peak_speed <= controller.max_joint_speed + 1e-9
        assert np.array_equal(base_start, data.body('base_link').xpos)
        assert not np.any(data.warning.number)
        reports.append({'grasp': grasp, 'table_contact_before_lift': pressed,
                        'lift_target_z_m': .90, 'final_errors': errors,
                        'peak_command_speed_rad_s': session.peak_speed})
    return reports


if __name__ == '__main__':
    started = time.perf_counter()
    report = {'status': 'PASS', 'mujoco_version': mujoco.__version__,
              'jacobian_max_absolute_error': check_jacobians(),
              'reachable_pose_cases': check_kinematic_cases(),
              'physics': check_physics(), 'edge_cases': check_edge_cases(),
              'keyboard_commands': check_keyboard(), 'table_release': check_table_release()}
    report['test_wall_seconds'] = time.perf_counter() - started
    destination = ROOT / 'outputs' / 'ik_validation.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
