"""Shared initialization and rendering helpers for the SH5 workspace."""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent


def neutral_controls(model):
    controls = np.zeros(model.nu)
    targets = {'arm_l_joint2': 0.18, 'arm_r_joint2': -0.18,
               'arm_l_joint4': -0.35, 'arm_r_joint4': -0.35}
    for name, value in targets.items():
        controls[model.actuator(name).id] = value
    return clamp_controls(model, controls)


def clamp_controls(model, controls):
    result = np.asarray(controls, dtype=float).copy()
    if result.shape != (model.nu,) or not np.isfinite(result).all():
        raise ValueError('Controls must be a finite vector with model.nu entries')
    limited = model.actuator_ctrllimited.astype(bool)
    result[limited] = np.clip(result[limited], model.actuator_ctrlrange[limited, 0],
                              model.actuator_ctrlrange[limited, 1])
    return result


def reset(model, data):
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = neutral_controls(model)
    # Set driven joint positions to the starting pose, leaving the floating base
    # and coupled gripper joints in the official model's default state.
    for i in range(model.nu):
        name = model.actuator(i).name
        if 'wheel_drive' not in name:
            joint_id = model.actuator_trnid[i, 0]
            data.qpos[model.jnt_qposadr[joint_id]] = data.ctrl[i]
    mujoco.mj_forward(model, data)


def camera_for(model):
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0, 0, 0.95]
    camera.distance = 2.8
    camera.azimuth = 135
    camera.elevation = -15
    return camera


def ensure_finite(data):
    if not all(np.isfinite(a).all() for a in (data.qpos, data.qvel, data.qacc)):
        raise RuntimeError('Non-finite physics state')
    if np.any(data.warning.number):
        raise RuntimeError(f'MuJoCo warning counters: {data.warning.number.tolist()}')
