"""Joint targets, per-arm control ownership, and simple SH5 hand synergies."""
import math

import mujoco
import numpy as np

from ik_control import INITIAL_THUMB_VALUE, SIDES, site_pose


def thumb_c_targets(side, value):
    """Open -> opposed support -> shallow C, mirrored for the official HX5.

    Keep CMC roll at zero: the thumb then bends in the same local X/Z plane
    as the index finger, rather than rolling upward/outward. First rotate MCP
    yaw to 90 degrees, then curl the distal joints gently around a cylinder.
    Values are joint-shape goals, not object-size or contact-force control.
    """
    def smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    opposition = smoothstep(value / .60)
    curl = smoothstep((value - .60) / .40)
    sign = 1 if side == 'l' else -1
    return sign * np.array([0.0, np.pi / 2 * opposition, -.35 * curl, -.40 * curl])


class JointTargets:
    def __init__(self, session):
        self.session = session
        model = session.model
        self.joints = model.actuator_trnid[:, 0].copy()
        robot_joints = {i for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE}
        if set(self.joints) != robot_joints or len(set(self.joints)) != model.nu:
            raise ValueError('Expected one actuator for each robot joint (free scene objects excluded)')
        self.qadr = model.jnt_qposadr[self.joints]
        self.wheels = np.array(['wheel_drive' in model.actuator(i).name for i in range(model.nu)])
        self.sliding = model.jnt_type[self.joints] == mujoco.mjtJoint.mjJNT_SLIDE
        self.limits = model.actuator_ctrlrange.copy()
        limited = model.jnt_limited[self.joints].astype(bool)
        for i in range(model.nu):
            if self.wheels[i]:
                self.limits[i] = [-np.pi, np.pi]  # UI position range for continuous wheels
            elif limited[i]:
                self.limits[i, 0] = max(self.limits[i, 0], model.jnt_range[self.joints[i], 0])
                self.limits[i, 1] = min(self.limits[i, 1], model.jnt_range[self.joints[i], 1])
        self.speeds = np.where(self.sliding, 0.08, 1.0)
        self.reset()

    def reset(self):
        data = self.session.data
        self.targets = data.ctrl.copy()
        self.targets[self.wheels] = data.qpos[self.qadr[self.wheels]]
        self.command = self.targets.copy()
        self.arm_modes = {s: 'IK' for s in SIDES}
        self.wheel_enabled = np.zeros(len(self.targets), dtype=bool)
        self.grasp_values = {s: {'thumb': INITIAL_THUMB_VALUE, 'grasp': 0.0} for s in SIDES}
        self.grasp_custom = {s: {'thumb': False, 'grasp': False} for s in SIDES}

    def enable_ik(self, side):
        if self.arm_modes[side] == 'IK':
            return
        controller, data = self.session.controller, self.session.data
        controller.command[side] = data.ctrl[controller.actuators[side]].copy()
        controller.targets[side] = site_pose(data, controller.sites[side])
        self.arm_modes[side] = 'IK'

    def set_joint(self, index, value):
        if index not in range(len(self.targets)) or not math.isfinite(value):
            raise ValueError('Invalid joint target')
        self.session.demo = False
        controller, data, model = self.session.controller, self.session.data, self.session.model
        for side in SIDES:
            acts = controller.actuators[side]
            if index in acts and self.arm_modes[side] == 'IK':
                self.targets[acts] = data.ctrl[acts]
                self.command[acts] = data.ctrl[acts]
                self.arm_modes[side] = 'JOINT'
            name = model.actuator(index).name
            if name.startswith(f'finger_{side}_joint'):
                number = int(name.rsplit('joint', 1)[1])
                if number <= 16:
                    self.grasp_custom[side]['thumb' if number <= 4 else 'grasp'] = True
        self.targets[index] = np.clip(value, *self.limits[index])
        if self.wheels[index]:
            self.wheel_enabled[index] = True

    def set_grasp(self, side, component, value):
        if side not in SIDES or component not in ('thumb', 'grasp') or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Hand synergy must be between 0 and 1')
        model = self.session.model
        # Thumb and three-finger presets remain independent. Little-finger
        # joints 17..20 retain their own targets, including manual edits.
        if component == 'thumb':
            numbers = range(1, 5)
            angles = thumb_c_targets(side, value)
        else:
            numbers = range(5, 17)
            angles = value * np.tile([0.0, 1.10, 0.90, 0.65], 3)
        for number, angle in zip(numbers, angles):
            index = model.actuator(f'finger_{side}_joint{number}').id
            self.targets[index] = np.clip(angle, *self.limits[index])
        self.grasp_values[side][component] = value
        self.grasp_custom[side][component] = False

    def prepare(self):
        """Build held controls first; IK subsequently replaces only active arms."""
        session = self.session
        active = np.ones(len(self.targets), dtype=bool)
        for side in SIDES:
            if self.arm_modes[side] == 'IK':
                acts = session.controller.actuators[side]
                active[acts] = False
                self.targets[acts] = session.data.ctrl[acts]
                self.command[acts] = session.data.ctrl[acts]
        positions = active & ~self.wheels
        delta = self.speeds * session.controller.dt
        self.command[positions] += np.clip(self.targets[positions] - self.command[positions], -delta[positions], delta[positions])
        controls = self.command.copy()
        controls[self.wheels] = 0
        # Official wheel actuators are velocity servos: an outer position loop
        # converts the joint-angle goal to a bounded velocity command.
        for index in np.flatnonzero(self.wheel_enabled):
            error = self.targets[index] - session.data.qpos[self.qadr[index]]
            controls[index] = np.clip(5.0 * error, -1.0, 1.0)
        session.controller.hold_controls[:] = controls

    def synchronize(self):
        session = self.session
        for side in SIDES:
            if self.arm_modes[side] == 'IK':
                acts = session.controller.actuators[side]
                self.targets[acts] = session.data.ctrl[acts]
            else:
                # In joint mode, pose sliders display the actual current pose.
                session.controller.targets[side] = site_pose(session.data, session.controller.sites[side])
        session.controller.update_markers(session.data)
