"""Fixed-base SH5 Cartesian control using MuJoCo Jacobians and damped IK.

World-frame poses; quaternion convention is w,x,y,z. Only the 14 arm joints
are optimized. This local IK solver does NOT provide collision avoidance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from ai_worker import ROOT, ensure_finite, reset

SIDES = ('l', 'r')
INITIAL_THUMB_VALUE = 0.60  # opposed, straight support before C-shape curling
PALM_OFFSET = np.array([0.0, 0.0, 0.06])  # hand-base frame, metres


def hammer_hand_rotation(side):
    """Thumb-side up, extended fingers forward, palms facing one another.

    The hand's +X is the finger-flexion/palm side and +Z is finger length.
    Thumb roots lie on local -Y on the left and +Y on the right.
    """
    if side not in SIDES:
        raise ValueError('Expected l or r hand')
    sign = 1 if side == 'l' else -1
    return np.array([[0., 0., 1.], [-sign, 0., 0.], [0., -sign, 0.]])


def rotation_vector(target, current):
    """SO(3) shortest rotation from current to target, expressed in world axes."""
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(target @ current.T).ravel())
    if quat[0] < 0:
        quat *= -1
    result = np.empty(3)
    mujoco.mju_quat2Vel(result, quat, 1.0)
    return result


def axis_rotation(axis, angle):
    quat = np.empty(4)
    mujoco.mju_axisAngle2Quat(quat, np.eye(3)[axis], angle)
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, quat)
    return result.reshape(3, 3)


@dataclass
class Pose:
    position: np.ndarray
    rotation: np.ndarray

    def copy(self):
        return Pose(self.position.copy(), self.rotation.copy())

    def validate(self):
        if self.position.shape != (3,) or self.rotation.shape != (3, 3):
            raise ValueError('Expected position (3,) and rotation (3,3)')
        if not np.isfinite(self.position).all() or not np.isfinite(self.rotation).all():
            raise ValueError('Target pose must be finite')
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(self.rotation), 1, atol=1e-6):
            raise ValueError('Target orientation must be a proper rotation matrix')


def site_pose(data, site_id):
    return Pose(data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy())


def pose_error(target, actual):
    return np.r_[target.position - actual.position,
                 rotation_vector(target.rotation, actual.rotation)]


def load_ik_model():
    """Derive a fixed-base model in memory, preserving the vendor XML files."""
    spec = mujoco.MjSpec.from_file(str(ROOT / 'scenes' / 'ai_worker_sh5.xml'))
    spec.delete(spec.joint('floating_base'))
    from table_scene import add_table_and_cans
    add_table_and_cans(spec)
    for side, color in [('l', [0.1, 0.85, 1, 1]), ('r', [1, 0.55, 0.1, 1])]:
        # The palm's convex collision hull fills the finger mounting recesses.
        # At straight zero angles it overlaps these proximal links (up to
        # ~9.6 mm), artificially splaying the fingers. Exclude only these four
        # articulated mounting pairs; finger/finger and object contacts remain.
        for link in (6, 10, 14, 18):
            spec.add_exclude(name=f'palm_mount_{side}_{link}',
                             bodyname1=f'hx5_{side}_base', bodyname2=f'finger_{side}_link{link}')
        spec.body(f'hx5_{side}_base').add_site(
            name=f'ik_palm_{side}', pos=PALM_OFFSET, size=[0.009, 0, 0], rgba=color)
        marker = spec.worldbody.add_body(name=f'ik_target_{side}', mocap=True)
        marker.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0],
                        rgba=color, contype=0, conaffinity=0)
        for axis, rgb in enumerate(np.eye(3)):
            marker.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                            size=[0.002, 0, 0], fromto=np.r_[np.zeros(3), 0.08 * rgb],
                            rgba=np.r_[rgb, 0.8], contype=0, conaffinity=0)
    model = spec.compile()
    data = mujoco.MjData(model)
    reset(model, data)
    # L-shaped arms with a neutral hammer grip: thumbs up and palms inward.
    lift = model.actuator('lift_joint').id
    data.ctrl[lift] = -0.35
    data.qpos[model.joint('lift_joint').qposadr[0]] = -0.35
    for side, sign in [('l', 1), ('r', -1)]:
        # Four straight, parallel fingers and a straight opposed thumb (60%
        # on the existing Thumb slider). Preserve the original hand geometry.
        for number in range(1, 21):
            name = f'finger_{side}_joint{number}'
            angle = sign * np.pi / 2 if number == 2 else 0.0
            data.ctrl[model.actuator(name).id] = angle
            data.qpos[model.joint(name).qposadr[0]] = angle
        # Obtain the official hand-base rotation at zero arm joint angles.
        for number in range(1, 8):
            data.qpos[model.joint(f'arm_{side}_joint{number}').qposadr[0]] = 0
        mujoco.mj_forward(model, data)
        hand_zero = data.site_xmat[model.site(f'ik_palm_{side}').id].reshape(3, 3).copy()
        q2, q3 = sign * 0.20, sign * 0.40
        wrist_rotation = axis_rotation(1, np.pi/2) @ axis_rotation(2, -q3) @ axis_rotation(0, -q2) @ hammer_hand_rotation(side) @ hand_zero.T
        q6 = np.arcsin(np.clip(-wrist_rotation[2, 0], -1, 1))
        q7 = np.arctan2(wrist_rotation[2, 1], wrist_rotation[2, 2])
        q5 = np.arctan2(wrist_rotation[1, 0], wrist_rotation[0, 0])
        for number, angle in enumerate([0, q2, q3, -np.pi/2, q5, q6, q7], start=1):
            name = f'arm_{side}_joint{number}'
            data.ctrl[model.actuator(name).id] = angle
            data.qpos[model.joint(name).qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    return model, data


class DualArmIK:
    def __init__(self, model, data, *, control_hz=100.0, max_joint_speed=1.0):
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError('control_hz must be finite and positive')
        if not math.isfinite(max_joint_speed) or max_joint_speed <= 0:
            raise ValueError('max_joint_speed must be finite and positive')
        self.model = model
        self.dt = 1.0 / control_hz
        self.max_joint_speed = max_joint_speed  # command slew limit, rad/s
        self.work = mujoco.MjData(model)  # IK never overwrites the physical qpos
        self.sites = {s: model.site(f'ik_palm_{s}').id for s in SIDES}
        self.actuators, self.qadr, self.dofs, self.bounds = {}, {}, {}, {}
        for side in SIDES:
            names = [f'arm_{side}_joint{i}' for i in range(1, 8)]
            joints = np.array([model.joint(n).id for n in names])
            acts = np.array([model.actuator(n).id for n in names])
            self.actuators[side] = acts
            self.qadr[side] = model.jnt_qposadr[joints]
            self.dofs[side] = model.jnt_dofadr[joints]
            self.bounds[side] = np.column_stack((
                np.maximum(model.jnt_range[joints, 0], model.actuator_ctrlrange[acts, 0]) + 1e-5,
                np.minimum(model.jnt_range[joints, 1], model.actuator_ctrlrange[acts, 1]) - 1e-5))
        self.all_dofs = np.concatenate(list(self.dofs.values()))
        self.jacp, self.jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        self.weights = np.array([1, 1, 1, 0.25, 0.25, 0.25])
        self.damping = 0.015
        self.iterations = 12
        self.reset(data)

    def reset(self, data):
        mujoco.mj_forward(self.model, data)
        self.command = {s: data.qpos[self.qadr[s]].copy() for s in SIDES}
        self.home = {s: site_pose(data, self.sites[s]) for s in SIDES}
        self.targets = {s: p.copy() for s, p in self.home.items()}
        self.hold_controls = data.ctrl.copy()
        self.last_result = {}
        self.update_markers(data)

    def update_markers(self, data):
        for side, pose in self.targets.items():
            index = self.model.body(f'ik_target_{side}').mocapid[0]
            data.mocap_pos[index] = pose.position
            mujoco.mju_mat2Quat(data.mocap_quat[index], pose.rotation.ravel())

    def _kinematics(self):
        mujoco.mj_kinematics(self.model, self.work)
        mujoco.mj_comPos(self.model, self.work)

    def solve(self, data, active_sides=SIDES):
        """Bounded local DLS, warm-started from the previous joint command.

        Position and orientation are solved together. Every inner iteration is
        bounded by the SAME per-control-tick box, so iterations cannot bypass
        the commanded joint-speed limit. Unreachable goals retain a residual.
        """
        for target in self.targets.values():
            target.validate()
        self.work.qpos[:] = data.qpos
        results = {}
        for side in active_sides:
            qadr, dofs = self.qadr[side], self.dofs[side]
            previous = self.command[side]
            bounds = self.bounds[side]
            delta = self.max_joint_speed * self.dt
            lower = np.maximum(bounds[:, 0], previous - delta)
            upper = np.minimum(bounds[:, 1], previous + delta)
            q = np.clip(previous, lower, upper)
            self.work.qpos[qadr] = q
            self._kinematics()
            target = self.targets[side]
            for _ in range(self.iterations):
                error = pose_error(target, site_pose(self.work, self.sites[side]))
                if np.linalg.norm(error[:3]) < 2e-5 and np.linalg.norm(error[3:]) < 2e-4:
                    break
                mujoco.mj_jacSite(self.model, self.work, self.jacp, self.jacr, self.sites[side])
                jac = np.vstack((self.jacp[:, dofs], self.jacr[:, dofs])) * self.weights[:, None]
                weighted_error = self.weights * error
                # Re-solve with outward directions at physical joint limits
                # blocked. Rate-limit faces only end this tick's progress;
                # they must not distort the joint-space search direction.
                free = np.ones(len(q), dtype=bool)
                dq = np.zeros_like(q)
                for _ in range(len(q) + 1):
                    reduced = jac[:, free]
                    dq[:] = 0
                    dq[free] = reduced.T @ np.linalg.solve(
                        reduced @ reduced.T + self.damping**2 * np.eye(6), weighted_error)
                    outward = free & (((q <= bounds[:, 0] + 1e-10) & (dq < 0)) |
                                      ((q >= bounds[:, 1] - 1e-10) & (dq > 0)))
                    if not np.any(outward):
                        break
                    free[outward] = False
                # One common scalar preserves all dq component ratios. Each
                # inner iteration uses the original per-tick bounds, so the
                # accumulated update still respects max_joint_speed.
                positive, negative = dq > 0, dq < 0
                step_scale = min(1.0,
                                 float(np.min((upper[positive] - q[positive]) / dq[positive], initial=1.0)),
                                 float(np.min((lower[negative] - q[negative]) / dq[negative], initial=1.0)))
                if step_scale <= 0:
                    break
                # Backtracking accepts only an improvement in weighted SE(3) error.
                accepted = False
                for scale in (1.0, 0.5, 0.25, 0.125):
                    # Clip only roundoff after uniform scaling, not individual
                    # components of an oversized DLS update.
                    candidate = np.clip(q + scale * step_scale * dq, lower, upper)
                    self.work.qpos[qadr] = candidate
                    self._kinematics()
                    next_error = self.weights * pose_error(target, site_pose(self.work, self.sites[side]))
                    if np.dot(next_error, next_error) < np.dot(weighted_error, weighted_error) - 1e-14:
                        q, accepted = candidate, True
                        break
                if not accepted:
                    self.work.qpos[qadr] = q
                    self._kinematics()
                    break
            self.command[side] = q.copy()
            residual = pose_error(target, site_pose(self.work, self.sites[side]))
            results[side] = {'position_error_m': float(np.linalg.norm(residual[:3])),
                             'orientation_error_deg': float(np.rad2deg(np.linalg.norm(residual[3:]))),
                             'command_speed_rad_s': float(np.max(np.abs(q - previous)) / self.dt),
                             'converged': bool(np.linalg.norm(residual[:3]) < 0.001 and np.linalg.norm(residual[3:]) < np.deg2rad(0.5))}
        data.ctrl[:] = self.hold_controls
        for side in active_sides:
            data.ctrl[self.actuators[side]] = self.command[side]
        self.last_result = results
        self.update_markers(data)
        return results

    def step_physics(self, data, steps):
        for _ in range(steps):
            # Simulation-only model-based bias feedforward (gravity + Coriolis).
            # Applied only to the arm DoFs, outside official actuator force caps.
            data.qfrc_applied[self.all_dofs] = data.qfrc_bias[self.all_dofs]
            mujoco.mj_step(self.model, data)
        mujoco.mj_forward(self.model, data)
        ensure_finite(data)

    def errors(self, data):
        result = {}
        for side in SIDES:
            error = pose_error(self.targets[side], site_pose(data, self.sites[side]))
            result[side] = {'position_error_mm': float(1000 * np.linalg.norm(error[:3])),
                            'orientation_error_deg': float(np.rad2deg(np.linalg.norm(error[3:])))}
        return result

    def set_demo(self, elapsed):
        # Independent world-frame Cartesian trajectories, not joint-space replay.
        phase = 2 * np.pi * elapsed / 8.0
        blend = 0.5 - 0.5 * np.cos(phase)
        for side, sign in [('l', 1), ('r', -1)]:
            home = self.home[side]
            position = home.position + np.array([0.05 * blend, sign * 0.035 * np.sin(phase), 0.045 * blend])
            rotation = axis_rotation(2, sign * np.deg2rad(12) * blend) @ axis_rotation(1, np.deg2rad(8) * np.sin(phase)) @ home.rotation
            self.targets[side] = Pose(position, rotation)
