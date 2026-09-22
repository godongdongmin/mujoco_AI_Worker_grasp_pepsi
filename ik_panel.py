"""Twelve Cartesian target sliders for the SH5 IK session.

Roll/pitch/yaw use radians and R = Rz(yaw) @ Ry(pitch) @ Rx(roll),
in world coordinates.
"""
from __future__ import annotations

import math
import numpy as np

from ik_control import Pose, SIDES, axis_rotation, site_pose

FIELDS = ('X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw')


def rpy_to_matrix(radians):
    angles = np.asarray(radians, dtype=float)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError('Roll, pitch and yaw must be three finite angles')
    roll, pitch, yaw = angles
    return axis_rotation(2, yaw) @ axis_rotation(1, pitch) @ axis_rotation(0, roll)


def matrix_to_rpy(rotation):
    """Principal ZYX Euler angles; at gimbal lock choose yaw=0 consistently."""
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-7:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw])


class PoseSliders:
    """Pose conversion independent of the widgets, also used by validation."""
    def __init__(self, session):
        self.session = session
        self.initial = {s: p.copy() for s, p in session.controller.targets.items()}
        self.values = {}
        self.last_pose = {}
        self.sync()

    def sync(self):
        for side in SIDES:
            pose = self.session.controller.targets[side]
            old = self.last_pose.get(side)
            if old is None or not np.array_equal(old.rotation, pose.rotation):
                angles = matrix_to_rpy(pose.rotation)
            else:
                angles = self.values[side][3:].copy()
            self.values[side] = np.r_[pose.position, angles]
            self.last_pose[side] = pose.copy()

    def set_value(self, side, index, value):
        if side not in SIDES or index not in range(6) or not math.isfinite(value):
            raise ValueError('Invalid slider target')
        if index >= 3:
            limit = np.pi / 2 if index == 4 else np.pi
            # The three-decimal UI rounds pi to 3.142. Snap only this half-step
            # display rounding to the exact endpoint; reject larger excesses.
            if abs(value) > limit + .0005:
                raise ValueError('Roll/yaw range: +/-pi rad; pitch: +/-pi/2 rad')
            value = float(np.clip(value, -limit, limit))
        self.session.joint_controls.enable_ik(side)
        # Capture keyboard/demo updates before modifying only the requested field.
        self.sync()
        self.values[side][index] = value
        previous = self.session.controller.targets[side]
        position = previous.position.copy()
        rotation = previous.rotation.copy()
        if index < 3:
            position[index] = value
        else:
            rotation = rpy_to_matrix(self.values[side][3:])
        pose = Pose(position, rotation)
        pose.validate()
        self.session.demo = False
        self.session.selected = side
        self.session.controller.targets[side] = pose
        self.last_pose[side] = pose.copy()

    def reset_hand(self, side):
        self.session.demo = False
        self.session.joint_controls.enable_ik(side)
        self.session.controller.targets[side] = self.initial[side].copy()
        self.sync()

    def hold_hand(self, side):
        self.session.demo = False
        self.session.joint_controls.enable_ik(side)
        controller = self.session.controller
        controller.targets[side] = site_pose(self.session.data, controller.sites[side])
        self.sync()
