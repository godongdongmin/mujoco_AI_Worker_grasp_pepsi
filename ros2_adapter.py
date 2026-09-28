"""ROS-independent command validation and state conversion for the SH5 session.

Call only from the simulation owner thread. ROS quaternions use x,y,z,w;
MuJoCo uses w,x,y,z. Targets persist until replaced, held, or reset.
"""
import glfw
import mujoco
import numpy as np

from ik_control import Pose, SIDES, site_pose


class SessionAdapter:
    def __init__(self, session):
        self.session = session
        self.names = [session.model.joint(int(j)).name
                      for j in session.joint_controls.joints]
        self.indices = {name: i for i, name in enumerate(self.names)}

    def set_pose(self, side, frame, position, quaternion_xyzw):
        if side not in SIDES or frame != 'world':
            raise ValueError('Pose targets require side l/r and frame_id world')
        position = np.asarray(position, dtype=float)
        quaternion = np.asarray(quaternion_xyzw, dtype=float)
        if (position.shape != (3,) or quaternion.shape != (4,)
                or not np.isfinite(position).all() or not np.isfinite(quaternion).all()):
            raise ValueError('Expected finite position[3] and quaternion[x,y,z,w]')
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-3, rtol=0):
            raise ValueError('Quaternion must have unit norm (tolerance 0.001)')
        quaternion = quaternion / norm
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion[[3, 0, 1, 2]])
        target = Pose(position.copy(), rotation.reshape(3, 3))
        target.validate()
        self.session.joint_controls.enable_ik(side)
        self.session.demo = False
        self.session.controller.targets[side] = target
        self.session.controller.update_markers(self.session.data)

    def set_joints(self, names, positions, velocities=(), efforts=()):
        names = list(names)
        values = np.asarray(positions, dtype=float)
        if (not names or len(set(names)) != len(names)
                or values.shape != (len(names),) or not np.isfinite(values).all()
                or len(velocities) or len(efforts)):
            raise ValueError('Joint command needs unique names and finite positions only')
        if any(name not in self.indices for name in names):
            raise ValueError('Unknown joint name')
        indices = [self.indices[name] for name in names]
        limits = self.session.joint_controls.limits[indices]
        if np.any(values < limits[:, 0]) or np.any(values > limits[:, 1]):
            raise ValueError('Joint command outside configured limits')
        # Validate the entire batch before changing any target or control mode.
        for index, value in zip(indices, values):
            self.session.joint_controls.set_joint(index, float(value))

    def set_grasp(self, side, component, value):
        self.session.joint_controls.set_grasp(side, component, value)

    def reset(self):
        self.session.key(glfw.KEY_F11)

    def hold(self):
        self.session.key(glfw.KEY_HOME)

    def snapshot(self):
        session = self.session
        joints = session.joint_controls.joints
        poses = {}
        for side in SIDES:
            pose = site_pose(session.data, session.controller.sites[side])
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, pose.rotation.ravel())
            poses[side] = (pose.position.tolist(), quat[[1, 2, 3, 0]].tolist())
        elapsed_ns = max(0, round((session.data.time - session.start_sim) * 1e9))
        return {
            'stamp': divmod(elapsed_ns, 1_000_000_000),
            'names': self.names.copy(),
            'positions': session.data.qpos[session.model.jnt_qposadr[joints]].tolist(),
            'velocities': session.data.qvel[session.model.jnt_dofadr[joints]].tolist(),
            # Effort intentionally omitted: actuator force excludes applied bias
            # feedforward and would misrepresent the total joint effort.
            'poses': poses,
        }
