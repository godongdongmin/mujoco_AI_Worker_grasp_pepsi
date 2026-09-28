"""Validate actual MuJoCo bridge semantics without requiring a ROS install."""
import unittest

import numpy as np

from ik_control import site_pose
from ros2_adapter import SessionAdapter
from run_ik import Session


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = Session()
        cls.bridge = SessionAdapter(cls.session)

    def setUp(self):
        self.bridge.reset()
        self.session.paused = False

    def test_snapshot_and_quaternion_roundtrip(self):
        state = self.bridge.snapshot()
        self.assertEqual(len(state['names']), 63)
        self.assertEqual(len(set(state['names'])), 63)
        self.assertEqual(len(state['velocities']), 63)
        for side in ('l', 'r'):
            expected = site_pose(self.session.data, self.session.controller.sites[side])
            position, quaternion = state['poses'][side]
            self.bridge.set_pose(side, 'world', position, quaternion)
            np.testing.assert_allclose(self.session.controller.targets[side].rotation,
                                       expected.rotation, atol=1e-12)

    def test_invalid_joint_batch_is_atomic(self):
        joints = self.session.joint_controls
        targets = joints.targets.copy()
        cases = [(['arm_l_joint1', 'missing'], [0.02, 0]),
                 (['arm_l_joint1', 'arm_l_joint1'], [0.02, 0.03]),
                 (['arm_l_joint1', 'arm_r_joint1'], [0.02, 100]),
                 (['arm_l_joint1'], [np.nan]), (['arm_l_joint1'], []), ([], [])]
        for names, values in cases:
            with self.subTest(names=names, values=values):
                with self.assertRaises(ValueError):
                    self.bridge.set_joints(names, values)
                np.testing.assert_array_equal(joints.targets, targets)
                self.assertEqual(joints.arm_modes, {'l': 'IK', 'r': 'IK'})
        with self.assertRaises(ValueError):
            self.bridge.set_joints(['arm_l_joint1'], [0], velocities=[1])

    def test_invalid_pose_keeps_previous_target(self):
        target = self.session.controller.targets['l'].copy()
        for frame, position, quaternion in [
                ('camera', [0, 0, 1], [0, 0, 0, 1]),
                ('', [0, 0, 1], [0, 0, 0, 1]),
                ('world', [0, np.nan, 1], [0, 0, 0, 1]),
                ('world', [0, 0, 1], [0, 0, 0, 0]),
                ('world', [0, 0, 1], [0, 0, 0, 2])]:
            with self.assertRaises(ValueError):
                self.bridge.set_pose('l', frame, position, quaternion)
            np.testing.assert_array_equal(self.session.controller.targets['l'].position, target.position)
            np.testing.assert_array_equal(self.session.controller.targets['l'].rotation, target.rotation)

    def test_joint_to_pose_mode_and_physical_tracking(self):
        self.bridge.set_joints(['arm_l_joint1'], [.01])
        self.assertEqual(self.session.joint_controls.arm_modes, {'l': 'JOINT', 'r': 'IK'})
        state = self.bridge.snapshot()
        position, quaternion = state['poses']['l']
        position[2] += .02
        self.bridge.set_pose('l', 'world', position, quaternion)
        self.assertEqual(self.session.joint_controls.arm_modes['l'], 'IK')
        for _ in range(300):
            self.session.tick()
        self.assertLess(self.session.controller.errors(self.session.data)['l']['position_error_mm'], 2)
        self.assertFalse(np.any(self.session.data.warning.number))

    def test_grasp_preserves_pinky_and_rejects_bad_values(self):
        self.bridge.set_joints(['finger_l_joint18'], [.2])
        self.bridge.set_grasp('l', 'grasp', .4)
        targets = self.session.joint_controls.targets.copy()
        self.assertEqual(targets[self.bridge.indices['finger_l_joint18']], .2)
        for value in (-.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                self.bridge.set_grasp('l', 'grasp', value)
        np.testing.assert_array_equal(self.session.joint_controls.targets, targets)

    def test_hold_and_reset_preserve_time(self):
        self.session.tick()
        self.session.tick()
        stamp = self.bridge.snapshot()['stamp']
        self.bridge.hold()
        for side in ('l', 'r'):
            np.testing.assert_allclose(self.session.controller.targets[side].position,
                                       site_pose(self.session.data, self.session.controller.sites[side]).position)
        self.session.paused = True
        self.bridge.reset()
        self.assertEqual(self.bridge.snapshot()['stamp'], stamp)
        self.assertTrue(self.session.paused)
        np.testing.assert_array_equal(self.session.data.qpos, self.session.initial_qpos)


if __name__ == '__main__':
    unittest.main(verbosity=2)
