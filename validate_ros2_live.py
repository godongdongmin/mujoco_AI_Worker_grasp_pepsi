"""Exercise real ROS 2 DDS topics/services against a local MuJoCo bridge.

Use an unused ROS_DOMAIN_ID and no other /clock publisher. Requires ROS 2.
"""
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_srvs.srv import SetBool, Trigger
from tf2_msgs.msg import TFMessage

from run_ik import Session
from run_ros2 import build_node_class


def main():
    rclpy.init()
    session = Session()
    bridge = build_node_class()(session)
    probe = rclpy.create_node('ai_worker_bridge_test')
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    executor.add_node(probe)
    received = {}
    subscriptions = [probe.create_subscription(kind, topic,
                     lambda msg, key=key: received.__setitem__(key, msg), 10)
                     for key, kind, topic in [
                         ('joints', JointState, 'joint_states'),
                         ('pose', PoseStamped, 'ai_worker/left/pose'),
                         ('clock', Clock, '/clock'), ('tf', TFMessage, '/tf')]]
    pose_pub = probe.create_publisher(PoseStamped, 'ai_worker/left/target_pose', 1)
    joint_pub = probe.create_publisher(JointState, 'ai_worker/joint_targets', 1)
    grasp_pub = probe.create_publisher(Float64, 'ai_worker/left/grasp', 1)
    pause = probe.create_client(SetBool, 'ai_worker/pause')
    reset = probe.create_client(Trigger, 'ai_worker/reset')
    hold = probe.create_client(Trigger, 'ai_worker/hold')

    def until(predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.02)
        assert predicate(), 'ROS condition timed out'

    def call(client, request):
        until(client.service_is_ready)
        result = client.call_async(request)
        until(result.done)
        assert result.result().success

    try:
        until(lambda: len(received) == 4 and all(p.get_subscription_count() > 0
              for p in (pose_pub, joint_pub, grasp_pub)))
        assert len(received['joints'].name) == 63
        assert len(received['joints'].effort) == 0
        assert received['pose'].header.frame_id == 'world'
        assert {t.child_frame_id for t in received['tf'].transforms} == {
            'ai_worker_left_hand', 'ai_worker_right_hand'}
        call(pause, SetBool.Request(data=True))
        paused_time = session.data.time
        cycles = session.cycles
        target = received['pose']
        target.pose.position.z += .02
        pose_pub.publish(target)
        until(lambda: abs(session.controller.targets['l'].position[2] - target.pose.position.z) < 1e-8)
        assert session.cycles == cycles and session.data.time == paused_time
        call(pause, SetBool.Request(data=False))
        until(lambda: session.data.time >= paused_time + 3)
        assert session.controller.errors(session.data)['l']['position_error_mm'] < 2
        call(pause, SetBool.Request(data=True))
        cmd = JointState(name=['arm_l_joint1'], position=[.02])
        joint_pub.publish(cmd)
        until(lambda: session.joint_controls.arm_modes['l'] == 'JOINT')
        assert session.joint_controls.arm_modes['r'] == 'IK'
        grasp_pub.publish(Float64(data=.4))
        until(lambda: session.joint_controls.grasp_values['l']['grasp'] == .4)
        call(hold, Trigger.Request())
        assert session.joint_controls.arm_modes['l'] == 'IK'
        before_reset = session.data.time
        call(reset, Trigger.Request())
        assert session.data.time == before_reset and session.paused
        assert session.joint_controls.grasp_values['l']['grasp'] == 0
        np.testing.assert_array_equal(session.data.qpos, session.initial_qpos)
        print('PASS: real DDS states/TF/clock, pose tracking, joint/grasp commands, pause/hold/reset')
    finally:
        executor.shutdown()
        probe.destroy_node()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
