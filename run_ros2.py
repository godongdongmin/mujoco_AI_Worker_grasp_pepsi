"""Optional ROS 2 bridge. Source ROS 2, then run this file with that Python env."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def build_node_class():
    # Lazy imports keep the original Windows runner independent of ROS 2.
    import rclpy
    from rclpy.clock import Clock as TimerClock, ClockType
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64
    from std_srvs.srv import SetBool, Trigger
    from tf2_ros import TransformBroadcaster

    from ros2_adapter import SessionAdapter

    class WorkerBridge(Node):
        def __init__(self, session):
            super().__init__('ai_worker_bridge')
            self.session = session
            self.adapter = SessionAdapter(session)
            self.last_error_log = float('-inf')
            self.state_phase = 0.0
            self.last_tick = time.monotonic()
            self.accumulator = 0.0
            self.dropped_wall_seconds = 0.0
            # Volatile depth-one setpoints: no historical command replay.
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE)
            self.joints_pub = self.create_publisher(JointState, 'joint_states', qos)
            self.clock_pub = self.create_publisher(Clock, '/clock', qos)
            self.pose_pubs = {}
            self.tf = TransformBroadcaster(self)
            self.create_subscription(JointState, 'ai_worker/joint_targets', self.joint_command, qos)
            for side, label in [('l', 'left'), ('r', 'right')]:
                self.pose_pubs[side] = self.create_publisher(
                    PoseStamped, f'ai_worker/{label}/pose', qos)
                self.create_subscription(
                    PoseStamped, f'ai_worker/{label}/target_pose',
                    lambda msg, s=side: self.pose_command(s, msg), qos)
                for component in ('thumb', 'grasp'):
                    self.create_subscription(
                        Float64, f'ai_worker/{label}/{component}',
                        lambda msg, s=side, c=component: self.accept(
                            lambda: self.adapter.set_grasp(s, c, msg.data)), qos)
            self.create_service(Trigger, 'ai_worker/reset', self.reset_service)
            self.create_service(Trigger, 'ai_worker/hold', self.hold_service)
            self.create_service(SetBool, 'ai_worker/pause', self.pause_service)
            # A steady clock keeps pause/resume/services alive even when /clock
            # stops or use_sim_time is enabled on this simulator node.
            self.timer = self.create_timer(session.controller.dt, self.advance,
                                          clock=TimerClock(clock_type=ClockType.STEADY_TIME))
            self.get_logger().info('SH5 bridge ready; targets persist; pose frame=world; m/rad')

        def accept(self, operation):
            try:
                operation()
            except ValueError as exc:
                now = time.monotonic()
                if now - self.last_error_log >= 1:
                    self.get_logger().warning(f'Rejected command: {exc}')
                    self.last_error_log = now

        def joint_command(self, msg):
            self.accept(lambda: self.adapter.set_joints(
                msg.name, msg.position, msg.velocity, msg.effort))

        def pose_command(self, side, msg):
            p, q = msg.pose.position, msg.pose.orientation
            self.accept(lambda: self.adapter.set_pose(
                side, msg.header.frame_id, [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]))

        def reset_service(self, request, response):
            self.adapter.reset()
            response.success, response.message = True, 'Reset robot/cans; pause state and simulation time preserved'
            return response

        def hold_service(self, request, response):
            self.adapter.hold()
            response.success, response.message = True, 'Both arms now target their current poses; fingers unchanged'
            return response

        def pause_service(self, request, response):
            self.session.paused = request.data
            response.success, response.message = True, 'Paused' if request.data else 'Running'
            return response

        def advance(self):
            now = time.monotonic()
            elapsed, self.last_tick = now - self.last_tick, now
            if self.session.paused:
                self.accumulator = 0.0
            else:
                # GUI rendering can delay ROS timers. Integrate fixed steps to
                # catch up, capped at 100 ms so a window stall cannot cause an
                # unbounded simulation burst. This is soft real-time pacing.
                pending = self.accumulator + elapsed
                self.dropped_wall_seconds += max(0.0, pending - .1)
                self.accumulator = min(.1, pending)
                while self.accumulator + 1e-10 >= self.session.controller.dt:
                    self.session.tick()
                    self.accumulator -= self.session.controller.dt
            self.state_phase += elapsed
            if self.state_phase >= .02 - 1e-10:
                self.state_phase = 0.0
                self.publish_state()

        def publish_state(self):
            state = self.adapter.snapshot()
            stamp = Time(sec=int(state['stamp'][0]), nanosec=int(state['stamp'][1]))
            self.clock_pub.publish(Clock(clock=stamp))
            joints = JointState()
            joints.header.stamp = stamp
            joints.name, joints.position, joints.velocity = (
                state['names'], state['positions'], state['velocities'])
            self.joints_pub.publish(joints)
            transforms = []
            for side, label in [('l', 'left'), ('r', 'right')]:
                position, quaternion = state['poses'][side]
                pose = PoseStamped()
                pose.header.stamp, pose.header.frame_id = stamp, 'world'
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
                (pose.pose.orientation.x, pose.pose.orientation.y,
                 pose.pose.orientation.z, pose.pose.orientation.w) = quaternion
                self.pose_pubs[side].publish(pose)
                transform = TransformStamped()
                transform.header = pose.header
                transform.child_frame_id = f'ai_worker_{label}_hand'
                (transform.transform.translation.x, transform.transform.translation.y,
                 transform.transform.translation.z) = position
                transform.transform.rotation = pose.pose.orientation
                transforms.append(transform)
            self.tf.sendTransform(transforms)

    return WorkerBridge


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui', action='store_true', help='Show the integrated Qt workspace')
    parser.add_argument('--duration', type=float, default=0, help='Wall seconds; 0 runs until stopped')
    parser.add_argument('--ui-screenshot', type=Path, help='Save the GUI before closing; requires --gui')
    parser.add_argument('--report', type=Path, default=Path('outputs/ros2_run.json'))
    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
        from rclpy.utilities import remove_ros_args
    except ImportError:
        parser.exit(2, 'ROS 2 Python modules unavailable. See ros2/README.md; run_ik.cmd remains usable.\n')
    ros_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    args = parser.parse_args(remove_ros_args(args=ros_argv)[1:])
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error('--duration must be finite and nonnegative')
    if args.ui_screenshot and not args.gui:
        parser.error('--ui-screenshot requires --gui')
    from run_ik import Session
    rclpy.init(args=ros_argv[1:])
    node, executor, window = None, None, None
    session, run_start, completed = None, None, False
    renderer_name = None
    try:
        session = Session()
        node = build_node_class()(session)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        run_start = time.monotonic()
        if args.gui:
            if Path('/dev/dxg').exists():
                # Limit these defaults to the WSL GUI process; respect user
                # overrides and leave native Linux/Windows rendering untouched.
                os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
                os.environ.setdefault('GALLIUM_DRIVER', 'd3d12')
                # Prefer the discrete adapter when WSL exposes an NVIDIA GPU.
                # Users may select another GPU explicitly via the Mesa variable.
                if 'MESA_D3D12_DEFAULT_ADAPTER_NAME' not in os.environ:
                    try:
                        gpu = subprocess.run(
                            ['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=2, check=False)
                        if gpu.returncode == 0 and 'NVIDIA' in gpu.stdout:
                            os.environ['MESA_D3D12_DEFAULT_ADAPTER_NAME'] = 'NVIDIA'
                    except (OSError, subprocess.TimeoutExpired):
                        pass
            from PySide6.QtCore import QTimer
            from integrated_gui import RobotWorkspace, make_app
            app = make_app()
            # WSL's D3D12 translation is costly for extra rendering passes.
            # These settings affect only lighting/reflections, never contacts.
            if Path('/dev/dxg').exists():
                session.model.light_castshadow[:] = False
                session.model.mat_reflectance[:] = 0
            window = RobotWorkspace(session, duration=args.duration, screenshot=args.ui_screenshot)
            window.physics_timer.stop()  # The ROS timer exclusively advances physics.
            timer = QTimer(window)

            def pump():
                if not rclpy.ok():
                    window.close()
                    return
                try:
                    # ROS callbacks, Qt widgets and MuJoCo share one thread.
                    # Bound work per event so incoming traffic cannot starve Qt.
                    for _ in range(4):
                        executor.spin_once(timeout_sec=0)
                except ExternalShutdownException:
                    window.close()
                except Exception:
                    import traceback
                    window.fail(traceback.format_exc())

            timer.timeout.connect(pump)
            timer.start(2)
            window.show()
            if window.viewport.render_context is not None:
                from OpenGL import GL
                window.viewport.makeCurrent()
                renderer_name = GL.glGetString(GL.GL_RENDERER).decode('utf-8', errors='replace')
                window.viewport.doneCurrent()
                node.get_logger().info(f'OpenGL renderer: {renderer_name}')
            # Model/context loading is initialization, not a missed control tick.
            node.last_tick = time.monotonic()
            node.accumulator = 0.0
            run_start = node.last_tick
            app.exec()
            timer.stop()
            if window.failure:
                raise RuntimeError(window.failure)
        else:
            start = time.monotonic()
            while rclpy.ok() and (not args.duration or time.monotonic() - start < args.duration):
                executor.spin_once(timeout_sec=.05)
        completed = True
    except (KeyboardInterrupt, ExternalShutdownException):
        completed = True
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if completed and session is not None and run_start is not None:
        wall_seconds = time.monotonic() - run_start
        report = {
            'gui': bool(args.gui), 'wall_seconds': wall_seconds,
            'simulation_seconds': session.data.time - session.start_sim,
            'control_cycles': session.cycles,
            'average_control_hz_wall_including_pauses': session.cycles / max(wall_seconds, 1e-9),
            'rendered_frames': window.viewport.frames if window is not None else 0,
            'opengl_renderer': renderer_name,
            'dropped_wall_seconds': node.dropped_wall_seconds,
            'arm_modes': session.joint_controls.arm_modes,
            'warnings': session.data.warning.number.tolist(),
            'final_state': node.adapter.snapshot(),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'ROS 2 run finished: {session.cycles} control cycles; report: {args.report}')


if __name__ == '__main__':
    main()
