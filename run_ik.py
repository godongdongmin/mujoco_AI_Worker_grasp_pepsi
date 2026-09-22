"""Interactive dual-arm SH5 inverse kinematics. Simulation only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import glfw
import mujoco
import numpy as np

from ai_worker import ROOT, camera_for
from ik_control import DualArmIK, SIDES, axis_rotation, hammer_hand_rotation, load_ik_model, site_pose


class Session:
    def __init__(self, control_hz=100.0, demo=False):
        self.model, self.data = load_ik_model()
        ratio = 1.0 / (control_hz * self.model.opt.timestep)
        if not math.isfinite(ratio) or ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError('Control Hz must divide the 500 Hz physics rate (e.g. 50, 100, 250)')
        self.substeps = int(round(ratio))
        self.controller = DualArmIK(self.model, self.data, control_hz=control_hz)
        self.initial_targets = {s: p.copy() for s, p in self.controller.targets.items()}
        for side, pose in self.initial_targets.items():
            pose.rotation[:] = hammer_hand_rotation(side)
        self.controller.targets = {s: p.copy() for s, p in self.initial_targets.items()}
        # Settle the cans and robot while maintaining the hammer-grip hand goals.
        for _ in range(round(control_hz)):
            self.controller.solve(self.data)
            self.controller.step_physics(self.data, self.substeps)
        self.controller.reset(self.data)
        self.controller.home = {s: p.copy() for s, p in self.initial_targets.items()}
        self.controller.targets = {s: p.copy() for s, p in self.initial_targets.items()}
        self.controller.update_markers(self.data)
        self.initial_qpos = self.data.qpos.copy()
        self.initial_controls = self.data.ctrl.copy()
        from joint_control import JointTargets
        self.joint_controls = JointTargets(self)
        self.start_sim = self.data.time
        self.demo_start = self.data.time
        self.demo = demo
        self.paused = False
        self.selected = 'l'
        self.edit_rotation = False
        self.solve_times = []
        self.peak_speed = 0.0
        self.error_samples = []
        self.cycles = 0

    def key(self, key):
        # Avoid letter/digit shortcuts that also toggle MuJoCo visual settings.
        if key == glfw.KEY_F8:
            self.selected = 'r' if self.selected == 'l' else 'l'
        elif key == glfw.KEY_F9:
            self.edit_rotation = not self.edit_rotation
        elif key == glfw.KEY_F12:
            self.paused = not self.paused
        elif key == glfw.KEY_F11:
            # Preserve simulation time so reports and bounded runs remain valid.
            current_time = self.data.time
            mujoco.mj_resetData(self.model, self.data)
            self.data.time = current_time
            self.data.qpos[:] = self.initial_qpos
            self.data.ctrl[:] = self.initial_controls
            self.controller.reset(self.data)
            self.controller.home = {s: p.copy() for s, p in self.initial_targets.items()}
            self.controller.targets = {s: p.copy() for s, p in self.initial_targets.items()}
            self.controller.update_markers(self.data)
            self.joint_controls.reset()
            self.demo = False
        elif key == glfw.KEY_F10:
            self.demo = not self.demo
            if self.demo:
                for side in SIDES:
                    self.joint_controls.enable_ik(side)
                self.controller.home = {s: site_pose(self.data, self.controller.sites[s]) for s in SIDES}
                self.demo_start = self.data.time
        elif key == glfw.KEY_HOME:
            self.demo = False
            for side in SIDES:
                self.joint_controls.enable_ik(side)
            self.controller.targets = {s: site_pose(self.data, self.controller.sites[s]) for s in SIDES}
        else:
            # World axes: +X forward, +Y robot-left, +Z up.
            axes = {glfw.KEY_UP: (0, 1), glfw.KEY_DOWN: (0, -1),
                    glfw.KEY_LEFT: (1, 1), glfw.KEY_RIGHT: (1, -1),
                    glfw.KEY_INSERT: (2, 1), glfw.KEY_DELETE: (2, -1)}
            if key in axes:
                self.demo = False
                self.joint_controls.enable_ik(self.selected)
                target = self.controller.targets[self.selected]
                axis, sign = axes[key]
                if self.edit_rotation:
                    target.rotation = axis_rotation(axis, sign * 0.05) @ target.rotation
                else:
                    target.position[axis] += sign * 0.01

    def tick(self):
        if self.demo:
            self.controller.set_demo(self.data.time - self.demo_start)
        started = time.perf_counter()
        self.joint_controls.prepare()
        active_sides = tuple(s for s in SIDES if self.joint_controls.arm_modes[s] == 'IK')
        result = self.controller.solve(self.data, active_sides)
        self.solve_times.append(time.perf_counter() - started)
        self.peak_speed = max([self.peak_speed] + [v['command_speed_rad_s'] for v in result.values()])
        self.controller.step_physics(self.data, self.substeps)
        self.joint_controls.synchronize()
        self.cycles += 1
        self.error_samples.append(self.controller.errors(self.data))

    def report(self, wall_seconds, active_wall_seconds, headless):
        times = np.array(self.solve_times) * 1000
        result = {'model': 'FFW-SH5', 'base': 'fixed', 'method': 'bounded damped least-squares IK',
                  'controlled_arm_joints': 14, 'hand_frame_offset_m': [0, 0, 0.06],
                  'arm_control_modes': self.joint_controls.arm_modes.copy(),
                  'joint_position_controls': len(self.joint_controls.targets),
                  'nominal_control_hz_sim_time': 1 / self.controller.dt,
                  'physics_hz_sim_time': 1 / self.model.opt.timestep,
                  'mode': 'headless accelerated' if headless else 'viewer paced',
                  'control_cycles': self.cycles, 'wall_seconds': wall_seconds,
                  'active_wall_seconds': active_wall_seconds,
                  'measured_control_hz_wall_time': self.cycles / max(active_wall_seconds, 1e-9),
                  'simulation_seconds': self.data.time - self.start_sim,
                  'peak_joint_command_speed_rad_s': self.peak_speed,
                  'ik_compute_ms': {'mean': float(times.mean()), 'p95': float(np.percentile(times, 95)), 'max': float(times.max())} if len(times) else {},
                  'final_actual_errors': self.controller.errors(self.data),
                  'warnings': self.data.warning.number.tolist(),
                  'limitations': ['No collision avoidance or reachability guarantee',
                                  'Bias feedforward is idealized, applied outside actuator force limits',
                                  'Command slew limit is not a measured physical joint-speed constraint',
                                  'No finger retargeting, base navigation, or hardware interface']}
        if self.error_samples:
            result['peak_actual_errors'] = {
                s: {key: max(e[s][key] for e in self.error_samples) for key in self.error_samples[0][s]}
                for s in SIDES}
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='Start the Cartesian trajectory demo')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--ui-screenshot', type=Path, help='Save the integrated window before closing')
    parser.add_argument('--duration', type=float, default=None,
                        help='Headless: simulation seconds; GUI: wall seconds; GUI default unlimited')
    parser.add_argument('--control-hz', type=float, default=100)
    parser.add_argument('--report', type=Path, default=ROOT / 'outputs' / 'ik_run.json')
    parser.add_argument('--render', type=Path)
    parser.add_argument('--record', type=Path, help='Headless GIF, at most 30 simulation seconds')
    args = parser.parse_args()
    duration = args.duration if args.duration is not None else (8 if args.headless else 0)
    if not math.isfinite(duration) or duration < 0 or (args.headless and duration <= 0):
        parser.error('Use a finite positive duration (zero allowed for unlimited GUI)')
    if not math.isfinite(args.control_hz) or args.control_hz <= 0:
        parser.error('Control Hz must be finite and positive')
    if args.record and (not args.headless or duration > 30):
        parser.error('--record requires --headless and duration <= 30')
    if args.ui_screenshot and args.headless:
        parser.error('--ui-screenshot requires the integrated GUI')
    try:
        session = Session(args.control_hz, args.demo)
    except ValueError as exc:
        parser.error(str(exc))
    model, data, controller = session.model, session.data, session.controller
    camera = camera_for(model)
    camera.lookat[:] = [0.32, 0, 0.78]
    camera.distance = 3.0
    renderer, frames = None, []
    workspace = None
    active_wall = 0.0
    wall_start = time.perf_counter()
    try:
        if args.render or args.record:
            renderer = mujoco.Renderer(model, height=600, width=800)
        if args.headless:
            next_frame = 0.0
            while data.time - session.start_sim < duration - controller.dt / 2:
                session.tick()
                if args.record and data.time - session.start_sim >= next_frame:
                    from PIL import Image
                    renderer.update_scene(data, camera=camera)
                    frames.append(Image.fromarray(renderer.render().copy()))
                    next_frame += 0.1
            active_wall = time.perf_counter() - wall_start
        else:
            try:
                from integrated_gui import run_workspace
            except ImportError as exc:
                raise RuntimeError('Integrated GUI dependency missing. Run setup.ps1.') from exc
            workspace = run_workspace(session, duration, args.ui_screenshot)
            active_wall = workspace['active_wall']
        wall_seconds = time.perf_counter() - wall_start
        if args.render:
            from PIL import Image
            renderer.update_scene(data, camera=camera)
            args.render.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(renderer.render()).save(args.render)
        if args.record and frames:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(args.record, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=False)
        report = session.report(wall_seconds, active_wall, args.headless)
        report['hand_slider_panel'] = workspace is not None
        report['gui'] = 'integrated Qt' if workspace is not None else 'none'
        if workspace is not None:
            report['workspace'] = workspace
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        if renderer is not None:
            renderer.close()


if __name__ == '__main__':
    main()
