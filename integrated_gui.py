"""Single-window Qt workspace with a native MuJoCo OpenGL viewport.

All simulation, command and rendering callbacks run on the Qt GUI thread.
The renderer blits its offscreen GPU buffer into QOpenGLWidget's framebuffer;
no separate MuJoCo window, screenshot streaming or image-copy loop is used.
"""
from __future__ import annotations

import math
from pathlib import Path
import time
import traceback

import glfw
import mujoco
import numpy as np
from OpenGL import GL
from PySide6.QtCore import Qt, QTimer, Signal, QSignalBlocker
from PySide6.QtGui import QFont, QKeySequence, QShortcut, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget, QDoubleSpinBox,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                               QPushButton, QScrollArea, QSlider,
                               QTabWidget, QToolBar, QVBoxLayout, QWidget)

from ai_worker import camera_for
from ik_control import SIDES, site_pose
from ik_panel import FIELDS, PoseSliders, matrix_to_rpy


STYLE = """
QMainWindow, QWidget { background: #f4f6f9; color: #223247; }
QToolBar { background: #ffffff; border: 0; border-bottom: 1px solid #dbe2eb; spacing: 7px; padding: 7px; }
QToolButton, QPushButton { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 5px; padding: 6px 9px; }
QToolButton:hover, QPushButton:hover { background: #e7eff9; border-color: #648fb8; }
QToolButton:checked { background: #d9e9f8; color: #124d7b; }
QDockWidget::title { background: #e7edf4; padding: 8px; font-weight: bold; }
QGroupBox { border: 1px solid #d8e0ea; border-radius: 6px; margin-top: 15px; padding: 9px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QDoubleSpinBox { background: white; border: 1px solid #ced8e3; border-radius: 4px; padding: 3px; }
QSlider::groove:horizontal { height: 5px; background: #d9e1eb; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #4783b8; border-radius: 2px; }
QSlider::handle:horizontal { background: #24639c; width: 12px; margin: -5px 0; border-radius: 6px; }
QTabWidget::pane { border: 1px solid #d5deea; }
QTabBar::tab { background: #e6ecf3; padding: 7px 14px; }
QTabBar::tab:selected { background: white; color: #185889; }
QStatusBar { background: #ffffff; border-top: 1px solid #dbe2eb; }
QScrollArea { border: none; }
QLabel#hint { color: #63748a; font-size: 11px; }
QLabel#actual { color: #46617d; font-family: Consolas; font-size: 11px; }
"""


def make_app():
    app = QApplication.instance()
    if app is None:
        fmt = QSurfaceFormat()
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setSamples(0)
        fmt.setSwapInterval(1)
        QSurfaceFormat.setDefaultFormat(fmt)
        app = QApplication([])
        app.setApplicationName('SH5 Robot Workspace')
        app.setFont(QFont('Segoe UI', 9))
        app.setStyle('Fusion')
    return app


class NoScrollSlider(QSlider):
    def wheelEvent(self, event):
        # Scrolling a long panel must not change a robot target accidentally.
        event.ignore()


class NoScrollSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class ValueControl(QWidget):
    changed = Signal(float)

    def __init__(self, label, low, high, value, decimals, unit=''):
        super().__init__()
        self.factor = 10 ** decimals
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 3)
        layout.setSpacing(3)
        line = QHBoxLayout()
        self.label = QLabel(label)
        line.addWidget(self.label)
        line.addStretch()
        self.spin = NoScrollSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(1 / self.factor)
        self.spin.setRange(low, high)
        self.spin.setSuffix(unit)
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(106)
        self.spin.setAccessibleName(label)
        line.addWidget(self.spin)
        layout.addLayout(line)
        self.slider = NoScrollSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(round(low * self.factor), round(high * self.factor))
        self.slider.setAccessibleName(label)
        layout.addWidget(self.slider)
        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)
        self.display(value)

    def _slider_changed(self, raw):
        value = raw / self.factor
        with QSignalBlocker(self.spin):
            self.spin.setValue(value)
        self.changed.emit(value)

    def _spin_changed(self, value):
        with QSignalBlocker(self.slider):
            self.slider.setValue(round(value * self.factor))
        self.changed.emit(value)

    def display(self, value):
        with QSignalBlocker(self.slider), QSignalBlocker(self.spin):
            if value < self.spin.minimum():
                self.spin.setMinimum(value - .05)
                self.slider.setMinimum(math.floor((value - .05) * self.factor))
            if value > self.spin.maximum():
                self.spin.setMaximum(value + .05)
                self.slider.setMaximum(math.ceil((value + .05) * self.factor))
            # Do not interrupt numeric text currently being edited.
            if not self.spin.hasFocus():
                self.spin.setValue(value)
            if not self.slider.isSliderDown():
                self.slider.setValue(round(value * self.factor))


class RobotViewport(QOpenGLWidget):
    failed = Signal(str)
    command = Signal(int)

    def __init__(self, session):
        super().__init__()
        self.session = session
        self.camera = camera_for(session.model)
        self.reset_camera()
        self.scene = None
        self.render_context = None
        self.option = mujoco.MjvOption()
        self.frames = 0
        self.failure = None
        self.last_mouse = None
        self.setMinimumSize(320, 260)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip('Drag left: orbit · drag right: pan · wheel: zoom\nDouble-click: reset camera\nArrow keys / Insert / Delete: move selected hand')

    def reset_camera(self, view='Perspective'):
        self.camera.lookat[:] = [.32, 0, .78]
        self.camera.distance = 3.0
        self.camera.azimuth, self.camera.elevation = {
            'Perspective': (135, -15), 'Front': (0, -10),
            'Side': (90, -10), 'Top': (90, -89)}.get(view, (135, -15))
        self.update()

    def initializeGL(self):
        try:
            self.session.model.vis.quality.offsamples = 0
            self.scene = mujoco.MjvScene(self.session.model, maxgeom=10000)
            self.render_context = mujoco.MjrContext(self.session.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self.context().aboutToBeDestroyed.connect(self.release_gl, Qt.ConnectionType.DirectConnection)
        except Exception:
            self._fail()

    def _fail(self):
        if self.failure is None:
            self.failure = traceback.format_exc()
            self.failed.emit(self.failure)

    def paintGL(self):
        if self.render_context is None or self.failure:
            return
        try:
            ratio = self.devicePixelRatioF()
            width, height = max(1, round(self.width() * ratio)), max(1, round(self.height() * ratio))
            context = self.render_context
            if context.offWidth != width or context.offHeight != height:
                mujoco.mjr_resizeOffscreen(width, height, context)
            mujoco.mjv_updateScene(self.session.model, self.session.data, self.option, None,
                                  self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
            mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), self.scene, context)
            # Qt renders widgets into a nonzero FBO. MuJoCo assumes framebuffer
            # zero for its WINDOW target, so explicitly blit to Qt's own FBO.
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, context.offFBO)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.defaultFramebufferObject())
            GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
            GL.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
                                 GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
            self.frames += 1
        except Exception:
            self._fail()

    def release_gl(self):
        if self.render_context is not None:
            self.makeCurrent()
            self.render_context.free()
            self.render_context = None
            self.scene = None
            self.doneCurrent()

    def mousePressEvent(self, event):
        self.setFocus()
        self.last_mouse = event.position()
        event.accept()

    def mouseMoveEvent(self, event):
        if self.last_mouse is None:
            return
        delta = event.position() - self.last_mouse
        self.last_mouse = event.position()
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if event.buttons() & Qt.MouseButton.RightButton:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif event.buttons() & Qt.MouseButton.LeftButton:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        mujoco.mjv_moveCamera(self.session.model, action, delta.x() / max(self.height(), 1),
                              delta.y() / max(self.height(), 1), self.camera)
        self.update()

    def mouseReleaseEvent(self, event):
        self.last_mouse = None

    def mouseDoubleClickEvent(self, event):
        self.reset_camera()

    def wheelEvent(self, event):
        mujoco.mjv_moveCamera(self.session.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                              0, .10 * event.angleDelta().y() / 120, self.camera)
        self.update()
        event.accept()

    def keyPressEvent(self, event):
        mapping = {Qt.Key.Key_Up: glfw.KEY_UP, Qt.Key.Key_Down: glfw.KEY_DOWN,
                   Qt.Key.Key_Left: glfw.KEY_LEFT, Qt.Key.Key_Right: glfw.KEY_RIGHT,
                   Qt.Key.Key_Insert: glfw.KEY_INSERT, Qt.Key.Key_Delete: glfw.KEY_DELETE,
                   Qt.Key.Key_Home: glfw.KEY_HOME}
        if event.key() in mapping:
            if not event.isAutoRepeat():
                self.command.emit(mapping[event.key()])
            event.accept()
        else:
            super().keyPressEvent(event)


def scroll_page():
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(10, 8, 10, 10)
    layout.setSpacing(9)
    scroll.setWidget(content)
    return scroll, layout


def hint(text, name='hint'):
    label = QLabel(text)
    label.setObjectName(name)
    label.setWordWrap(True)
    return label


class RobotWorkspace(QMainWindow):
    def __init__(self, session, *, duration=0, screenshot=None, start_timers=True):
        super().__init__()
        self.session = session
        self.control = PoseSliders(session)
        self.closed = False
        self.failure = None
        self.screenshot = screenshot
        self.active_wall = 0.0
        self.wall_start = time.perf_counter()
        self.last_tick = self.wall_start
        self.accumulator = 0.0
        self.sample_start = self.wall_start
        self.sample_cycles = session.cycles
        self.sample_frames = 0
        self.actual_hz = self.fps = 0.0
        self.controls, self.grasp_controls, self.joint_controls = {}, {}, {}
        self.error_labels, self.actual_labels, self.mode_labels = {}, {}, {}
        self.grasp_labels, self.docks = {}, {}
        self.setWindowTitle('SH5 Robot Workspace — MuJoCo')
        self.setStyleSheet(STYLE)
        self.setDockNestingEnabled(False)
        self.viewport = RobotViewport(session)
        self.viewport.failed.connect(self.fail)
        self.viewport.command.connect(self.command)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.addWidget(self.viewport, 1)
        footer = hint('  Mouse: left orbit · right pan · wheel zoom  |  World: +X forward · +Y left · +Z up')
        footer.setMinimumHeight(27)
        center_layout.addWidget(footer)
        self.setCentralWidget(center)
        for side in SIDES:
            self._hand_panel(side)
        self._joint_panel()
        self._toolbar()
        self._shortcuts()
        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label, 1)
        self.refresh()
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1540, screen.width() - 50), min(960, screen.height() - 70))
        self.setMinimumSize(1040, 650)
        self.move(screen.x() + 20, screen.y() + 20)
        self.resizeDocks([self.docks['l'], self.docks['r']], [320, 320], Qt.Orientation.Horizontal)
        self.docks['joints'].hide()
        self.physics_timer = QTimer(self)
        self.physics_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.physics_timer.timeout.connect(self.advance)
        self.draw_timer = QTimer(self)
        self.draw_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.draw_timer.timeout.connect(self.viewport.update)
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.refresh)
        if start_timers:
            self.physics_timer.start(2)
            self.draw_timer.start(20)
            self.ui_timer.start(80)
        if duration:
            QTimer.singleShot(round(duration * 1000), self.close)

    def _dock(self, key, title, widget, area):
        dock = QDockWidget(title, self)
        dock.setObjectName(key)
        # Embedded panels can be resized, moved and hidden, never detached.
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable |
                         QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock.setWidget(widget)
        if key != 'joints':
            dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
            dock.setMinimumWidth(285)
        else:
            dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(area, dock)
        self.docks[key] = dock
        return dock

    def _hand_panel(self, side):
        scroll, layout = scroll_page()
        title = 'LEFT HAND' if side == 'l' else 'RIGHT HAND'
        self.mode_labels[side] = hint('')
        layout.addWidget(self.mode_labels[side])
        pose_group = QGroupBox('Position / orientation')
        poses = QVBoxLayout(pose_group)
        poses.setSpacing(2)
        poses.addWidget(hint('World targets · position in m · absolute RPY in rad'))
        for index, field in enumerate(FIELDS):
            value = float(self.control.values[side][index])
            lo, hi = (value - .25, value + .25) if index < 3 else ((-np.pi/2, np.pi/2) if index == 4 else (-np.pi, np.pi))
            control = ValueControl(field, lo, hi, value, 3, ' m' if index < 3 else ' rad')
            control.changed.connect(lambda value, s=side, i=index: self.pose_changed(s, i, value))
            self.controls[side, index] = control
            poses.addWidget(control)
        buttons = QHBoxLayout()
        for label, hold in [('Use current pose', True), ('Reset target', False)]:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, s=side, h=hold: self.hand_action(s, h))
            buttons.addWidget(button)
        poses.addLayout(buttons)
        self.error_labels[side] = hint('')
        self.actual_labels[side] = hint('', 'actual')
        poses.addWidget(self.error_labels[side])
        poses.addWidget(self.actual_labels[side])
        layout.addWidget(pose_group)
        grasp_group = QGroupBox('Hand grasp')
        grasp = QVBoxLayout(grasp_group)
        grasp.addWidget(hint('Thumb: straight → C-shape\nGrasp: index / middle / ring · little finger independent'))
        for component in ('thumb', 'grasp'):
            value = self.session.joint_controls.grasp_values[side][component] * 100
            control = ValueControl(component.title(), 0, 100, value, 1, '%')
            control.changed.connect(lambda value, s=side, c=component: self.grasp_changed(s, c, value))
            self.grasp_controls[side, component] = control
            grasp.addWidget(control)
        self.grasp_labels[side] = hint('')
        grasp.addWidget(self.grasp_labels[side])
        button = QPushButton('Open thumb + grasp')
        button.clicked.connect(lambda checked=False, s=side: self.open_hand(s))
        grasp.addWidget(button)
        layout.addWidget(grasp_group)
        layout.addStretch()
        self._dock(side, title, scroll, Qt.DockWidgetArea.LeftDockWidgetArea if side == 'l' else Qt.DockWidgetArea.RightDockWidgetArea)

    def _joint_panel(self):
        outer = QWidget()
        layout = QVBoxLayout(outer)
        tools = QHBoxLayout()
        tools.addWidget(hint('All 63 joints · arm edits switch that arm to Joint mode'))
        tools.addStretch()
        for side, label in [('l', 'Left → IK'), ('r', 'Right → IK')]:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, s=side: self.resume_ik(s))
            tools.addWidget(button)
        layout.addLayout(tools)
        self.joint_tabs = QTabWidget()
        layout.addWidget(self.joint_tabs)
        groups = {name: [] for name in ('Base / lift / head', 'Left arm', 'Right arm', 'Left fingers', 'Right fingers')}
        controls = self.session.joint_controls
        for i in range(self.session.model.nu):
            name = self.session.model.actuator(i).name
            group = next((g for prefix, g in [('arm_l_', 'Left arm'), ('arm_r_', 'Right arm'),
                                             ('finger_l_', 'Left fingers'), ('finger_r_', 'Right fingers')]
                          if name.startswith(prefix)), 'Base / lift / head')
            groups[group].append(i)
        for name, indices in groups.items():
            scroll, rows = scroll_page()
            grid = QGridLayout()
            grid.setHorizontalSpacing(22)
            rows.addLayout(grid)
            for column in range(3):
                grid.setColumnStretch(column, 1)
            for order, i in enumerate(indices):
                control = ValueControl(self.session.model.actuator(i).name,
                                       controls.limits[i, 0], controls.limits[i, 1],
                                       controls.targets[i], 3,
                                       ' m' if controls.sliding[i] else ' rad')
                control.changed.connect(lambda value, index=i: self.joint_changed(index, value))
                grid.addWidget(control, order // 3, order % 3)
                self.joint_controls[i] = control
            rows.addStretch()
            self.joint_tabs.addTab(scroll, name)
        self.joint_tabs.setMinimumHeight(155)
        dock = self._dock('joints', 'JOINT POSITION', outer, Qt.DockWidgetArea.BottomDockWidgetArea)
        dock.visibilityChanged.connect(lambda shown: QTimer.singleShot(0, self._size_joint_panel) if shown else None)

    def _size_joint_panel(self):
        if not self.closed and self.docks['joints'].isVisible():
            self.resizeDocks([self.docks['joints']], [280], Qt.Orientation.Vertical)

    def _toolbar(self):
        toolbar = QToolBar('Simulation', self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        branding = QLabel(' SH5  /  MuJoCo   ')
        branding.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        toolbar.addWidget(branding)
        self.pause_action = toolbar.addAction('Pause', lambda: self.command(glfw.KEY_F12))
        self.demo_action = toolbar.addAction('Start demo', lambda: self.command(glfw.KEY_F10))
        toolbar.addAction('Reset robot', lambda: self.command(glfw.KEY_F11))
        toolbar.addAction('Hold both', lambda: self.command(glfw.KEY_HOME))
        toolbar.addSeparator()
        self.camera_choice = QComboBox()
        self.camera_choice.addItems(['Perspective', 'Front', 'Side', 'Top'])
        self.camera_choice.currentTextChanged.connect(self.viewport.reset_camera)
        toolbar.addWidget(self.camera_choice)
        toolbar.addAction('Reset camera', lambda: self.viewport.reset_camera(self.camera_choice.currentText()))
        toolbar.addSeparator()
        menu = self.menuBar().addMenu('View')
        for key, label in [('l', 'Left'), ('r', 'Right'), ('joints', 'Joints')]:
            action = self.docks[key].toggleViewAction()
            action.setText(label)
            toolbar.addAction(action)
            menu.addAction(action)
        menu.addAction('Restore layout', self.restore_layout)
        self.menuBar().addMenu('File').addAction('Exit', self.close)

    def _shortcuts(self):
        self.shortcuts = []
        for name, key in [('F8', glfw.KEY_F8), ('F9', glfw.KEY_F9), ('F10', glfw.KEY_F10),
                          ('F11', glfw.KEY_F11), ('F12', glfw.KEY_F12)]:
            shortcut = QShortcut(QKeySequence(name), self)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(lambda k=key: self.command(k))
            self.shortcuts.append(shortcut)
        shortcut = QShortcut(QKeySequence('End'), self.viewport)
        shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        shortcut.activated.connect(self.restore_layout)
        self.shortcuts.append(shortcut)

    def restore_layout(self):
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.docks['l'])
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.docks['r'])
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.docks['joints'])
        self.docks['l'].show()
        self.docks['r'].show()
        self.docks['joints'].hide()
        self.resizeDocks([self.docks['l'], self.docks['r']], [320, 320], Qt.Orientation.Horizontal)

    def pose_changed(self, side, index, value):
        self.control.set_value(side, index, value)
        self.refresh()

    def grasp_changed(self, side, component, value):
        self.session.joint_controls.set_grasp(side, component, value / 100)
        self.refresh()

    def joint_changed(self, index, value):
        self.session.joint_controls.set_joint(index, value)
        self.refresh()

    def open_hand(self, side):
        for component in ('thumb', 'grasp'):
            self.session.joint_controls.set_grasp(side, component, 0)
        self.refresh()

    def hand_action(self, side, hold):
        (self.control.hold_hand if hold else self.control.reset_hand)(side)
        self.refresh()

    def resume_ik(self, side):
        self.session.joint_controls.enable_ik(side)
        self.refresh()

    def command(self, key):
        self.session.key(key)
        self.accumulator = 0
        self.last_tick = time.perf_counter()
        self.refresh()

    def advance(self):
        if self.closed or self.failure:
            return
        now = time.perf_counter()
        elapsed, self.last_tick = now - self.last_tick, now
        if self.session.paused:
            self.accumulator = 0
            return
        self.active_wall += elapsed
        # At most 50 ms of catch-up per event: window drags/minimization or a
        # slow computer must not build an unbounded command backlog.
        self.accumulator = min(.05, self.accumulator + elapsed)
        try:
            while self.accumulator + 1e-10 >= self.session.controller.dt:
                self.session.tick()
                self.accumulator -= self.session.controller.dt
        except Exception:
            self.fail(traceback.format_exc())

    def refresh(self):
        if self.closed:
            return
        self.control.sync()
        controls = self.session.joint_controls
        for side in SIDES:
            self.mode_labels[side].setText(f'{controls.arm_modes[side]} mode  ·  {"selected" if side == self.session.selected else "independent control"}')
            for index, value in enumerate(self.control.values[side]):
                self.controls[side, index].display(float(value))
            for component in ('thumb', 'grasp'):
                self.grasp_controls[side, component].display(controls.grasp_values[side][component] * 100)
            custom = [c.title() for c in ('thumb', 'grasp') if controls.grasp_custom[side][c]]
            self.grasp_labels[side].setText('Custom joints: ' + ', '.join(custom) if custom else 'Little finger: controlled separately in Joints')
            actual = site_pose(self.session.data, self.session.controller.sites[side])
            self.actual_labels[side].setText('Actual XYZ (m): ' + '  '.join(f'{v:.3f}' for v in actual.position) + '\nActual RPY (rad): ' + '  '.join(f'{v:.3f}' for v in matrix_to_rpy(actual.rotation)))
            error = self.session.controller.errors(self.session.data)[side]
            self.error_labels[side].setText('Joint mode · sliders show actual pose' if controls.arm_modes[side] == 'JOINT' else
                                             f"Tracking error: {error['position_error_mm']:.2f} mm / {np.deg2rad(error['orientation_error_deg']):.4f} rad")
        if self.docks['joints'].isVisible():
            for i, control in self.joint_controls.items():
                control.display(float(controls.targets[i]))
        self.pause_action.setText('Resume' if self.session.paused else 'Pause')
        self.demo_action.setText('Stop demo' if self.session.demo else 'Start demo')
        now = time.perf_counter()
        interval = now - self.sample_start
        if interval >= .5:
            self.actual_hz = (self.session.cycles - self.sample_cycles) / interval
            self.fps = (self.viewport.frames - self.sample_frames) / interval
            self.sample_start, self.sample_cycles, self.sample_frames = now, self.session.cycles, self.viewport.frames
        state = 'PAUSED' if self.session.paused else ('DEMO' if self.session.demo else 'MANUAL')
        self.status_label.setText(f' {state}   |   Control {self.actual_hz:.1f} / {1/self.session.controller.dt:.0f} Hz   |   Physics 500 Hz (sim)   |   View {self.fps:.0f} fps   |   F8: {self.session.selected.upper()}   F9: {"rotation" if self.session.edit_rotation else "position"}')

    def fail(self, detail):
        if self.failure:
            return
        self.failure = detail
        print(detail, flush=True)
        # Stop visibly and close; a failed renderer must not leave an invisible
        # simulation running. The command-line runner returns a nonzero exit.
        self.session.paused = True
        QTimer.singleShot(0, self.close)

    def save_screenshot(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.grab().save(str(path)):
            raise RuntimeError(f'Could not save {path}')

    def closeEvent(self, event):
        if not self.closed:
            self.physics_timer.stop()
            self.draw_timer.stop()
            self.ui_timer.stop()
            if self.screenshot and not self.failure:
                self.save_screenshot(self.screenshot)
            self.viewport.release_gl()
            self.closed = True
        event.accept()


def run_workspace(session, duration=0, screenshot=None):
    app = make_app()
    window = RobotWorkspace(session, duration=duration, screenshot=screenshot)
    window.show()
    app.exec()
    if window.failure:
        raise RuntimeError('Integrated viewer failed:\n' + window.failure)
    return {'active_wall': window.active_wall, 'frames': window.viewport.frames,
            'window_count': 1, 'control_widgets': len(window.controls) + len(window.grasp_controls) + len(window.joint_controls)}
