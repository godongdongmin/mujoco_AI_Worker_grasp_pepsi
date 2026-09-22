"""Exercise the actual Qt widgets and MuJoCo framebuffer in one visible window."""
import json

import glfw
import numpy as np

from ai_worker import ROOT
from run_ik import Session


def validate():
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from integrated_gui import RobotWorkspace, make_app

    app = make_app()
    session = Session()
    window = RobotWorkspace(session, start_timers=False)
    output = ROOT / 'outputs'
    output.mkdir(exist_ok=True)
    window.show()
    QTest.qWait(350)
    try:
        assert window.failure is None, window.failure
        assert window.viewport.frames > 0
        assert len(window.controls) == 12 and len(window.grasp_controls) == 4
        assert len(window.joint_controls) == 63
        visible_windows = [w for w in QApplication.topLevelWidgets() if w.isVisible()]
        assert visible_windows == [window], visible_windows
        assert not window.docks['joints'].isVisible()
        for side in ('l', 'r'):
            assert window.grasp_controls[side, 'thumb'].spin.value() == 60
            for index in range(6):
                widget = window.controls[side, index]
                assert widget.spin.suffix() == (' m' if index < 3 else ' rad')
                value = round(window.control.values[side][index] + (.005 if index < 3 else .02), 3)
                widget.slider.setValue(round(value * widget.factor))
                assert np.isclose(window.control.values[side][index], value)
        for _ in range(300):
            session.tick()
        window.refresh()
        errors = session.controller.errors(session.data)
        assert max(e['position_error_mm'] for e in errors.values()) < 2, errors
        assert max(e['orientation_error_deg'] for e in errors.values()) < .5, errors
        # Numeric entry and Use current pose must route through the same model.
        spin = window.controls['l', 0].spin
        spin.setFocus()
        spin.selectAll()
        QTest.keyClicks(spin, '0.510')
        QTest.keyClick(spin, Qt.Key.Key_Return)
        assert np.isclose(session.controller.targets['l'].position[0], .510)
        window.viewport.setFocus()
        window.hand_action('l', True)
        assert np.allclose(session.controller.targets['l'].position, session.data.site('ik_palm_l').xpos)

        for side in ('l', 'r'):
            for component in ('thumb', 'grasp'):
                window.grasp_controls[side, component].slider.setValue(350)
                assert session.joint_controls.grasp_values[side][component] == .35
        window.docks['joints'].show()
        app.processEvents()
        for i, control in window.joint_controls.items():
            joint = session.joint_controls
            assert control.spin.suffix() == (' m' if joint.sliding[i] else ' rad')
            value = np.clip(joint.targets[i] + (.005 if joint.sliding[i] else .02),
                            control.spin.minimum(), control.spin.maximum())
            raw = round(value * control.factor)
            control.slider.setValue(raw)
            expected = np.clip(control.slider.value() / control.factor, *joint.limits[i])
            assert np.isclose(joint.targets[i], expected), (i, expected, joint.targets[i])
        assert session.joint_controls.arm_modes == {'l': 'JOINT', 'r': 'JOINT'}
        window.controls['l', 0].slider.setValue(window.controls['l', 0].slider.value() + 5)
        assert session.joint_controls.arm_modes == {'l': 'IK', 'r': 'JOINT'}
        window.resume_ik('r')
        assert session.joint_controls.arm_modes == {'l': 'IK', 'r': 'IK'}
        window.command(glfw.KEY_F11)
        for side in ('l', 'r'):
            assert session.joint_controls.grasp_values[side] == {'thumb': .6, 'grasp': 0}
        window.joint_tabs.setCurrentIndex(4)
        page = window.joint_tabs.currentWidget()
        QTest.qWait(80)  # settle the newly shown tab and dock resize before scrolling
        page.verticalScrollBar().setValue(page.verticalScrollBar().maximum())
        app.processEvents()
        last = window.joint_controls[session.model.actuator('finger_r_joint20').id]
        last_pos = last.mapTo(page.viewport(), QPoint(0, 0))
        assert last_pos.y() + last.height() <= page.viewport().height() + 1
        window.viewport.update()
        QTest.qWait(80)
        window.save_screenshot(output / 'integrated_joints.png')

        # Camera interaction, resize / framebuffer recreation and dock hiding.
        window.restore_layout()
        before = (window.viewport.camera.azimuth, window.viewport.camera.elevation)
        center = window.viewport.rect().center()
        QTest.mousePress(window.viewport, Qt.MouseButton.LeftButton, pos=center)
        QTest.mouseMove(window.viewport, center + QPoint(55, 25), delay=30)
        QTest.mouseRelease(window.viewport, Qt.MouseButton.LeftButton, pos=center + QPoint(55, 25))
        assert before != (window.viewport.camera.azimuth, window.viewport.camera.elevation)
        distance = window.viewport.camera.distance
        event = QWheelEvent(QPointF(center), QPointF(window.viewport.mapToGlobal(center)),
                            QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(window.viewport, event)
        assert window.viewport.camera.distance < distance, (distance, window.viewport.camera.distance, event.angleDelta().y())
        window.docks['l'].hide()
        window.docks['r'].hide()
        app.processEvents()
        assert not window.docks['l'].isVisible() and not window.docks['r'].isVisible()
        window.restore_layout()
        window.viewport.reset_camera()
        window.resize(1200, 720)
        QTest.qWait(100)
        assert window.viewport.width() >= 320 and window.viewport.height() >= 260
        for side in ('l', 'r'):
            page = window.docks[side].widget()
            page.verticalScrollBar().setValue(page.verticalScrollBar().maximum())
            app.processEvents()
            control = window.grasp_controls[side, 'grasp']
            pos = control.mapTo(page.viewport(), QPoint(0, 0))
            assert pos.y() + control.height() <= page.viewport().height() + 1
        window.viewport.update()
        QTest.qWait(100)
        window.save_screenshot(output / 'integrated_compact.png')

        # Run the real timers and window shortcuts, not just a headless loop.
        window.resize(1540, 960)
        for side in ('l', 'r'):
            window.docks[side].widget().verticalScrollBar().setValue(0)
        window.command(glfw.KEY_F11)
        window.physics_timer.start(2)
        window.draw_timer.start(20)
        window.ui_timer.start(80)
        window.pause_action.trigger()
        paused_time = session.data.time
        QTest.qWait(120)
        assert session.data.time == paused_time
        window.pause_action.trigger()
        QTest.qWait(120)
        assert session.data.time > paused_time
        window.demo_action.trigger()
        assert session.demo
        QTest.qWait(1400)
        window.refresh()
        assert session.demo  # display synchronization must not send commands
        assert window.actual_hz > 0 and window.viewport.frames > 10
        window.demo_action.trigger()
        window.command(glfw.KEY_F11)
        QTest.qWait(300)
        window.viewport.setFocus()
        selected = session.selected
        QTest.keyClick(window.viewport, Qt.Key.Key_F8)
        assert selected != session.selected
        QTest.keyClick(window.viewport, Qt.Key.Key_F8)
        window.save_screenshot(output / 'integrated_initial.png')
        assert window.failure is None, window.failure
        assert not np.any(session.data.warning.number)
        return {'status': 'PASS', 'visible_windows': 1, 'pose_sliders': 12,
                'grasp_sliders': 4, 'joint_sliders': 63,
                'physical_tracking_after_widget_edits': errors,
                'numeric_input_modes_reset_pause_demo': 'PASS',
                'camera_orbit_zoom_and_resize': 'PASS',
                'small_window_scrolling_and_last_joint_reachable': 'PASS',
                'rendered_frames': window.viewport.frames,
                'measured_control_hz_last_sample': window.actual_hz,
                'warnings': session.data.warning.number.tolist()}
    finally:
        window.close()
        app.processEvents()
        assert window.closed and window.viewport.render_context is None
        assert not window.physics_timer.isActive()
        assert not window.draw_timer.isActive()


if __name__ == '__main__':
    report = validate()
    (ROOT / 'outputs' / 'integrated_gui_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
