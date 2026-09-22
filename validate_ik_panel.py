"""Check RPY conversion, independent target editing, and optional integrated Qt widgets."""
import argparse
import json

import numpy as np

from ai_worker import ROOT
from ik_control import SIDES
from ik_panel import PoseSliders, matrix_to_rpy, rpy_to_matrix
from run_ik import Session


def validate_logic():
    rng = np.random.default_rng(22)
    cases = rng.uniform([-np.pi, -np.pi/2, -np.pi], [np.pi, np.pi/2, np.pi], (100, 3)).tolist()
    cases += [[.2, np.pi/2, .4], [-.3, -np.pi/2, 1.2], [np.pi, 0, -np.pi], [0, np.pi/2 - 1e-8, 0]]
    for angles in cases:
        matrix = rpy_to_matrix(angles)
        assert np.allclose(rpy_to_matrix(matrix_to_rpy(matrix)), matrix, atol=2e-7)
    session = Session()
    controls = PoseSliders(session)
    initial = {s: p.copy() for s, p in session.controller.targets.items()}
    for side in SIDES:
        assert np.array_equal(initial[side].rotation, session.controller.targets[side].rotation)
        for index in range(6):
            other = 'r' if side == 'l' else 'l'
            untouched = session.controller.targets[other].copy()
            before = session.controller.targets[side].copy()
            value = float(controls.values[side][index] + (0.005 if index < 3 else .02))
            session.demo = True
            controls.set_value(side, index, value)
            after = session.controller.targets[side]
            assert not session.demo
            assert np.array_equal(untouched.position, session.controller.targets[other].position)
            assert np.array_equal(untouched.rotation, session.controller.targets[other].rotation)
            if index < 3:
                assert np.array_equal(before.rotation, after.rotation)
                assert np.isclose(after.position[index], value)
            else:
                assert np.array_equal(before.position, after.position)
                assert np.allclose(after.rotation, rpy_to_matrix(controls.values[side][3:]))
            after.validate()
    for _ in range(300):
        session.tick()
    errors = session.controller.errors(session.data)
    for error in errors.values():
        assert error['position_error_mm'] < 2.0, errors
        assert error['orientation_error_deg'] < 0.5, errors
    for side in SIDES:
        controls.hold_hand(side)
        assert session.controller.errors(session.data)[side]['position_error_mm'] < 1e-9
        controls.reset_hand(side)
        assert np.array_equal(initial[side].position, session.controller.targets[side].position)
        assert np.array_equal(initial[side].rotation, session.controller.targets[side].rotation)
    # UI endpoint rounding snaps to exact pi; larger excesses are rejected.
    controls.set_value('l', 5, 3.142)
    assert np.isclose(controls.values['l'][5], np.pi)
    controls.set_value('l', 4, -1.571)
    assert np.isclose(controls.values['l'][4], -np.pi/2)
    for index, value in [(5, 3.2), (4, 1.6)]:
        try:
            controls.set_value('l', index, value)
        except ValueError:
            pass
        else:
            raise AssertionError('Out-of-range radian target accepted')
    try:
        controls.set_value('l', 0, float('nan'))
        raise AssertionError('Invalid input accepted')
    except ValueError:
        pass
    return {'rpy_roundtrips_including_gimbal_lock': len(cases), 'independent_controls': 12,
            'physical_tracking_after_3s': errors, 'hold_reset_and_input_validation': 'PASS'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integrated', action='store_true', help='Validate the single-window Qt workspace')
    args = parser.parse_args()
    report = {'status': 'PASS', 'logic': validate_logic()}
    if args.integrated:
        from validate_integrated_gui import validate
        report['integrated'] = validate()
    name = 'integrated_gui_validation.json' if args.integrated else 'ik_panel_validation.json'
    path = ROOT / 'outputs' / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
