"""Validate mirrored thumb opposition, physical C shape and 66 mm can clearance.

Can clearance is a static geometry check on a copied state, not a grasp/lift.
"""
import argparse
import json

import mujoco
import numpy as np

from ai_worker import ROOT, ensure_finite
from ik_control import SIDES
from joint_control import thumb_c_targets
from run_ik import Session


def validate(render=False):
    session = Session()
    model, data, controls = session.model, session.data, session.joint_controls
    commands = np.array([thumb_c_targets('l', v) for v in np.linspace(0, 1, 101)])
    assert np.all(np.diff(commands[:, 1]) >= -1e-12)
    assert np.all(np.diff(commands[:, 2:], axis=0) <= 1e-12)
    assert np.allclose(commands[:61, 2:], 0)
    for side in SIDES:
        acts = [model.actuator(f'finger_{side}_joint{i}').id for i in range(1, 5)]
        for value in np.linspace(0, 1, 101):
            q = thumb_c_targets(side, value)
            assert np.all(q >= controls.limits[acts, 0])
            assert np.all(q <= controls.limits[acts, 1])
        assert np.allclose(thumb_c_targets('r', 1), -thumb_c_targets('l', 1))
        controls.set_grasp(side, 'grasp', .5)
    # Exercise the actual servo in both directions, including the phase boundary.
    for value in (0, .3, .6, .8, 1, .6, 0, 1):
        for side in SIDES:
            controls.set_grasp(side, 'thumb', value)
        for _ in range(350):
            session.tick()
        for side in SIDES:
            q = np.array([data.qpos[model.joint(f'finger_{side}_joint{i}').qposadr[0]]
                          for i in range(1, 5)])
            if value == 0:
                # The official collision mesh stops MCP yaw short of zero.
                # Preserve the existing open command; verify reopening rather
                # than requiring the servo to penetrate its palm collision.
                assert abs(q[0]) < .05 and abs(q[1]) < .35
                assert np.max(np.abs(q[2:])) < .005
            else:
                assert np.max(np.abs(q - thumb_c_targets(side, value))) < .015, (side, value, q)
        ensure_finite(data)

    result = {}
    for side in SIDES:
        sign = 1 if side == 'l' else -1
        base = data.body(f'hx5_{side}_base')
        rotation = base.xmat.reshape(3, 3)
        tip = data.body(f'finger_{side}_link4')
        # Approximate distal centre-line endpoint, used only for orientation QA.
        endpoint = tip.xpos + tip.xmat.reshape(3, 3) @ np.array([0, -.035 * sign, 0])
        local_tip = rotation.T @ (endpoint - base.xpos)
        thumb_root_y = model.body(f'finger_{side}_link1').pos[1]
        assert abs(local_tip[1] - thumb_root_y) < .002
        assert .10 < local_tip[0] < .13 and .07 < local_tip[2] < .09

        # Place a copy of the scene's upright can inside the C-shaped opening.
        # No changes to the live state, scene layout, gravity or contact rules.
        work = mujoco.MjData(model)
        work.qpos[:] = data.qpos
        adr = model.joint('pepsi_can_1_free').qposadr[0]
        local_center = np.array([.0675, 0, .10])
        work.qpos[adr:adr + 3] = base.xpos + rotation @ local_center
        work.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, work)
        can = model.geom('can_1_collision').id
        hand_geoms = [g for g in range(model.ngeom) if model.geom_contype[g] and
                      (model.body(model.geom_bodyid[g]).name.startswith(f'finger_{side}_') or
                       model.body(model.geom_bodyid[g]).name == f'hx5_{side}_base')]
        clearance = min(mujoco.mj_geomDistance(model, work, can, g, .2, None) for g in hand_geoms)
        assert clearance > .0005, (side, clearance)
        result[side] = {'thumb_target_deg': np.rad2deg(thumb_c_targets(side, 1)).tolist(),
                        'thumb_endpoint_local_m': local_tip.tolist(),
                        'can_center_local_m': local_center.tolist(),
                        'static_can_clearance_mm': float(clearance * 1000)}
    assert not np.any(data.warning.number)
    if render:
        from PIL import Image, ImageDraw
        (ROOT / 'outputs').mkdir(exist_ok=True)
        canvas = Image.new('RGB', (1200, 640), 'white')
        draw = ImageDraw.Draw(canvas)
        with mujoco.Renderer(model, height=600, width=600) as renderer:
            for column, side in enumerate(SIDES):
                camera = mujoco.MjvCamera()
                camera.lookat[:] = data.body(f'hx5_{side}_base').xpos + \
                    data.body(f'hx5_{side}_base').xmat.reshape(3, 3) @ np.array([.06, 0, .10])
                camera.distance = .38
                camera.azimuth = 180
                camera.elevation = -80
                renderer.update_scene(data, camera=camera)
                canvas.paste(Image.fromarray(renderer.render()), (600 * column, 40))
                draw.text((600 * column + 18, 14),
                          f'{"Left" if side == "l" else "Right"}: Thumb 100% / Grasp 50% (top view)', fill='black')
        canvas.save(ROOT / 'outputs' / 'thumb_c_shape.png')
    return {'status': 'PASS', 'slider_sweep_and_reopening': 'PASS',
            'grasp_value': .5, 'hands': result, 'warnings': data.warning.number.tolist(),
            'can_check': 'Static clearance only; no grasp or lift success claimed'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    report = validate(parser.parse_args().render)
    output = ROOT / 'outputs' / 'thumb_c_validation.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
