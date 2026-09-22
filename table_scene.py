"""Low table and two upright, freely movable Pepsi cans for the SH5 scene."""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
TABLE_CENTER = np.array([0.70, 0.0, 0.0])
TABLE_TOP = 0.72
CAN_RADIUS = 0.033
CAN_HEIGHT = 0.124
CAN_MASS = 0.370
CAN_POSITIONS = ((0.54, 0.26), (0.54, -0.26))


def _logo_mesh():
    """Project a round logo onto the can wall, front and back, using mesh UVs.

    The downloaded PNG is unmodified. UV coordinates sample the official logo
    circle inside its padded square source image (roughly pixels 300..701).
    """
    vertices, texcoords, faces = [], [], []
    segments, rings, patch_radius = 64, 8, 0.027
    shell_radius = CAN_RADIUS + 0.00015
    for front in (-1, 1):
        start = len(vertices)
        vertices.append([front * shell_radius, 0, 0])
        texcoords.append([0.5, 0.5])
        for ring in range(1, rings + 1):
            for segment in range(segments):
                theta = 2 * np.pi * segment / segments
                u, v = ring / rings * np.cos(theta), ring / rings * np.sin(theta)
                y, z = front * patch_radius * u, patch_radius * v
                x = front * np.sqrt(shell_radius**2 - y**2)
                vertices.append([x, y, z])
                texcoords.append([0.5 + 0.201 * u, 0.5 - 0.201 * v])
        for segment in range(segments):
            nxt = (segment + 1) % segments
            faces.append([start, start + 1 + segment, start + 1 + nxt])
        for ring in range(rings - 1):
            a = start + 1 + ring * segments
            b = a + segments
            for segment in range(segments):
                nxt = (segment + 1) % segments
                faces.extend([[a + segment, b + segment, b + nxt], [a + segment, b + nxt, a + nxt]])
    return np.array(vertices).ravel(), np.array(texcoords).ravel(), np.array(faces).ravel()


def add_table_and_cans(spec):
    spec.add_texture(name='pepsi_official_logo', type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file=str(ROOT / 'assets' / 'pepsi' / 'pepsi_logo.png'))
    material = spec.add_material(name='pepsi_logo_material', rgba=[1, 1, 1, 1],
                                 specular=0.15, shininess=0.25)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'pepsi_official_logo'
    vertex, texcoord, face = _logo_mesh()
    spec.add_mesh(name='pepsi_logo_patches', uservert=vertex, usertexcoord=texcoord, userface=face)
    table = spec.worldbody.add_body(name='work_table', pos=TABLE_CENTER)
    table.add_geom(name='table_top', type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.30, 0.45, 0.02], pos=[0, 0, TABLE_TOP - 0.02],
                   rgba=[0.62, 0.48, 0.32, 1], friction=[0.8, 0.01, 0.001])
    for x in (-0.255, 0.255):
        for y in (-0.405, 0.405):
            table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, 0.34],
                           pos=[x, y, 0.34], rgba=[0.24, 0.28, 0.33, 1])
    for number, (x, y) in enumerate(CAN_POSITIONS, start=1):
        can = spec.worldbody.add_body(name=f'pepsi_can_{number}',
                                     pos=[x, y, TABLE_TOP + CAN_HEIGHT/2 + 0.001])
        can.add_freejoint(name=f'pepsi_can_{number}_free')
        # The collision cylinder alone supplies mass and inertia. Decorations
        # contribute no mass and no contact surfaces.
        can.add_geom(name=f'can_{number}_collision', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                     size=[CAN_RADIUS, CAN_HEIGHT/2, 0], mass=CAN_MASS,
                     rgba=[0, 0, 0, 0], group=3, condim=4,
                     friction=[0.8, 0.01, 0.001], solref=[0.006, 1])
        visual = dict(contype=0, conaffinity=0, mass=0, group=2)
        can.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CAN_RADIUS, 0.055, 0],
                     rgba=[0.015, 0.13, 0.77, 1], **visual)
        can.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname='pepsi_logo_patches',
                     material='pepsi_logo_material', **visual)
        for sign in (-1, 1):
            can.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0315, 0.002, 0],
                         pos=[0, 0, sign * 0.060], rgba=[0.78, 0.81, 0.84, 1], **visual)
            can.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0325, 0.0025, 0],
                         pos=[0, 0, sign * 0.0565], rgba=[0.65, 0.7, 0.76, 1], **visual)
        can.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.010, 0.006, 0.001],
                     pos=[0.003, 0, 0.0625], rgba=[0.35, 0.39, 0.44, 1], **visual)
        can.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.006, 0.003, 0.0012],
                     pos=[-0.005, 0, 0.063], rgba=[0.84, 0.86, 0.9, 1], **visual)
