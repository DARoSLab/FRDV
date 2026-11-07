"""Loader (URDF/MJCF), automatic keypoint mapping and config generation."""
import os
import textwrap

import mujoco
import numpy as np
import pytest
import yaml

from frdv.automap import analyze
from frdv.robot_model import load_model, surface_points

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

URDF = textwrap.dedent("""
<robot name="biped">
  <link name="base"><inertial><mass value="5"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="0.2 0.3 0.4"/></geometry></visual></link>
  {legs}
</robot>""")
LEG = textwrap.dedent("""
  <link name="{s}_thigh"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 -0.2"/><geometry><cylinder radius="0.04" length="0.4"/></geometry></visual></link>
  <link name="{s}_shin"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 -0.2"/><geometry><cylinder radius="0.04" length="0.4"/></geometry></visual></link>
  <link name="{s}_foot"><inertial><mass value="0.5"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0.05 0 -0.04"/><geometry><box size="0.22 0.1 0.04"/></geometry></visual></link>
  <link name="{s}_hip_yaw"><inertial><mass value="0.5"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="{s}_hip_yaw_j" type="revolute"><parent link="base"/><child link="{s}_hip_yaw"/>
    <origin xyz="0 {y} -0.2"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="{s}_hip_j" type="revolute"><parent link="{s}_hip_yaw"/><child link="{s}_thigh"/>
    <axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="{s}_knee_j" type="revolute"><parent link="{s}_thigh"/><child link="{s}_shin"/>
    <origin xyz="0 0 -0.4"/><axis xyz="0 1 0"/><limit lower="0" upper="2" effort="1" velocity="1"/></joint>
  <joint name="{s}_ankle_j" type="revolute"><parent link="{s}_shin"/><child link="{s}_foot"/>
    <origin xyz="0 0 -0.4"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
""")


@pytest.fixture
def biped_urdf(tmp_path):
    p = tmp_path / "biped.urdf"
    p.write_text(URDF.format(legs=LEG.format(s="left", y=0.1) + LEG.format(s="right", y=-0.1)))
    return str(p)


def test_urdf_gets_free_joint_and_keeps_visuals(biped_urdf):
    m = load_model(biped_urdf)
    assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert m.ngeom >= 7                     # visuals kept (URDF import would discard them)
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_hip_yaw") >= 0   # massless link not fused


def test_automap_urdf_biped(biped_urdf):
    lay = analyze(load_model(biped_urdf))
    kp = {k: b for k, (b, _) in lay.keypoints.items()}
    for s in ("left", "right"):
        assert kp[f"{s}_knee"] == f"{s}_shin"
        assert kp[f"{s}_ankle"] == f"{s}_foot"
        assert kp[f"{s}_hip"] in (f"{s}_thigh", f"{s}_hip_yaw")
    assert len(lay.foot_points) == 8
    assert 0.9 < lay.height < 1.3


def test_automap_g1_matches_tuned():
    from conftest import menagerie_or_skip
    lay = analyze(load_model(menagerie_or_skip("unitree_g1")))
    kp = {k: b for k, (b, _) in lay.keypoints.items()}
    assert kp["left_knee"] == "left_knee_link" and kp["right_knee"] == "right_knee_link"
    assert kp["left_elbow"] == "left_elbow_link" and kp["right_shoulder"] == "right_shoulder_pitch_link"
    assert kp["left_ankle"] == "left_ankle_pitch_link"
    assert {"left_thumb", "right_forearm_up"} <= set(lay.markers)
    assert abs(lay.height - 1.32) < 0.1


def test_surface_points_cover_feet(biped_urdf):
    m = load_model(biped_urdf)
    pts = surface_points(m)
    bodies = {b for b, _ in pts}
    assert {"left_foot", "right_shin", "base"} <= bodies


def test_new_robot_config_is_valid(biped_urdf):
    from frdv.cli.common import foot_points, validate_robot
    from frdv.cli.new_robot import build_config
    text, _ = build_config(biped_urdf, "biped")
    cfg = yaml.safe_load(text)
    assert validate_robot(cfg) == []
    assert all(isinstance(f, tuple) and len(f[1]) == 3 for f in foot_points(cfg))
    assert cfg["ground_points"] == "auto"
