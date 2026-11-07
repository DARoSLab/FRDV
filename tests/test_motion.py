"""Motion loaders, target building, smoothing and config/CLI plumbing."""
import os
from dataclasses import replace

import numpy as np
import pytest

from frdv import Retargeter, motion as M
from frdv.parallel import chunk_bounds, handoff
from frdv.smooth import despike, find_spikes, joint_steps

BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0 10 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0 10 0
    }
  }
  JOINT LeftLeg
  {
    OFFSET 10 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftFoot
    {
      OFFSET 0 -80 0
      CHANNELS 3 Zrotation Xrotation Yrotation
    }
  }
}
MOTION
Frames: 3
Frame Time: 0.0333333
5 90 7 0 0 0 0 0 0 0 0 0 0 0 0
5 90 7 0 0 90 0 0 0 0 0 0 0 0 0
5 95 7 0 0 0 90 0 0 0 0 0 0 0 0
"""


@pytest.fixture()
def bvh_file(tmp_path):
    p = tmp_path / "clip.bvh"
    p.write_text(BVH)
    return str(p)


def test_bvh_fk_and_markers(bvh_file):
    m = M.load(bvh_file, "bvh", markers=[{"name": "Tip", "parent": "Spine", "offset": [0, 0, 5]}])
    assert m.fps == pytest.approx(30.0, rel=1e-4)
    i = m.index
    assert np.allclose(m.positions[0, i("LeftFoot")], [15, 10, 7])
    # frame 1: root yaw (Y) 90 deg rotates the leg offset (10,0,0) -> (0,0,-10)
    assert np.allclose(m.positions[1, i("LeftLeg")], [5, 90, -3], atol=1e-9)
    # marker rides on Spine: frame 2 has Spine Z-rotation 90 deg, so local +z offset stays +z
    assert np.allclose(m.positions[2, i("Tip")], m.positions[2, i("Spine")] + [0, 0, 5], atol=1e-9)
    assert m.rest is not None and np.allclose(m.rest[i("LeftFoot")], [10, -80, 0])


def test_build_targets_units_rotation_ground(bvh_file):
    m = M.load(bvh_file, "bvh")
    ds = {"units": 0.01, "rotation": [[0, 0, 1], [1, 0, 0], [0, 1, 0]], "root": "Hips",
          "keypoints": {"pelvis": "Hips", "left_ankle": "LeftFoot"},
          "scale": {"mode": "fixed", "value": 1.0}, "ground": {"mode": "frame0"}}
    tg = M.build_targets(m, ds, robot_height=1.0, robot_keypoints=["pelvis", "left_ankle", "left_toe"])
    assert tg.keypoints == ["pelvis", "left_ankle"]           # left_toe has no dataset joint
    assert np.allclose(tg.X[0, 0, :2], 0.0)                     # root XY zeroed at frame 0
    assert tg.X[0, :, 2].min() >= -1e-12                        # grounded
    assert tg.X[0, 0, 2] == pytest.approx(0.80)                 # hips 80 cm above the foot, z-up
    s = M.build_targets(m, {**ds, "scale": {"mode": "height"}}, 1.0, ["pelvis"]).scale
    assert np.allclose(s, 1.0 / M.build_targets(m, ds, 1.0, ["pelvis"]).human_height)


def test_configs_compose_and_validate():
    pytest.importorskip("hydra")
    from conftest import menagerie_or_skip
    menagerie_or_skip("unitree_g1")
    from frdv.cli.common import ik_config, to_dict, validate_robot
    from frdv.cli.config import available, compose
    assert available("dataset") == ["amass", "lafan1"]
    for ds in available("dataset"):
        c = to_dict(compose([f"dataset={ds}", "data=x"]))
        assert c["dataset"]["name"] == ds
    c = to_dict(compose(["robot=g1", "dataset=lafan1", "data=x", "solver.accel_weight=7"]))
    assert validate_robot(c["robot"]) == []
    ik = ik_config(c["robot"], c["solver"], ["pelvis", "left_knee"], c["dataset"])
    assert ik.accel_weight == 7.0 and ik.targets == ["pelvis", "left_knee_link"]


def test_new_robot_guesses_g1():
    import mujoco
    from conftest import menagerie_or_skip
    from frdv.cli.new_robot import guess_keypoints
    m = mujoco.MjModel.from_xml_path(menagerie_or_skip("unitree_g1"))
    kp = guess_keypoints(m)
    assert kp["left_knee"] == "left_knee_link" and kp["right_hip"] == "right_hip_pitch_link"
    assert kp["pelvis"] == "pelvis"


def _walk_targets(cfg, T=200):
    from frdv import IKProblem
    p = IKProblem(cfg)
    out = np.empty((T, p.S, 3))
    for t in range(T):
        dv = np.zeros(p.nv)
        dv[0] = 0.01 * t
        dv[6:] = 0.3 * np.sin(2 * np.pi * t / 60 + np.arange(p.nv - 6))
        q = p.integrate(p.q_nominal, dv)
        q[7:] = np.clip(q[7:], p.kin.q_min, p.kin.q_max)
        p.kin.forward(q, with_jac=False)
        out[t] = p.kin.positions(p.pts)[: p.S]
    return out


def test_despike_removes_injected_snap(ik_cfg):
    X = _walk_targets(ik_cfg)
    r = Retargeter(replace(ik_cfg, accel_weight=20.0))
    q = r.run(X)
    q_bad = q.copy()
    q_bad[100, 7 + 3] += 0.8          # one-frame knee-ish snap
    hits, _ = find_spikes(joint_steps(r.problem, q_bad))
    assert hits
    rep = despike(r, X, q_bad)
    assert rep.spikes_after == 0 and rep.windows
    assert np.abs(q_bad[100, 10] - q[100, 10]) < 0.3


def test_handoff_exact_with_accel_term(ik_cfg):
    cfg = replace(ik_cfg, accel_weight=50.0)
    X = _walk_targets(cfg, 240)
    r = Retargeter(cfg)
    seq = r.run(X)
    bounds = chunk_bounds(len(X), 4)
    q = np.empty_like(seq)
    for k, (s, e) in enumerate(bounds):
        a = max(0, s - 10) if k else 0
        q[s:e] = Retargeter(cfg).run(X[a:e])[s - a:]
    handoff(r, X, q, bounds, seam_tol=1e-7)
    err = max(np.abs(r.problem.difference(a, b)).max() for a, b in zip(seq, q))
    assert err < 1e-5


def test_calibration_fit_recovers_offset(ik_cfg):
    """fit_step must recover a marker's link-frame offset exactly from posed data."""
    from frdv import IKProblem
    from frdv.cli.calibrate import fit_step
    p = IKProblem(ik_cfg)
    kin = p.kin
    true = {"shank_marker": np.array([0.03, -0.02, 0.05]), "site_marker": np.array([0.01, 0.04, 0.0])}
    frames = {"shank_marker": kin.frame_point("L_shank_link"), "site_marker": kin.frame_point("L_HEEL")}
    rng = np.random.default_rng(0)
    qs = np.array([p.integrate(p.q_nominal, rng.normal(size=p.nv) * 0.4) for _ in range(30)])
    X = {k: [] for k in true}
    for q in qs:
        kin.forward(q, with_jac=False)
        for k, (b, local) in frames.items():
            R = kin.data.xmat[b].reshape(3, 3)
            X[k].append(kin.data.xpos[b] + R @ (local + true[k]))
    off, spread, n = fit_step(kin, frames, [(qs, {k: np.array(v) for k, v in X.items()})])
    assert n == 30
    for k in true:
        assert np.allclose(off[k], true[k], atol=1e-9) and spread[k] < 1e-9


def test_g1_lafan1_calibration_is_shipped():
    pytest.importorskip("hydra")
    from conftest import menagerie_or_skip
    menagerie_or_skip("unitree_g1")
    from frdv.cli.common import calibrated_keypoints, ik_config, load_calibration, to_dict
    from frdv.cli.config import compose
    c = to_dict(compose(["data=x", "dataset.augmentation=true"]))
    offsets, path = load_calibration(c["robot"], c["dataset"])
    assert offsets is not None and set(calibrated_keypoints(c["robot"])) <= set(offsets)
    ik = ik_config(c["robot"], c["solver"], ["left_thumb"], c["dataset"])
    assert isinstance(ik.targets[0], tuple) and np.allclose(ik.targets[0][1], offsets["left_thumb"])
