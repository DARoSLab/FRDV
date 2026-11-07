"""SMPL+H input contract, compact FK, and persistent AMASS subject profiles."""

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from frdv import motion as M
from frdv.cli.common import find_files, target_offset, validate_subject_input
from frdv.cli.config import available, compose
from frdv.cli.retarget import _save
from frdv.motion import amass
from frdv.motion._smplh import _load_smplh_model_data
from frdv.motion.amass_joints import BODY_JOINT_NAMES, BODY_PARENTS, UPPER_BODY_JOINT_NAMES


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setenv("FRDV_CONFIG_DIR", str(tmp_path / "frdv_conf"))
    monkeypatch.setenv("FRDV_NO_CONFIG_CACHE", "1")


@pytest.fixture
def model(tmp_path):
    """Small synthetic model with the real 52-joint/16-beta SMPL+H schema."""
    parents = [-1 if BODY_PARENTS[n] is None else BODY_JOINT_NAMES.index(BODY_PARENTS[n])
               for n in BODY_JOINT_NAMES] + [20] * 30
    rng = np.random.default_rng(31)
    arrays = dict(J_regressor=np.eye(52), v_template=rng.normal(size=(52, 3)),
                  shapedirs=rng.normal(scale=0.02, size=(52, 3, 16)),
                  kintree_table=np.array([parents, np.arange(52)]))
    path = tmp_path / "models/male/model.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, **arrays)
    return path.parent.parent, arrays


@pytest.fixture
def sample():
    rng = np.random.default_rng(47)
    return dict(poses=rng.normal(scale=0.4, size=(7, 156)), trans=rng.normal(size=(7, 3)),
                betas=np.arange(16, dtype=float) / 10, gender=np.array("male"),
                mocap_framerate=np.array(120.0))


def write_clip(path, sample, **updates):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **(sample | updates))
    return str(path)


def test_fk_matches_homogeneous_reference_and_uses_all_betas(model, sample, tmp_path):
    root, arrays = model
    path = write_clip(tmp_path / "clip.npz", sample)
    loaded = amass.load(path, smplh_model_path=str(root))
    # Independent V1-style FK: shape all joints, then accumulate homogeneous matrices.
    joints = arrays["J_regressor"] @ (
        arrays["v_template"] + np.einsum("vcb,b->vc", arrays["shapedirs"], sample["betas"]))
    expected = []
    for pose, trans in zip(sample["poses"], sample["trans"]):
        transforms = []
        for j in range(22):
            parent = int(arrays["kintree_table"][0, j])
            local = np.eye(4)
            local[:3, :3] = Rotation.from_rotvec(pose[j * 3:j * 3 + 3]).as_matrix()
            local[:3, 3] = joints[j] if j == 0 else joints[j] - joints[parent]
            transforms.append(local if j == 0 else transforms[parent] @ local)
        expected.append(np.array([t[:3, 3] for t in transforms]) + trans)
    np.testing.assert_allclose(loaded.positions, expected, atol=1e-12)
    assert loaded.fps == 120
    assert loaded.names == BODY_JOINT_NAMES
    np.testing.assert_allclose(loaded.rest[0], 0)
    assert len(loaded.parents) == 22

    changed = sample["poses"].copy()
    changed[:, 66:] += 10  # Finger poses do not move body targets.
    path = write_clip(tmp_path / "finger.npz", sample, poses=changed)
    np.testing.assert_array_equal(amass.load(path, str(root)).positions, loaded.positions)
    betas = sample["betas"].copy()
    betas[15] += 1
    path = write_clip(tmp_path / "shape.npz", sample, betas=betas)
    assert not np.allclose(amass.load(path, str(root)).positions, loaded.positions)


@pytest.mark.parametrize("field,value,message", [
    ("poses", np.zeros((2, 165)), "poses"),
    ("poses", np.zeros((0, 156)), "poses"),
    ("trans", np.zeros((1, 3)), "trans"),
    ("betas", np.zeros(10), "betas"),
    ("betas", np.full(16, np.nan), "finite"),
    ("gender", np.array("unknown"), "gender"),
    ("mocap_framerate", np.array(0.0), "frame rate"),
    ("mocap_framerate", np.array(np.inf), "frame rate"),
])
def test_bad_motion_rejected(tmp_path, sample, field, value, message):
    path = write_clip(tmp_path / "bad.npz", sample, **{field: value})
    with pytest.raises(ValueError, match=message):
        amass.read_motion(path)


@pytest.mark.parametrize("field", ["poses", "trans", "betas", "gender", "mocap_framerate"])
def test_missing_metadata_is_not_invented(tmp_path, sample, field):
    del sample[field]
    path = write_clip(tmp_path / "missing.npz", sample)
    with pytest.raises(ValueError, match="missing|requires"):
        amass.read_motion(path)


def test_alternate_fps_and_gender_bytes(tmp_path, sample):
    del sample["mocap_framerate"]
    path = write_clip(tmp_path / "clip.npz", sample, mocap_frame_rate=60.0, gender=np.array(b"male"))
    assert amass.read_motion(path)[3:] == ("male", 60.0)
    path = write_clip(tmp_path / "conflict.npz", sample, mocap_framerate=120.0, mocap_frame_rate=60.0)
    with pytest.raises(ValueError, match="disagree"):
        amass.read_motion(path)


def test_model_cache_refreshes_and_gender_never_falls_back(model):
    root, arrays = model
    a = _load_smplh_model_data(str(root), "male")
    assert a is _load_smplh_model_data(str(root), "male")
    with pytest.raises(FileNotFoundError, match="female"):
        _load_smplh_model_data(str(root), "female")
    arrays["v_template"] = arrays["v_template"] + 1
    np.savez(root / "male/model.npz", **arrays)
    b = _load_smplh_model_data(str(root), "male")
    np.testing.assert_allclose(b["J_template"], a["J_template"] + 1)
    arrays["shapedirs"] = arrays["shapedirs"][..., :10]
    np.savez(root / "male/model.npz", **arrays)
    with pytest.raises(ValueError, match="16-coefficient"):
        _load_smplh_model_data(str(root), "male")


def test_v1_manual_alignment_and_no_frame_zero_grounding(model, sample, tmp_path):
    root, _ = model
    mot = amass.load(write_clip(tmp_path / "clip.npz", sample), str(root))
    ds = OmegaConf.to_container(compose(["dataset=amass", "data=unused.npz"]).dataset, resolve=True)
    expected = (mot.positions * ds["scale"]["value"]) @ np.array(ds["rotation"]).T
    upper = [mot.names.index(n) for n in UPPER_BODY_JOINT_NAMES]
    expected[:, upper] += ds["upper_body_offset"]
    expected -= np.r_[expected[0, 0, :2], 0]
    expected += ds["offset"]
    for keep in (False, True):
        tg = M.build_targets(mot, ds, 1.32, list(ds["keypoints"]), keep_skeleton=keep)
        indices = [mot.names.index(n) for n in tg.source_joints]
        np.testing.assert_allclose(tg.X, expected[:, indices], atol=1e-12)
        if keep:
            np.testing.assert_allclose(tg.skeleton, expected, atol=1e-12)
    shifted = M.Motion(names=mot.names, positions=mot.positions + [0, 0, 5], fps=mot.fps,
                       rest=mot.rest, parents=mot.parents)
    raised = M.build_targets(shifted, ds, 1.32, list(ds["keypoints"]))
    np.testing.assert_allclose(raised.X - tg.X, np.broadcast_to([0, 0, 3.75], tg.X.shape), atol=1e-12)
    assert raised.human_height == tg.human_height  # Shape/rest height ignores starting pose.
    assert target_offset({"target_offset": [0, 0, 0.03]}, ds) == [0, 0, 0]


def test_dataset_defaults_profiles_and_explicit_scaling_override():
    assert len(available("subject")) == 21
    for profile in available("subject"):
        if not profile.startswith('unitree_g1/'):
            continue
        cfg = compose(["dataset=amass", f"subject={profile}", "data=unused.npz"])
        assert cfg.subject.collection == "accad"
        assert cfg.dataset.scale.mode == "manual"
        assert cfg.dataset.scale.value == cfg.dataset.manual_scale
        assert not cfg.dataset.augmentation
    base = compose(["dataset=amass", "data=unused.npz"])
    assert list(base.dataset.scale.value) == [0.87, 0.83, 0.75]
    profile = "subject=amass/accad/Female1Walking_c3d"
    auto = compose(["dataset=amass", profile, "scaling=auto", "data=unused.npz"])
    assert auto.dataset.scale.mode == "limbs"
    assert auto.subject.name == "Female1Walking_c3d"
    with pytest.raises(ValueError, match="requires dataset"):
        compose([profile, "data=unused.npz"])


def test_other_amass_collections_need_only_a_config(tmp_path):
    path = tmp_path / "frdv_conf/calibration/unitree_g1/amass/subjects/cmu/person42.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("# @package _global_\nsubject:\n  name: person42\n  collection: cmu\n"
                    "  dataset: amass\n  robot: unitree_g1\ndataset:\n"
                    "  manual_scale: [0.71, 0.82, 0.93]\n")
    assert "unitree_g1/amass/subjects/cmu/person42" in available("subject")
    cfg = compose(["dataset=amass", "subject=amass/cmu/person42", "data=unused.npz"])
    assert cfg.subject.collection == "cmu"
    assert list(cfg.dataset.scale.value) == [0.71, 0.82, 0.93]


def test_selected_profile_rejects_mixed_body_shapes(tmp_path, sample):
    first = write_clip(tmp_path / "a.npz", sample)
    second = write_clip(tmp_path / "b.npz", sample)
    cfg = {"dataset": {"format": "amass"}, "subject": {"name": "person"}}
    validate_subject_input(cfg, [first, second])
    betas = sample["betas"].copy()
    betas[-1] += 0.01
    write_clip(tmp_path / "b.npz", sample, betas=betas)
    with pytest.raises(ValueError, match="Multiple body-shape fits"):
        validate_subject_input(cfg, [first, second])
    # Shared default allows different bodies; no subject fit is claimed.
    validate_subject_input({"dataset": cfg["dataset"], "subject": {"name": None}}, [first, second])


def test_nested_inputs_outputs_and_viewer_preserve_same_clip_names(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    files = [write_clip(data / subject / "walk.npz", {}) for subject in ("person1", "person2")]
    assert find_files(str(data), "**/*.npz") == files
    tg = SimpleNamespace(fps=120, X=np.zeros((2, 1, 3)), keypoints=["pelvis"],
                         source_joints=["Spine1"], skeleton=None)
    cfg = {"data": str(data), "robot": {"name": "unitree_g1"}, "dataset": {"name": "amass"},
           "subject": {"name": None}}
    q = np.zeros((2, 7)); q[:, 3] = 1
    ik = SimpleNamespace(targets=["pelvis"], xml_path="toy.xml")
    for path in files:
        _save(path, str(out), tg, q.copy(), cfg, ik, ["root"])
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.npz")) == [
        "person1/walk.npz", "person2/walk.npz"]
    pytest.importorskip("viser")
    from frdv.cli.view import load_clips
    clips = load_clips(str(out), clip="walk")
    assert {c["clip"] for c in clips} == {"person1/walk", "person2/walk"}
