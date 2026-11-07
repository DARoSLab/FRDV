"""Manual source-axis scaling and optional synthetic orientation targets."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from frdv import motion as M
from frdv.cli.calibrate import calibrate
from frdv.cli.config import compose
from frdv.cli.retarget import _loader_opts


@pytest.fixture
def clip():
    return M.Motion(names=["Hips", "LeftFoot"], fps=30.0,
                    positions=np.array([[[5, 90, 7], [15, 10, 7]],
                                        [[5, 95, 8], [15, 15, 8]]], dtype=float))


@pytest.fixture
def dataset():
    return {"units": 0.01, "rotation": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            "root": "Hips", "keypoints": {"pelvis": "Hips", "left_ankle": "LeftFoot"},
            "scale": {"mode": "manual", "value": [0.5, 0.6, 0.7]},
            "ground": {"mode": "frame0"}}


def test_manual_scale_matches_v1_source_axis_order(clip, dataset):
    # V1 [0.005, 0.006, 0.007] includes cm -> m; V2 handles units separately.
    expected = (clip.positions * [0.005, 0.006, 0.007]) @ np.asarray(dataset["rotation"]).T
    expected -= np.r_[expected[0, 0, :2], 0.0]
    expected[..., 2] -= expected[0, :, 2].min()
    tg = M.build_targets(clip, dataset, 1.32, ["pelvis", "left_ankle"], keep_skeleton=True)
    assert np.allclose(tg.X, expected)
    assert np.allclose(tg.skeleton, expected)
    assert tg.segment_scale is None
    assert np.allclose(tg.scale, [0.5, 0.6, 0.7])
    # Source Y becomes world Z; swapping scale/rotation would yield a different height.
    assert tg.X[0, 0, 2] == pytest.approx(0.48)


def test_manual_scale_does_not_depend_on_robot_height_or_first_pose_height(clip, dataset):
    clip.positions[..., 1] = 0.0
    a = M.build_targets(clip, dataset, 1.0, ["pelvis", "left_ankle"])
    b = M.build_targets(clip, dataset, 3.0, ["pelvis", "left_ankle"])
    assert np.isfinite(a.X).all()
    assert np.array_equal(a.X, b.X)


@pytest.mark.parametrize("value", [[0, 1, 1], [-1, 1, 1], [np.nan, 1, 1], [1, 2], None])
def test_invalid_manual_scale_is_rejected(clip, dataset, value):
    dataset["scale"]["value"] = value
    with pytest.raises(ValueError, match="scal"):
        M.build_targets(clip, dataset, 1.32, ["pelvis", "left_ankle"])


def test_manual_preset_and_user_subject_profile(monkeypatch, tmp_path):
    user = tmp_path / "frdv_conf"
    (user / "scaling").mkdir(parents=True)
    (user / "scaling/subject1_g1.yaml").write_text(
        "# @package _global_\ndataset:\n  scale:\n    mode: manual\n"
        "    value: [0.7, 0.8, 0.9]\n  offset: [0.01, 0.02, -0.03]\n")
    monkeypatch.setenv("FRDV_CONFIG_DIR", str(user))
    monkeypatch.setenv("FRDV_NO_CONFIG_CACHE", "1")
    manual = compose(["scaling=manual", "data=clip.bvh"])
    assert manual.dataset.scale.mode == "manual"
    assert list(manual.dataset.scale.value) == [0.76, 0.78, 0.74]
    subject = compose(["scaling=subject1_g1", "data=clip.bvh"])
    assert list(subject.dataset.scale.value) == [0.7, 0.8, 0.9]
    assert subject.dataset.scale.mode == "manual"
    assert list(subject.dataset.offset) == [0.01, 0.02, -0.03]


def test_augmentation_switch_controls_geometry_and_targets():
    ds = {"markers": [{"name": "LeftThumb", "parent": "LeftHand", "offset": [8, 0, 5]}],
          "keypoints": {"pelvis": "Hips", "left_wrist": "LeftHand", "left_thumb": "LeftThumb"},
          "scale": {"mode": "manual", "value": [1, 1, 1]}}
    # Even a preloaded clip containing the synthetic name must not track it by default.
    clip = M.Motion(names=["Hips", "LeftHand", "LeftThumb"], fps=30,
                    positions=np.array([[[0, 0, 1], [0.3, 0, 1], [0.4, 0, 1.1]]]))
    keys = list(ds["keypoints"])
    off = M.build_targets(clip, ds, 1.0, keys)
    assert off.keypoints == ["pelvis", "left_wrist"]
    assert _loader_opts(ds) == {"markers": []}
    on_ds = {**ds, "augmentation": True}
    on = M.build_targets(clip, on_ds, 1.0, keys)
    assert on.keypoints == keys
    assert _loader_opts(on_ds)["markers"] == ds["markers"]
    assert ds["keypoints"]["left_thumb"] == "LeftThumb"  # original config was not mutated


def test_disabled_augmentation_skips_calibration(monkeypatch, tmp_path):
    cfg = OmegaConf.to_container(compose(["data=unused.bvh"]), resolve=False)

    def unexpected_fit(*args, **kwargs):
        pytest.fail("disabled augmentation must not fit markers or load a robot model")

    monkeypatch.setattr("frdv.cli.calibrate.fit_offsets", unexpected_fit)
    out = tmp_path / "markers.yaml"
    assert calibrate(cfg, ["unused.bvh"], out=out, log=lambda msg: None) is None
    assert not out.exists()


def test_lafan1_marker_opt_in_restores_all_fourteen_targets():
    off = compose(["data=unused.bvh"])
    on = compose(["data=unused.bvh", "dataset.augmentation=true"])
    assert len(M.keypoint_map(off.dataset)) == 16
    assert len(M.enabled_markers(on.dataset)) == 14
    assert len(M.keypoint_map(on.dataset)) == 30
