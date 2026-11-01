"""Explicit subject tasks, inherited defaults, and calibration persistence."""

from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from mocap_retargeting.data_loading import mocap_transform
from mocap_retargeting.kinematic_retargeting import retarget_motion

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = "unitree_g1/subject_scaling/amass/accad/Female1Walking_c3d"


@pytest.fixture
def conf_root(tmp_path):
    path = tmp_path / "conf"
    shutil.copytree(ROOT / "conf", path)
    return path


def compose_task(root, task="unitree_g1/amass", overrides=()):
    with initialize_config_dir(config_dir=str(root), version_base=None):
        return compose(config_name="config", overrides=[f"task={task}", *overrides])


def test_subject_task_inherits_base_and_amass_and_cli_wins(conf_root):
    cfg = compose_task(conf_root, SUBJECT)
    base = compose_task(conf_root)
    assert cfg.data_type == "amass" and cfg.robot == "unitree_g1"
    assert cfg.mapping == base.mapping and cfg.foot_links == base.foot_links
    assert cfg.mujoco_dst_xml_path == base.mujoco_dst_xml_path
    assert cfg.optimizer == base.optimizer and cfg.sqp == base.sqp
    assert cfg.mocap.ground is False
    assert cfg.mocap.scale != base.mocap.scale
    assert cfg.upper_body_offset != base.upper_body_offset
    override = compose_task(conf_root, SUBJECT, ["mocap.scale=[0.6,0.7,0.8]"])
    assert list(override.mocap.scale) == [0.6, 0.7, 0.8]


def test_omitted_subject_fields_inherit_and_configs_do_not_leak(conf_root):
    path = conf_root / "task/unitree_g1/subject_scaling/amass/cmu/01.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "# @package _global_\ndefaults:\n  - /task/unitree_g1/amass\n  - _self_\nmocap:\n  scale: [0.7, 0.8, 0.9]\nupper_body_offset: [0, 0, 0]\n"
    )
    defaults = compose_task(conf_root)
    before = OmegaConf.to_container(defaults)
    cfg = compose_task(conf_root, "unitree_g1/subject_scaling/amass/cmu/01")
    transform = mocap_transform.resolve_mocap_transform(cfg)
    np.testing.assert_array_equal(transform.scale, [0.7, 0.8, 0.9])
    np.testing.assert_array_equal(
        transform.rotation.reshape(-1), defaults.mocap.rotation
    )
    np.testing.assert_array_equal(transform.offset, defaults.mocap.offset)
    np.testing.assert_array_equal(transform.upper_body_offset, [0, 0, 0])
    assert OmegaConf.to_container(defaults) == before
    assert OmegaConf.to_container(compose_task(conf_root)) == before


def test_general_amass_retains_existing_defaults():
    cfg = compose_task(ROOT / "conf")
    transform = mocap_transform.resolve_mocap_transform(cfg)
    np.testing.assert_array_equal(transform.scale, [0.87, 0.83, 0.75])
    np.testing.assert_array_equal(transform.offset, [0.04, 0, 0])
    np.testing.assert_array_equal(
        transform.rotation, np.array(cfg.mocap.rotation).reshape(3, 3)
    )
    np.testing.assert_array_equal(transform.upper_body_offset, [0.03, 0, 0.065])


@pytest.mark.parametrize(
    "field,value",
    [
        ("scale", [1, -1, 1]),
        ("scale", [1, float("nan"), 1]),
        ("offset", [0, 0]),
        ("offset", [0, float("inf"), 0]),
        ("rotation", [0] * 9),
        ("rotation", [1, 0, 0]),
        ("upper_body_offset", [0, 0]),
        ("upper_body_offset", [0, float("nan"), 0]),
    ],
)
def test_invalid_selected_transform_is_rejected(field, value):
    cfg = compose_task(ROOT / "conf")
    target = cfg if field == "upper_body_offset" else cfg.mocap
    target[field] = value
    with pytest.raises(ValueError, match=field):
        mocap_transform.resolve_mocap_transform(cfg)


def test_save_updates_only_selected_subject_and_preserves_inheritance(conf_root):
    selected = conf_root / f"task/{SUBJECT}.yaml"
    defaults = conf_root / "task/unitree_g1/amass.yaml"
    other = selected.with_name("Male1General_c3d.yaml")
    before_default, before_other = defaults.read_bytes(), other.read_bytes()
    original = yaml.safe_load(selected.read_text())
    cfg = compose_task(conf_root, SUBJECT)
    transform = mocap_transform.resolve_mocap_transform(cfg)
    saved_path = mocap_transform.save_mocap_transform(
        transform,
        selected,
        [0.7, 0.8, 0.9],
        np.eye(3),
        [0.1, -0.02, 0.03],
        upper_body_offset=[0.02, -0.01, 0.08],
    )
    assert saved_path == selected
    assert (
        defaults.read_bytes() == before_default and other.read_bytes() == before_other
    )
    saved = yaml.safe_load(selected.read_text())
    assert saved["defaults"] == original["defaults"]
    assert saved["mocap"]["offset"] == [0.1, -0.02, 0.03]
    assert saved["upper_body_offset"] == [0.02, -0.01, 0.08]
    assert "upper_body_offset" not in saved["mocap"]
    result = mocap_transform.resolve_mocap_transform(compose_task(conf_root, SUBJECT))
    np.testing.assert_array_equal(result.scale, [0.7, 0.8, 0.9])
    np.testing.assert_array_equal(result.rotation, np.eye(3))


def test_save_keeps_resolved_upper_body_offset_when_it_is_not_edited(conf_root):
    cfg = compose_task(conf_root, SUBJECT)
    transform = mocap_transform.resolve_mocap_transform(cfg)
    path = conf_root / f"task/{SUBJECT}.yaml"
    mocap_transform.save_mocap_transform(
        transform, path, transform.scale, transform.rotation, transform.offset
    )
    assert yaml.safe_load(path.read_text())["upper_body_offset"] == list(
        transform.upper_body_offset
    )


def test_lafan_does_not_apply_amass_upper_body_offsets():
    cfg = compose_task(ROOT / "conf", "unitree_g1/lafan")
    with open_dict(cfg):
        cfg.upper_body_offset = [1, 2, 3]
    assert mocap_transform.resolve_mocap_transform(cfg).upper_body_offset is None


@pytest.mark.parametrize("task", ["unitree_g1/lafan", "unitree_g1/amass", SUBJECT])
@pytest.mark.parametrize("ground", [None, False, True])
def test_retargeting_uses_selected_config_for_any_motion_path(
    conf_root, tmp_path, monkeypatch, task, ground
):
    cfg = compose_task(conf_root, task)
    if ground is not None:
        cfg.mocap.ground = ground
    original = OmegaConf.to_container(cfg)
    expected = mocap_transform.resolve_mocap_transform(cfg)
    captured = {}
    loader = SimpleNamespace(
        supports_upper_body_offset=cfg.data_type == "amass",
        preprocess=lambda **kwargs: captured.update(kwargs),
        process_all_frames_vectorized=lambda: None,
    )
    monkeypatch.setattr(retarget_motion, "load_mocap", lambda *_: loader)
    for path in (
        tmp_path / "renamed_folder/motion.npz",
        tmp_path / "ACCAD/AnotherSubject/motion.npz",
        tmp_path / "jit_warmup.npz",
    ):
        assert retarget_motion.prepare_motion(cfg, path) is loader
        np.testing.assert_array_equal(captured["scale"], expected.scale)
        np.testing.assert_array_equal(captured["rotation_matrix"], expected.rotation)
        np.testing.assert_array_equal(captured["manual_offset"], expected.offset)
        if cfg.data_type == "amass":
            np.testing.assert_array_equal(
                captured["upper_body_offset"], expected.upper_body_offset
            )
        assert captured["ground"] is cfg.mocap.ground
        assert captured["zero_root_xy"]
    assert OmegaConf.to_container(cfg) == original


@pytest.mark.parametrize(
    "path",
    sorted((ROOT / "conf/task/unitree_g1/subject_scaling/amass/accad").glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_each_accad_task_composes_and_retains_amass_coordinate_rotation(path):
    cfg = compose_task(
        ROOT / "conf", path.relative_to(ROOT / "conf/task").with_suffix("").as_posix()
    )
    default = compose_task(ROOT / "conf")
    transform = mocap_transform.resolve_mocap_transform(cfg)
    assert cfg.data_type == "amass"
    np.testing.assert_array_equal(
        transform.rotation.reshape(-1), default.mocap.rotation
    )
    assert np.all(transform.scale > 0)
