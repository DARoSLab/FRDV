"""Preserve saved robot motion files and optional artifact logging."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from mocap_retargeting.retargeted_motion_io import retargeted_motion_io as motion_io


@pytest.mark.parametrize("failure", ["empty", "mismatched", "nonfinite"])
def test_invalid_trajectory_is_rejected_before_creating_an_output(
    tmp_path, monkeypatch, failure
):
    monkeypatch.chdir(tmp_path)
    q = np.zeros((3, 8))
    q[:, 6] = 1
    mocap = np.zeros((3, 1, 3))
    if failure == "empty":
        q, mocap = [], []
    elif failure == "mismatched":
        mocap = mocap[:-1]
    else:
        q[0, 7] = np.nan
    cfg = OmegaConf.create(
        {"robot": "unitree_g1", "mocap": {"fps": 120}, "verbose": False}
    )
    with pytest.raises(ValueError):
        motion_io.save_motion(cfg, q, mocap, [], 0, "failed.npz", ["Pelvis"])
    assert not (tmp_path / "log").exists()


@pytest.mark.parametrize("logging_enabled", [False, True])
@pytest.mark.parametrize("source_fps", [None, 59.94])
def test_robot_motion_round_trip_and_optional_logging(
    tmp_path, monkeypatch, logging_enabled, source_fps
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(motion_io.time, "strftime", lambda _: "2000-01-02_03-04-05")
    artifacts = []

    class Artifact:
        def __init__(self, **kwargs):
            self.attributes = kwargs
            self.files = []

        def add_file(self, path):
            self.files.append(path)

    # Disabled logging must also work when the optional logger cannot be imported.
    logger = SimpleNamespace(Artifact=Artifact, log_artifact=artifacts.append)
    monkeypatch.setitem(sys.modules, "wandb", logger if logging_enabled else None)
    cfg = OmegaConf.create(
        {
            "robot": "unitree_g1",
            "mocap": {"fps": 60},
            "verbose": False,
            "wandb": {"enable": logging_enabled},
        }
    )
    q = np.zeros((3, 10))
    q[:, 6] = 1
    q[:, 0] = [0, 0.1, 0.2]
    mocap = np.arange(18, dtype=float).reshape(3, 2, 3)
    names = ["Hips", "LeftFoot"]
    skeleton = [("Hips", "LeftFoot")]
    motion_io.save_motion(
        cfg, q.tolist(), mocap.tolist(), skeleton, 11, "walk.bvh", names, fps=source_fps
    )
    path = (
        tmp_path
        / "log/kin_trajectories/unitree_g1"
        / "unitree_g1_walk_11_14_2000-01-02_03-04-05.npy"
    )
    saved = motion_io.load_trajectory(path)
    assert set(saved) == {"q", "fps", "mocap", "skeleton", "joint_names"}
    np.testing.assert_array_equal(saved["q"], q)
    np.testing.assert_array_equal(saved["mocap"], mocap)
    expected_fps = 60 if source_fps is None else source_fps
    assert saved["fps"] == expected_fps
    assert saved["skeleton"] == skeleton and saved["joint_names"] == names
    assert len(artifacts) == int(logging_enabled)
    if logging_enabled:
        artifact = artifacts[0]
        assert artifact.files == [str(path.relative_to(tmp_path))]
        assert artifact.attributes["name"] == path.name
        assert artifact.attributes["type"] == "motion"
        assert artifact.attributes["metadata"] == {
            "robot": "unitree_g1",
            "mocap": "walk",
            "start_index": 11,
            "end_index": 14,
            "fps": expected_fps,
        }
