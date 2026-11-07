"""Public configuration scope for LAFAN1, AMASS, robots, and subject profiles."""

import pytest

pytest.importorskip("hydra")

from hydra.errors import MissingConfigException
from omegaconf import OmegaConf

from frdv import motion
from frdv.cli.config import available, cli, compose


ROBOTS = ("g1",)
REMOVED_ROBOTS = ("apollo", "berkeley_humanoid", "h1", "n1", "op3", "t1", "talos", "toddlerbot")


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setenv("FRDV_CONFIG_DIR", str(tmp_path / "frdv_conf"))
    monkeypatch.setenv("FRDV_NO_CONFIG_CACHE", "1")


def test_builtin_presets():
    from frdv.assets import MENAGERIE_MODELS
    assert available("dataset") == ["amass", "lafan1"]
    assert available("robot") == list(ROBOTS)
    assert available("scaling") == ["auto", "manual"]
    assert MENAGERIE_MODELS == {'unitree_g1': 'g1.xml'}


@pytest.mark.parametrize('robot', REMOVED_ROBOTS)
def test_archived_robot_presets_are_not_available(robot):
    with pytest.raises(MissingConfigException):
        compose([f'robot={robot}', 'data=unused'])


def test_demo_rejects_archived_robot_before_downloading(monkeypatch):
    from frdv.cli import demo
    monkeypatch.setattr(demo, 'fetch_lafan1', lambda *a: pytest.fail('invalid robot must not download data'))
    assert demo.ROBOTS == ['g1']
    with pytest.raises(SystemExit) as error:
        demo.main(['--robots', 'h1', '--no-view'])
    assert error.value.code == 2


@pytest.mark.parametrize("robot", ROBOTS)
def test_each_robot_composes_with_lafan1(robot):
    cfg = compose([f"robot={robot}", "data=clip.bvh"])
    assert cfg.dataset.name == "lafan1"
    assert cfg.dataset.format == "bvh"
    assert cfg.dataset.scale.mode == "limbs"
    assert cfg.dataset.augmentation is False
    assert not motion.enabled_markers(cfg.dataset)
    assert all(k not in motion.keypoint_map(cfg.dataset) for k, v in cfg.robot.keypoints.items()
               if v.get("calibrate"))
    assert cfg.robot.keypoints
    assert cfg.solver.despike.enabled
    # Composition must not resolve/download the robot model.
    assert OmegaConf.is_interpolation(cfg.robot, "xml")


@pytest.mark.parametrize("dataset", ["soma", "optitrack", "bvh"])
def test_removed_dataset_presets_are_rejected(dataset):
    with pytest.raises(MissingConfigException):
        compose([f"dataset={dataset}", "data=clip.bvh"])


@pytest.mark.parametrize("format", ["optitrack"])
def test_removed_motion_loaders_are_rejected(format):
    with pytest.raises(ValueError, match="unknown motion format"):
        motion.load("unused.npz", format)


def test_help_advertises_supported_datasets_and_g1(capsys):
    def unexpected_run(cfg):
        pytest.fail("--help must not execute retargeting")

    assert cli("Retarget LAFAN1", unexpected_run, argv=["--help"]) == 0
    output = capsys.readouterr().out
    assert "datasets: amass, lafan1\n" in output
    assert "scaling:  auto, manual\n" in output
    robots = next(line for line in output.splitlines() if line.startswith("robots:"))
    assert robots.partition(":")[2].strip().split(", ") == list(ROBOTS)
