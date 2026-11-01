"""Regression checks for explicit model selection and the standalone motion tools."""

from pathlib import Path
import shutil
from types import SimpleNamespace
from concurrent.futures import Future

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
import yaml

from mocap_retargeting.mujoco_integration.mujoco_interface import resolve_robot_xml
from mocap_retargeting.mujoco_integration.mujoco_interface import mj_to_pin
from tools import batch_retarget
from tools.video_export import render_mujoco


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "conf/task/unitree_g1/lafan.yaml"


@pytest.mark.parametrize("dataset,extension", [("lafan1", ".bvh"), ("amass", ".npz")])
def test_batch_discovery_selects_format_and_optional_subdirectories(
    tmp_path, dataset, extension
):
    (tmp_path / "nested").mkdir()
    direct = tmp_path / ("direct" + extension.upper())
    nested = tmp_path / "nested" / ("motion" + extension)
    direct.touch()
    nested.touch()
    (tmp_path / "unrelated.npy").touch()
    assert batch_retarget.discover_motions(tmp_path, dataset) == [direct]
    assert batch_retarget.discover_motions(tmp_path, dataset, recursive=True) == sorted(
        [direct, nested]
    )


@pytest.mark.parametrize("dataset", ["lafan1", "amass"])
def test_batch_command_composes_the_selected_dataset_and_preserves_paths(
    tmp_path, monkeypatch, dataset
):
    task, extension = batch_retarget.DATASETS[dataset]
    path = tmp_path / ("clip = 1" + extension)
    model_path = str(tmp_path / "body models") if dataset == "amass" else None
    job = batch_retarget.RetargetJob(
        str(path),
        f"unitree_g1/{task}",
        start_index=1,
        end_index=3,
        smplh_model_path=model_path,
    )
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout="Total Frames Retargeted: 2\n", stderr="")

    monkeypatch.setattr(batch_retarget.subprocess, "run", run)
    assert batch_retarget.retarget_file(job)[0]
    cmd, options = calls[0]
    config_dir = tmp_path / "conf"
    shutil.copytree(ROOT / "conf", config_dir)
    base_path = config_dir / "base.yaml"
    base = yaml.safe_load(base_path.read_text())
    base["n_motion_chunk_workers"] = 7
    base_path.write_text(yaml.safe_dump(base))
    assert not any(arg.startswith("n_motion_chunk_workers=") for arg in cmd)
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=cmd[3:])
    assert cfg.data_type == dataset
    assert cfg.mocap_src_path == str(path)
    assert (cfg.start_index, cfg.end_index, cfg.n_motion_chunk_workers) == (1, 3, 7)
    assert cfg.jit_compile is True
    assert cfg.jit_recompile is False
    if dataset == "amass":
        assert cfg.smplh_model_path == model_path
    assert options["cwd"] == ROOT


@pytest.mark.parametrize(
    "output", ["Total Frames Retargeted: 0\n", "Skipped invalid input.\n"]
)
def test_batch_does_not_report_skipped_inputs_as_success(monkeypatch, output):
    monkeypatch.setattr(
        batch_retarget.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout=output, stderr=""),
    )
    assert not batch_retarget.retarget_file(
        batch_retarget.RetargetJob("invalid.bvh", "unitree_g1/lafan")
    )[0]


@pytest.mark.parametrize("use_jit", [True, False])
def test_batch_prepares_before_launching_every_full_motion(
    tmp_path, monkeypatch, use_jit
):
    first = tmp_path / "a.bvh"
    second = tmp_path / "b.bvh"
    first.touch()
    second.touch()
    events, submitted = [], []

    def prepare(job, frames):
        events.append("warmup")
        assert job.motion_path == str(first)
        assert frames == 3
        return True

    class Executor:
        def __init__(self, max_workers):
            assert events == (["warmup"] if use_jit else [])
            assert max_workers == 4
            events.append("pool")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, job):
            submitted.append(job)
            future = Future()
            future.set_result((True, job.motion_path, ""))
            return future

    monkeypatch.setattr(batch_retarget, "prepare_jit_cache", prepare)
    monkeypatch.setattr(
        batch_retarget.concurrent.futures, "ProcessPoolExecutor", Executor
    )
    arguments = [
        "--input_dir",
        str(tmp_path),
        "--num_workers",
        "4",
        "--start_index",
        "1",
        "--end_index",
        "8",
        "--no_wandb",
    ]
    if not use_jit:
        arguments += ["--no-jit"]
    assert batch_retarget.main(arguments) == 0
    assert [job.motion_path for job in submitted] == [str(first), str(second)]
    assert all((job.start_index, job.end_index) == (1, 8) for job in submitted)
    assert all(
        job.jit_compile is use_jit and not job.jit_recompile for job in submitted
    )


def test_failed_jit_warmup_stops_before_creating_workers(tmp_path, monkeypatch):
    (tmp_path / "motion.bvh").touch()
    monkeypatch.setattr(batch_retarget, "prepare_jit_cache", lambda *args: False)

    def unexpected_pool(*args, **kwargs):
        pytest.fail("No workers should start after a failed JIT warmup")

    monkeypatch.setattr(
        batch_retarget.concurrent.futures, "ProcessPoolExecutor", unexpected_pool
    )
    assert batch_retarget.main(["--input_dir", str(tmp_path)]) == 1


def test_batch_selected_subject_task_reaches_warmup_and_every_job(
    tmp_path, monkeypatch
):
    for name in ("first.npz", "second.npz"):
        (tmp_path / name).touch()
    task = "unitree_g1/subject_scaling/amass/accad/Female1Walking_c3d"
    warmups, jobs = [], []

    def warmup(job, frames):
        warmups.append(job)
        return True

    class Executor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, job):
            jobs.append(job)
            future = Future()
            future.set_result((True, job.motion_path, ""))
            return future

    monkeypatch.setattr(batch_retarget, "prepare_jit_cache", warmup)
    monkeypatch.setattr(
        batch_retarget.concurrent.futures, "ProcessPoolExecutor", Executor
    )
    assert (
        batch_retarget.main(
            [
                "--input_dir",
                str(tmp_path),
                "--dataset",
                "amass",
                "--task",
                task,
                "--smplh_model_path",
                str(tmp_path),
                "--num_workers",
                "2",
                "--no_wandb",
            ]
        )
        == 0
    )
    assert len(warmups) == 1 and len(jobs) == 2
    assert all(job.task == task for job in [*warmups, *jobs])


@pytest.mark.parametrize("task", ["unitree_g1/lafan", "missing_subject"])
def test_batch_rejects_mismatched_or_unavailable_tasks(tmp_path, task):
    (tmp_path / "motion.npz").touch()
    with pytest.raises(SystemExit) as exc:
        batch_retarget.main(
            [
                "--input_dir",
                str(tmp_path),
                "--dataset",
                "amass",
                "--task",
                task,
                "--smplh_model_path",
                str(tmp_path),
            ]
        )
    assert exc.value.code == 2


@pytest.mark.parametrize("success", [True, False])
def test_warmup_compiles_once_and_removes_only_its_own_partial_output(
    tmp_path, monkeypatch, success
):
    source = tmp_path / "source.bvh"
    source.write_text("original motion")
    job = batch_retarget.RetargetJob(
        str(source), "unitree_g1/lafan", wandb_enable=True, start_index=4, end_index=5
    )
    trajectories = tmp_path / "log/kin_trajectories/unitree_g1"
    trajectories.mkdir(parents=True)
    existing = trajectories / "existing.npy"
    existing.write_bytes(b"keep this output")
    calls = []

    def run(warmup, *, warmup_dir):
        alias = Path(warmup.motion_path)
        assert alias.is_symlink() and alias.read_text() == source.read_text()
        assert warmup.jit_compile and warmup.jit_recompile
        assert (warmup.start_index, warmup.end_index) == (4, 5)
        assert not warmup.wandb_enable and warmup.wandb_project is None
        partial = trajectories / f"unitree_g1_{alias.stem}_4_5_timestamp.npy"
        partial.write_bytes(b"temporary")
        calls.append((alias, warmup_dir, partial))
        return success, str(alias), ""

    monkeypatch.setattr(batch_retarget, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(batch_retarget, "retarget_file", run)
    assert batch_retarget.prepare_jit_cache(job, 3) is success
    alias, directory, partial = calls[0]
    assert not alias.exists() and not directory.exists() and not partial.exists()
    assert existing.read_bytes() == b"keep this output"
    assert source.read_text() == "original motion"
    assert (job.start_index, job.end_index) == (4, 5)


def test_warmup_command_keeps_cache_in_place_and_disables_auxiliary_processing(
    tmp_path, monkeypatch
):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout="Total Frames Retargeted: 3\n", stderr="")

    monkeypatch.setattr(batch_retarget.subprocess, "run", run)
    job = batch_retarget.RetargetJob(
        "first.bvh", "unitree_g1/lafan", end_index=3, jit_recompile=True
    )
    assert batch_retarget.retarget_file(job, warmup_dir=tmp_path)[0]
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(
            config_name="config", overrides=calls[0][3:], return_hydra_config=True
        )
    assert cfg.jit_compile and cfg.jit_recompile
    assert cfg.n_motion_chunk_workers == 1
    assert not cfg.enable_refine and not cfg.save_chunks and not cfg.wandb.enable
    assert cfg.output_dir == str(tmp_path)
    assert cfg.hydra.run.dir == str(tmp_path)
    assert cfg.hydra.job.chdir is False


@pytest.mark.parametrize("task", ["lafan", "amass"])
def test_bundled_model_config_resolves_outside_the_repository(
    tmp_path, monkeypatch, task
):
    monkeypatch.chdir(tmp_path)
    config = ROOT / f"conf/task/unitree_g1/{task}.yaml"
    expected = ROOT / yaml.safe_load(config.read_text())["mujoco_dst_xml_path"]
    assert resolve_robot_xml(config) == expected


def test_custom_model_path_is_relative_to_its_config(tmp_path):
    xml = tmp_path / "robot.xml"
    xml.write_text("<mujoco/>")
    config = tmp_path / "task.yaml"
    config.write_text("mujoco_dst_xml_path: robot.xml\n")
    assert resolve_robot_xml(config) == xml


@pytest.mark.parametrize("subject", ["Female1Walking_c3d", "Male1General_c3d"])
def test_subject_config_inherits_the_amass_robot_xml(subject, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = ROOT / f"conf/task/unitree_g1/subject_scaling/amass/accad/{subject}.yaml"
    expected = resolve_robot_xml(ROOT / "conf/task/unitree_g1/amass.yaml")
    assert resolve_robot_xml(path) == expected


@pytest.mark.parametrize("contents", ["", "[]", "robot: unitree_g1\n"])
def test_missing_model_selection_is_an_error(tmp_path, contents):
    config = tmp_path / "task.yaml"
    config.write_text(contents)
    with pytest.raises(ValueError):
        resolve_robot_xml(config)


def test_missing_processed_model_does_not_silently_select_the_source(tmp_path):
    source = tmp_path / "source.xml"
    source.write_text("<mujoco/>")
    config = tmp_path / "task.yaml"
    config.write_text(
        yaml.safe_dump(
            {"mujoco_dst_xml_path": "missing.xml", "mujoco_src_xml_path": str(source)}
        )
    )
    with pytest.raises(FileNotFoundError, match="missing.xml"):
        resolve_robot_xml(config)


@pytest.mark.parametrize("arguments", [[], ["--config", "missing-task.yaml"]])
def test_playback_rejects_missing_config_before_starting_a_server(
    tmp_path, monkeypatch, arguments
):
    from mocap_retargeting import play
    from mocap_retargeting.visualization import playback_app

    def unexpected_server(*args, **kwargs):
        pytest.fail("A server must not start before configuration validation")

    monkeypatch.setattr(playback_app.viser, "ViserServer", unexpected_server)
    monkeypatch.setattr(play, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["play", *arguments])
    with pytest.raises(SystemExit) as exc:
        play.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("task", [
    None,
    "unitree_g1/amass",
    "unitree_g1/subject_scaling/amass/accad/Female1Walking_c3d",
])
def test_playback_uses_default_lafan_config_or_explicit_override(
    tmp_path, monkeypatch, task
):
    from mocap_retargeting import play

    config = ROOT / "conf/task" / ((task or "unitree_g1/lafan") + ".yaml")
    captured = {}

    class App:
        def __init__(self, args, xml_path):
            captured.update(args=args, xml_path=Path(xml_path))

        def run(self):
            return "playback"

    monkeypatch.setattr(play, "PlaybackApp", App)
    arguments = ["play"] if task is None else ["play", "--config", str(config)]
    monkeypatch.setattr("sys.argv", arguments)
    monkeypatch.chdir(tmp_path)
    assert play.main() == "playback"
    assert Path(captured["args"].config) == config
    assert captured["args"].path == "log/kin_trajectories/unitree_g1/"
    assert captured["xml_path"] == resolve_robot_xml(config)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--width", "0"],
        ["--width", "321"],
        ["--camera-distance", "-1"],
        ["--camera-elevation", "nan"],
    ],
)
def test_render_rejects_invalid_camera_or_resolution_arguments(arguments):
    with pytest.raises(SystemExit) as exc:
        render_mujoco.parse_args(["--config", str(CONFIG), *arguments])
    assert exc.value.code == 2


@pytest.mark.parametrize("custom", [False, True])
def test_render_applies_camera_and_resolution_arguments(tmp_path, monkeypatch, custom):
    model = mujoco.MjModel.from_xml_path(str(resolve_robot_xml(CONFIG)))
    q = mj_to_pin(model.key_qpos[0])
    trajectory = tmp_path / "trajectory.npy"
    np.save(trajectory, {"q": np.array([q]), "fps": 60})
    renderers, saved = [], []

    class Renderer:
        def __init__(self, model, height, width):
            self.height, self.width, self.closed = height, width, False
            assert model.vis.global_.offwidth >= width
            assert model.vis.global_.offheight >= height
            renderers.append(self)

        def update_scene(self, data, camera, scene_option):
            self.camera = (camera.distance, camera.azimuth, camera.elevation)
            self.lookat = camera.lookat.copy()

        def render(self):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def close(self):
            self.closed = True

    monkeypatch.setattr(render_mujoco.mujoco, "Renderer", Renderer)
    monkeypatch.setattr(
        render_mujoco.imageio,
        "mimsave",
        lambda path, frames, fps, **kw: saved.append((path, frames, fps, kw)),
    )
    monkeypatch.chdir(tmp_path)
    arguments = ["--config", str(CONFIG), "--path", str(trajectory)]
    if custom:
        arguments += [
            "--width",
            "1920",
            "--height",
            "1080",
            "--camera-distance",
            "4",
            "--camera-azimuth-offset",
            "135",
            "--camera-elevation",
            "-25",
            "--track-body",
            "pelvis",
        ]
    render_mujoco.main(arguments)
    body_name = "pelvis" if custom else "head_link"
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    assert body_id >= 0
    data = mujoco.MjData(model)
    data.qpos = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    forward = data.xmat[body_id].reshape(3, 3) @ np.array([1, 0, 0])
    yaw = np.degrees(np.arctan2(forward[1], forward[0]))
    expected_camera = (4, yaw + 135, -25) if custom else (3, yaw + 180, -15)
    np.testing.assert_allclose(renderers[0].camera, expected_camera)
    np.testing.assert_allclose(renderers[0].lookat, data.xpos[body_id])
    assert renderers[0].closed
    assert saved[0][1][0].shape == ((1080, 1920, 3) if custom else (720, 1280, 3))
    assert saved[0][2] == 60
    assert saved[0][3]["macro_block_size"] == 1


def test_explicit_tracking_body_must_exist():
    model = mujoco.MjModel.from_xml_path(str(resolve_robot_xml(CONFIG)))
    with pytest.raises(ValueError, match="does not exist"):
        render_mujoco.tracking_body_id(model, "missing_body")
