"""Exercise application initialization and GUI callbacks without opening a server."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

from mocap_retargeting.data_loading.mocap_transform import MocapTransform
from mocap_retargeting.mocap_calibration import calibration_app
from mocap_retargeting.mujoco_integration.mujoco_interface import mj_to_pin
from mocap_retargeting.visualization import playback_app
from mocap_retargeting.visualization.mujoco_scene import ViserMujocoScene

ROOT = Path(__file__).resolve().parents[1]


class Handle:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.value = kwargs.get("initial_value", False)
        self.visible = True
        self.callbacks = []
        self.removed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def on_click(self, callback):
        self.callbacks.append(callback)
        return callback

    def on_update(self, callback):
        self.callbacks.append(callback)
        return callback

    def remove(self):
        self.removed = True

    def __getattr__(self, name):
        if name.startswith("add_"):
            return lambda *args, **kwargs: Handle(**kwargs)
        raise AttributeError(name)


class Api:
    def __init__(self):
        self.handles = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            handle = Handle(**kwargs)
            self.handles.append((name, args, handle))
            return handle

        return call


class Server:
    def __init__(self, **kwargs):
        self.scene = Api()
        self.gui = Api()

    def flush(self):
        pass

    def atomic(self):
        return Handle()

    def get_clients(self):
        return {}

    def on_client_connect(self, callback):
        self.client_callback = callback
        return callback


def config():
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        return compose(config_name="config")


def test_playback_initializes_and_callbacks_control_real_robot_frames(
    tmp_path, monkeypatch
):
    cfg = config()
    xml = str(ROOT / cfg.mujoco_dst_xml_path)
    model = mujoco.MjModel.from_xml_path(xml)
    q = mj_to_pin(model.key_qpos[0])
    poses = np.tile(q, (3, 1))
    poses[:, 0] += [0, 0.1, 0.2]
    path = tmp_path / "motion.npy"
    np.save(path, {"q": poses, "fps": 30})
    monkeypatch.setattr(playback_app.viser, "ViserServer", Server)
    app = playback_app.PlaybackApp(
        SimpleNamespace(path=str(path), num=1, quat_format="xyzw", no_mesh=False), xml
    )
    app.initialize()
    assert app.state["total_frames"] == 3
    assert len(app.state["motions"]) == 1
    app._on_play_btn(None)
    assert app.state["playing"]
    app._on_pause_btn(None)
    assert app.state["paused"]
    app._on_stop_btn(None)
    assert not app.state["playing"] and app.state["current_frame"] == 0
    app.frame_slider.value = 2
    app._on_frame_slider(None)
    assert app.state["current_frame"] == 2
    np.testing.assert_allclose(app.state["mj_data"].qpos[:3], poses[2, :3])
    app.mocap_checkbox.value = True
    app._on_mocap_checkbox(None)
    assert app.state["show_mocap"]
    app.sites_checkbox.value = False
    app._on_sites_checkbox(None)
    app.update_visualization(1)
    assert not app.state["show_sites"]


def test_calibration_callbacks_export_to_selected_config_and_xml(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    cfg = config()
    source_xml = ROOT / cfg.mujoco_src_xml_path
    cfg.mujoco_src_xml_path = str(source_xml)
    cfg.mujoco_dst_xml_path = str(tmp_path / "calibrated.xml")
    config_path = tmp_path / "conf/task/unitree_g1/lafan.yaml"
    config_path.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "conf/task/unitree_g1/lafan.yaml", config_path)
    monkeypatch.setattr(calibration_app, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(calibration_app.viser, "ViserServer", Server)
    app = calibration_app.CalibrationApp(cfg)
    app.initialize()
    np.testing.assert_array_equal(
        [app.scale_x_slider.value, app.scale_y_slider.value, app.scale_z_slider.value],
        cfg.mocap.scale,
    )
    np.testing.assert_array_equal(app.loader.rotation.reshape(-1), cfg.mocap.rotation)
    assert app.point_cloud_handle is not None and app.bones_handle is not None
    app.x_offset_slider.value = 0.123
    app.update_visualization()
    app._on_save_all_button(None)
    saved = yaml.safe_load(config_path.read_text())
    assert saved["mocap"]["offset"][0] == 0.123
    assert Path(cfg.mujoco_dst_xml_path).is_file()
    assert saved["data_type"] == "lafan1"


def test_calibration_preserves_full_profile_rotation_and_saves_only_subject(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(ROOT)
    cfg = config()
    cfg.mujoco_src_xml_path = str(ROOT / cfg.mujoco_src_xml_path)
    cfg.mujoco_dst_xml_path = str(tmp_path / "calibrated.xml")
    task = tmp_path / "conf/task/unitree_g1/lafan.yaml"
    task.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "conf/task/unitree_g1/lafan.yaml", task)
    original_task = task.read_bytes()
    profile = task.parent / "subject_scaling/amass/accad/Female1General_c3d.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        "# @package _global_\ndefaults:\n  - /task/unitree_g1/lafan\n  - _self_\n"
    )
    angle = np.deg2rad(10)
    tilt = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )
    rotation = tilt @ np.array(cfg.mocap.rotation).reshape(3, 3)
    transform = MocapTransform(
        scale=np.array([0.006, 0.008, 0.007]),
        rotation=rotation,
        offset=np.array([0.12, -0.03, 0.08]),
    )
    monkeypatch.setattr(calibration_app, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(calibration_app.viser, "ViserServer", Server)
    monkeypatch.setattr(
        calibration_app, "resolve_mocap_transform", lambda *_: transform
    )
    app = calibration_app.CalibrationApp(cfg, config_path=profile)
    app.initialize()
    np.testing.assert_array_equal(app.loader.scale, transform.scale)
    np.testing.assert_array_equal(app.loader.manual_offset, transform.offset)
    np.testing.assert_allclose(app.loader.rotation, rotation)
    assert app.yaw_slider.value == 0
    app.yaw_slider.value = 15
    app.update_visualization()
    yaw = np.deg2rad(15)
    expected = (
        np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )
        @ rotation
    )
    np.testing.assert_allclose(app.loader.rotation, expected)
    app._on_save_all_button(None)
    saved = yaml.safe_load(profile.read_text())["mocap"]
    np.testing.assert_allclose(np.array(saved["rotation"]).reshape(3, 3), expected)
    np.testing.assert_array_equal(saved["scale"], transform.scale)
    np.testing.assert_array_equal(saved["offset"], transform.offset)
    assert task.read_bytes() == original_task


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("task", [
    "unitree_g1/lafan", "unitree_g1/amass",
    "unitree_g1/subject_scaling/amass/accad/Female1Walking_c3d",
])
def test_process_command_routes_save_to_the_selected_yaml(monkeypatch, direct, task):
    from mocap_retargeting import process

    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(
            config_name=f"task/{task}" if direct else "config",
            overrides=[] if direct else [f"task={task}"],
            return_hydra_config=True,
        )
    captured = {}

    class App:
        def __init__(self, cfg, *, config_path):
            captured["path"] = config_path

        def run(self):
            return "calibration"

    monkeypatch.setattr(process, "CalibrationApp", App)
    runtime = HydraConfig.instance()
    previous = runtime.cfg
    try:
        runtime.set_config(cfg)
        assert process.main.__wrapped__(cfg) == "calibration"
    finally:
        runtime.cfg = previous
    assert Path(captured["path"]) == ROOT / "conf/task" / (task + ".yaml")


def selected_calibration_path(cfg, monkeypatch):
    from mocap_retargeting import process

    captured = {}

    class App:
        def __init__(self, cfg, *, config_path):
            captured["path"] = Path(config_path)

        def run(self):
            return "calibration"

    monkeypatch.setattr(process, "CalibrationApp", App)
    runtime = HydraConfig.instance()
    previous = runtime.cfg
    try:
        runtime.set_config(cfg)
        assert process.main.__wrapped__(cfg) == "calibration"
    finally:
        runtime.cfg = previous
    return captured["path"]


@pytest.mark.parametrize("name", ["custom_subject", "custom_subject.yaml", "config"])
def test_process_saves_to_the_external_main_config(tmp_path, monkeypatch, name):
    folder = tmp_path / "external configs"
    folder.mkdir()
    path = folder / (name.removesuffix(".yaml") + ".yaml")
    path.write_text("review_marker: external\n")
    with initialize_config_dir(config_dir=str(folder), version_base=None):
        cfg = compose(config_name=name, return_hydra_config=True)
    assert cfg.review_marker == "external"
    assert selected_calibration_path(cfg, monkeypatch) == path


def test_process_resolves_task_in_hydra_search_path_order(tmp_path, monkeypatch):
    main = tmp_path / "main"
    main.mkdir()
    sources = [tmp_path / "first", tmp_path / "second"]
    for index, folder in enumerate(sources):
        (folder / "task").mkdir(parents=True)
        (folder / "task/subject.yaml").write_text(
            f"# @package _global_\nreview_marker: source_{index}\n"
        )
    (main / "config.yaml").write_text(yaml.safe_dump({
        "defaults": [{"task": "subject"}, "_self_"],
        "hydra": {"searchpath": [folder.as_uri() for folder in sources]},
    }))
    with initialize_config_dir(config_dir=str(main), version_base=None):
        cfg = compose(config_name="config", return_hydra_config=True)
    assert cfg.review_marker == "source_0"
    monkeypatch.chdir(tmp_path)
    assert selected_calibration_path(cfg, monkeypatch) == sources[0] / "task/subject.yaml"


@pytest.mark.parametrize("rates", [(30, 120), (30, 30)])
def test_playback_warns_once_for_mixed_fps_and_keeps_frame_indexing(
    tmp_path, monkeypatch, caplog, rates
):
    cfg = config()
    xml = str(ROOT / cfg.mujoco_dst_xml_path)
    model = mujoco.MjModel.from_xml_path(xml)
    poses = np.tile(mj_to_pin(model.key_qpos[0]), (3, 1))
    poses[:, 0] = [0, 0.1, 0.2]
    for index, rate in enumerate(rates):
        np.save(tmp_path / f"motion_{index}.npy", {"q": poses, "fps": rate})
    monkeypatch.setattr(playback_app.viser, "ViserServer", Server)
    app = playback_app.PlaybackApp(
        SimpleNamespace(path=str(tmp_path), num=2, quat_format="xyzw", no_mesh=False), xml
    )
    app.initialize()
    app.update_visualization(1)
    app.update_visualization(2)
    warnings = [record for record in caplog.records if "different FPS" in record.message]
    assert len(warnings) == int(rates[0] != rates[1])
    if warnings:
        assert "30, 120" in warnings[0].message
        assert "first motion's rate (30 FPS)" in warnings[0].message
    last = app.state["motions"][-1]
    assert app.state["total_frames"] == 3
    np.testing.assert_allclose(
        app.state["mj_data"].qpos[:3], poses[2, :3] + last["offset"]
    )


def test_scene_contact_and_debug_helpers_update_real_mujoco_geometry():
    cfg = config()
    model = mujoco.MjModel.from_xml_path(str(ROOT / cfg.mujoco_dst_xml_path))
    scene = ViserMujocoScene.create(Server(), model, num_envs=1)
    scene.create_visualization_gui(show_debug_viz_control=True)
    scene.create_geom_groups_gui(Handle())
    scene.mj_data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, scene.mj_data)
    scene.show_contact_points = True
    scene.show_contact_forces = True
    scene.update_from_mjdata(scene.mj_data)
    scene.debug_visualization_enabled = True
    scene.add_arrow(np.zeros(3), np.array([1.0, 0.0, 0.0]), (255, 0, 0, 255), 0.01)
    scene._sync_arrows()
    scene.clear_debug_all()
