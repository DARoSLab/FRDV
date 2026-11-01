from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mocap_retargeting.data_loading.amass.loader import AMASSLoader
from mocap_retargeting.data_loading.lafan1.loader import Lafan1Loader
from mocap_retargeting.data_loading.dataset_selection import (
    load_mocap,
    task_config_filename,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "assets/mocap/sample/walk3_subject5.bvh"


def test_lafan_loads_and_preprocesses_bvh():
    loader = load_mocap(SAMPLE, {"data_type": "lafan1"})
    assert isinstance(loader, Lafan1Loader)
    loader.preprocess(scale=0.01)
    loader.process_all_frames_vectorized()
    frames = loader.get_frames(0, 3)
    assert frames.shape == (3, 27, 3)
    assert np.isfinite(frames).all()
    assert {"LeftHand_EndSite", "RightHand_EndSite"} <= set(loader.joint_names_ordered)
    motion = loader.to_motion_data()
    assert motion.positions is loader.motion_tensor
    assert motion.joint_names == loader.joint_names_ordered
    assert motion.skeleton == loader.get_skeleton_structure()
    assert motion.fps == pytest.approx(1 / loader.frame_time)


def test_lafan_targets_match_original_mocap_and_robot_frames():
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["task=unitree_g1/lafan"])
    loader = load_mocap(SAMPLE, cfg)
    model = mujoco.MjModel.from_xml_path(str(ROOT / cfg.mujoco_dst_xml_path))
    for item in cfg.mapping:
        assert item.bvh in loader.joints
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.robot) >= 0
        assert (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"mocap_{item.bvh}") >= 0
        )


@pytest.fixture
def amass_loader(tmp_path, monkeypatch):
    # A tiny synthetic body model exercises real FK without licensed model files.
    model_dir = tmp_path / "smplh/neutral"
    model_dir.mkdir(parents=True)
    joints = np.zeros((52, 3), dtype=np.float32)
    joints[7] = [-0.1, -0.8, 0.0]
    joints[8] = [0.1, -0.8, 0.0]
    joints[10] = [-0.1, 0.0, 0.0]
    joints[11] = [0.1, 0.0, 0.0]
    joints[15] = [0.0, 1.65, 0.0]
    joints[16] = [-0.2, 1.4, 0.0]
    joints[17] = [0.2, 1.4, 0.0]
    parents = np.array(
        [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        + [20] * 15
        + [21] * 15,
        dtype=np.int32,
    )
    np.savez(
        model_dir / "model.npz",
        J_regressor=np.eye(52, dtype=np.float32),
        v_template=joints,
        shapedirs=np.zeros((52, 3, 16), dtype=np.float32),
        kintree_table=np.stack([parents, np.arange(52)]),
    )
    motion_path = tmp_path / "motion.npz"
    trans = np.array([[0.0, 0.0, 1.0], [0.25, 0.0, 1.2]], dtype=np.float32)
    np.savez(
        motion_path,
        poses=np.zeros((2, 156), dtype=np.float32),
        trans=trans,
        betas=np.zeros(16, dtype=np.float32),
        gender="neutral",
        mocap_framerate=60.0,
    )
    monkeypatch.setenv("SMPLH_MODEL_PATH", str(model_dir.parent))
    return load_mocap(motion_path, {"data_type": " AMASS "})


def test_amass_loads_without_removed_dataset_dependencies(amass_loader):
    loader = amass_loader
    assert isinstance(loader, AMASSLoader)
    assert loader.get_frames().shape == (2, 22, 3)
    np.testing.assert_allclose(
        loader.get_frames()[:, 0], [[0.0, 0.0, 1.0], [0.25, 0.0, 1.2]]
    )
    assert loader.frame_time == pytest.approx(1 / 60)
    motion = loader.to_motion_data()
    assert motion.positions is loader.motion_tensor
    assert motion.fps == pytest.approx(60)


def test_subject_calibration_recovers_known_scale_and_offsets(amass_loader):
    from tools.model_preparation.calibrate_amass_subjects import fit_subject_transform

    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["task=unitree_g1/amass"])
    loader = amass_loader
    raw_motion = loader.motion_tensor_raw.copy()
    # Independent XYZ variation makes every calibration control identifiable.
    raw = np.random.default_rng(42).normal(size=(22, 3)) * [0.3, 0.1, 0.4]
    raw[7:9, 2] = [-0.8, -0.79]
    loader._tpose_raw = raw[None].copy()
    scale = np.array([0.72, 0.85, 0.81])
    offset = np.array([0.035, -0.01, 0.02])
    upper = np.array([-0.02, 0.005, 0.04])
    rotation = np.array(cfg.mocap.rotation).reshape(3, 3)
    expected = (raw * scale) @ rotation.T
    expected[:, 2] -= expected[[7, 8], 2].min() - 0.05 * scale[2]
    from mocap_retargeting.data_loading.amass.joint_definitions import (
        UPPER_BODY_JOINT_NAMES,
    )

    expected[
        [loader.joint_names_ordered.index(name) for name in UPPER_BODY_JOINT_NAMES]
    ] += upper
    expected -= np.r_[expected[0, :2], 0]
    expected += offset
    targets = np.array(
        [expected[loader.joint_names_ordered.index(item.bvh)] for item in cfg.mapping]
    )
    fitted, before, after = fit_subject_transform(cfg, loader, targets)
    np.testing.assert_allclose(fitted.scale, scale, atol=1e-6)
    np.testing.assert_allclose(fitted.offset, offset, atol=1e-6)
    np.testing.assert_allclose(fitted.upper_body_offset, upper, atol=1e-6)
    np.testing.assert_array_equal(fitted.rotation, rotation)
    np.testing.assert_array_equal(loader.motion_tensor_raw, raw_motion)
    assert after < 1e-7 and before > after


def test_subject_calibration_rejects_unidentifiable_controls(amass_loader):
    from tools.model_preparation.calibrate_amass_subjects import fit_subject_transform

    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["task=unitree_g1/amass"])
    amass_loader._tpose_raw[:, :, 1] = 0
    targets = np.zeros((len(cfg.mapping), 3))
    with pytest.raises(ValueError, match="cannot determine"):
        fit_subject_transform(cfg, amass_loader, targets)


def test_subject_calibration_rejects_mixed_shapes_in_one_folder(tmp_path):
    from tools.model_preparation.calibrate_amass_subjects import validate_subject_files

    paths = [tmp_path / "a.npz", tmp_path / "b.npz"]
    for path in paths:
        np.savez(path, gender="female", betas=np.zeros(16))
    validate_subject_files(paths)
    np.savez(paths[1], gender="female", betas=np.ones(16))
    with pytest.raises(ValueError, match="Multiple body-shape fits"):
        validate_subject_files(paths)


def test_amass_default_preprocessing_preserves_height_after_grounded_preview(
    amass_loader,
):
    loader = amass_loader
    source = loader.get_frames().copy()
    loader.preprocess(ground=True)
    loader.preprocess()
    np.testing.assert_allclose(loader.get_frames(), source)
    assert loader.get_tpose_positions()["L_Ankle"][2] == pytest.approx(0.05)
    np.testing.assert_allclose(loader.get_frames(), source)


def test_amass_standing_reference_uses_scene_floor_after_tilt_without_shifting_motion(
    amass_loader,
):
    from scipy.spatial.transform import Rotation

    loader = amass_loader
    rotation = Rotation.from_euler("xyz", [10, 5, 25], degrees=True).as_matrix()
    offset = np.array([0.12, -0.03, 0.08])
    loader.preprocess(
        scale=[0.9, 0.8, 0.7],
        rotation_matrix=rotation,
        manual_offset=offset,
        ground=False,
        ground_clearance=0.025,
    )
    recorded = loader.get_frames().copy()
    reference = loader.get_tpose_positions()
    assert min(reference[name][2] for name in ("L_Ankle", "R_Ankle")) == pytest.approx(
        0.025 * 0.7 + offset[2]
    )
    np.testing.assert_allclose(reference["Pelvis"][:2], offset[:2])
    np.testing.assert_array_equal(loader.get_frames(), recorded)


def rewrite_npz(path, *, remove=(), **updates):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files if key not in remove}
    values.update(updates)
    np.savez(path, **values)


@pytest.mark.parametrize(
    "field", ["poses", "trans", "betas", "gender", "mocap_framerate"]
)
def test_amass_rejects_missing_metadata_instead_of_inventing_defaults(
    amass_loader, field
):
    path = amass_loader.file_path
    rewrite_npz(path, remove=[field])
    with pytest.raises(ValueError, match="missing|requires"):
        AMASSLoader(path)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("poses", np.zeros((2, 165)), "poses"),
        ("poses", np.zeros((0, 156)), "poses"),
        ("trans", np.zeros((1, 3)), "trans"),
        ("betas", np.zeros(10), "betas"),
        ("gender", "unknown", "gender"),
        ("gender", np.array(["male", "female"]), "gender"),
        ("mocap_framerate", 0, "frame rate"),
        ("mocap_framerate", -60, "frame rate"),
        ("mocap_framerate", np.nan, "frame rate"),
        ("mocap_framerate", np.array([30, 60]), "frame rate"),
        ("mocap_frame_rate", 30, "disagree"),
    ],
)
def test_amass_rejects_incompatible_metadata(amass_loader, field, value, message):
    rewrite_npz(amass_loader.file_path, **{field: value})
    with pytest.raises(ValueError, match=message):
        AMASSLoader(amass_loader.file_path)


@pytest.mark.parametrize("field", ["poses", "trans", "betas"])
def test_amass_rejects_nonfinite_motion_data(amass_loader, field):
    with np.load(amass_loader.file_path) as archive:
        value = archive[field].copy()
    value.flat[0] = np.inf
    rewrite_npz(amass_loader.file_path, **{field: value})
    with pytest.raises(ValueError, match=f"{field}.*finite"):
        AMASSLoader(amass_loader.file_path)


def test_amass_accepts_alternate_fps_metadata_and_preserves_fractional_rate(
    amass_loader,
):
    rewrite_npz(
        amass_loader.file_path, remove=["mocap_framerate"], mocap_frame_rate=59.94
    )
    loader = AMASSLoader(amass_loader.file_path)
    assert loader.to_motion_data().fps == pytest.approx(59.94)


def test_amass_uses_subject_shape_and_root_rotation_but_ignores_finger_poses(
    amass_loader,
):
    model_path = Path(amass_loader.smplh_model_path) / "neutral/model.npz"
    with np.load(model_path) as model:
        shapedirs = model["shapedirs"].copy()
    shapedirs[7, 1, 15] = 0.1
    rewrite_npz(model_path, shapedirs=shapedirs)
    poses = np.zeros((2, 156))
    poses[:, 0] = np.pi / 2
    poses[:, 66:] = 1.1
    betas = np.zeros(16)
    betas[15] = 2
    rewrite_npz(amass_loader.file_path, poses=poses, betas=betas)
    loader = AMASSLoader(amass_loader.file_path)
    # Canonical ankle [-.1, -.8, 0] becomes [-.1, -.6, 0] through beta 16,
    # then [-.1, 0, -.6] through the root rotation, plus each frame's trans.
    expected = np.array([[-0.1, 0, 0.4], [0.15, 0, 0.6]])
    np.testing.assert_allclose(loader.get_frames()[:, 7], expected, atol=1e-6)
    reference = loader.get_tpose_positions()
    assert reference["L_Ankle"][2] - reference["R_Ankle"][2] == pytest.approx(0.2)
    assert reference["R_Ankle"][2] == pytest.approx(loader.ground_clearance)
    poses[:, 66:] = 0
    rewrite_npz(amass_loader.file_path, poses=poses)
    np.testing.assert_array_equal(
        AMASSLoader(amass_loader.file_path).get_frames(), loader.get_frames()
    )


def test_amass_selects_the_requested_gender_model_without_neutral_fallback(
    amass_loader,
):
    root = Path(amass_loader.smplh_model_path)
    male = root / "male"
    male.mkdir()
    with np.load(root / "neutral/model.npz") as model:
        values = {key: model[key] for key in model.files}
    values["v_template"] = values["v_template"] + np.array([0.1, 0, 0])
    np.savez(male / "model.npz", **values)
    rewrite_npz(amass_loader.file_path, gender=b" MALE ")
    loader = AMASSLoader(amass_loader.file_path)
    np.testing.assert_allclose(loader.get_frames()[:, 0, 0], [0.1, 0.35])
    (male / "model.npz").unlink()
    with pytest.raises(FileNotFoundError, match="male/model.npz"):
        AMASSLoader(amass_loader.file_path)


@pytest.mark.parametrize("problem", ["shape_basis", "nonfinite", "hierarchy"])
def test_amass_rejects_incompatible_models(amass_loader, problem):
    path = Path(amass_loader.smplh_model_path) / "neutral/model.npz"
    if problem == "shape_basis":
        rewrite_npz(path, shapedirs=np.zeros((52, 3, 300)))
    elif problem == "nonfinite":
        rewrite_npz(path, v_template=np.full((52, 3), np.nan))
    else:
        with np.load(path) as model:
            tree = model["kintree_table"].copy()
        tree[0, 5] = 0
        rewrite_npz(path, kintree_table=tree)
    with pytest.raises(ValueError, match="SMPL\\+H model"):
        AMASSLoader(amass_loader.file_path)


@pytest.mark.parametrize("clearance", [0, 0.025, 0.05])
def test_amass_ground_clearance_matches_motion_and_calibration(amass_loader, clearance):
    from mocap_retargeting.mocap_calibration.calibration_app import (
        prepare_calibration_positions,
    )

    loader = amass_loader
    positions = prepare_calibration_positions(
        loader,
        [0.9, 0.8, 0.7],
        np.eye(3),
        0,
        0,
        0.1,
        ground=True,
        ground_clearance=clearance,
    )
    expected = clearance * 0.7 + 0.1
    assert loader.get_frames()[:, [7, 8], 2].min() == pytest.approx(expected)
    assert min(positions[name][2] for name in ("L_Ankle", "R_Ankle")) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("clearance", [-0.01, np.inf, np.nan])
def test_amass_rejects_invalid_ground_clearance(amass_loader, clearance):
    with pytest.raises(ValueError, match="ground_clearance"):
        amass_loader.preprocess(ground=True, ground_clearance=clearance)


def test_mixed_amass_fps_reaches_chunks_refinement_and_final_outputs(
    amass_loader, tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from mocap_retargeting.kinematic_retargeting import retarget_motion
    from mocap_retargeting.kinematic_retargeting.motion_chunking import (
        chunk_retargeting,
    )
    from mocap_retargeting.kinematic_retargeting.refinement import trajectory_refinement

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    with np.load(amass_loader.file_path) as archive:
        values = {key: archive[key] for key in archive.files}
    values["poses"] = np.tile(values["poses"], (2, 1))
    values["trans"] = np.tile(values["trans"], (2, 1))
    for name, fps in (("a", 59.94), ("b", 120)):
        np.savez(inputs / f"{name}.npz", **{**values, "mocap_framerate": fps})

    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=unitree_g1/amass",
                "n_motion_chunk_workers=2",
                "enable_refine=true",
                "save_chunks=true",
                "stitch_overlap_frames=0",
            ],
        )
    # A conflicting legacy override must not replace each source file's timing.
    OmegaConf.update(cfg, "mocap.fps", 24, force_add=True)
    cfg.mocap_src_path = str(inputs)
    monkeypatch.chdir(tmp_path)
    problem = SimpleNamespace(
        robot=SimpleNamespace(q0_robot=np.zeros(36), active_mapping={}),
        sqp_cfg=None,
        generate_solver_functions=lambda: (None, None),
    )
    monkeypatch.setattr(retarget_motion, "RetargetingProblem", lambda _: problem)
    context = chunk_retargeting.ChunkWorkerContext(None, cfg, np.zeros(36), {}, None)
    monkeypatch.setattr(chunk_retargeting, "_worker_context", context)

    class Retargeter:
        def __init__(self, cfg, solver, mocap, *args, **kwargs):
            self.mocap = mocap

        def retarget(self, names):
            q = np.zeros((len(self.mocap), 36))
            q[:, 6] = 1
            return q.tolist(), self.mocap.tolist()

    monkeypatch.setattr(chunk_retargeting, "RetargetingSolver", Retargeter)

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def starmap(self, function, tasks):
            return [function(*task) for task in tasks]

    monkeypatch.setattr(retarget_motion.multiprocessing, "Pool", Pool)
    rates_seen = []
    detect = trajectory_refinement.detect_spikes

    def record_rate(q, fps, threshold):
        rates_seen.append(fps)
        return detect(q, fps, threshold)

    monkeypatch.setattr(trajectory_refinement, "detect_spikes", record_rate)
    retarget_motion.run_retargeting(cfg)
    assert rates_seen == pytest.approx([59.94, 120])
    outputs = list((tmp_path / "log/kin_trajectories/unitree_g1").glob("*.npy"))
    assert len(outputs) == 6  # Two chunks and one final output per motion.
    for path in outputs:
        saved = np.load(path, allow_pickle=True).item()
        assert saved["fps"] == pytest.approx(59.94 if "_a_" in path.name else 120)
        assert len(saved["q"]) == (4 if "_0_4_" in path.name else 2)
    assert cfg.mocap.fps == 24


@pytest.mark.parametrize("ground", [None, False, True])
def test_amass_task_grounding_applies_to_motion_and_calibration(amass_loader, ground):
    from mocap_retargeting.kinematic_retargeting.retarget_motion import prepare_motion
    from mocap_retargeting.mocap_calibration.calibration_app import process_mocap_sites

    overrides = ["task=unitree_g1/amass"]
    if ground is not None:
        overrides.append(f"mocap.ground={str(ground).lower()}")
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    assert cfg.mocap.ground is (ground is True)
    loader = prepare_motion(cfg, amass_loader.file_path)
    ankle_index = loader.joint_names_ordered.index("L_Ankle")
    ankle_heights = loader.get_frames()[:, ankle_index, 2]
    scale_z, offset_z = cfg.mocap.scale[2], cfg.mocap.offset[2]
    if ground:
        assert ankle_heights.min() == pytest.approx(0.05 * scale_z + offset_z)
    else:
        np.testing.assert_allclose(
            ankle_heights, np.array([1.0, 1.2]) * scale_z + offset_z
        )

    positions, _ = process_mocap_sites(
        cfg,
        None,
        None,
        loader,
        np.array(cfg.mocap.scale),
        np.array(cfg.mocap.rotation).reshape(3, 3),
        "unused.yaml",
        x_offset=cfg.mocap.offset[0],
        y_offset=cfg.mocap.offset[1],
        z_offset=offset_z,
        save=False,
        compute_offsets=False,
    )
    # The synthetic standing reference is grounded even when the recorded motion isn't.
    expected_height = cfg.mocap.ground_clearance * scale_z + offset_z
    assert positions["L_Ankle"][2] == pytest.approx(expected_height)


@pytest.mark.parametrize("subject_specific", [False, True])
def test_amass_upper_body_offset_matches_retargeting_preview_and_site_export(
    amass_loader, tmp_path, monkeypatch, subject_specific
):
    import yaml
    from mocap_retargeting.data_loading import mocap_transform
    from mocap_retargeting.kinematic_retargeting.retarget_motion import prepare_motion
    from mocap_retargeting.mocap_calibration.calibration_app import process_mocap_sites

    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["task=unitree_g1/amass"])
    profile = tmp_path / "subject.yaml"
    if subject_specific:
        profile.write_text(
            "# @package _global_\ndefaults:\n  - /task/unitree_g1/amass\n  - _self_\n"
        )
    cfg.mujoco_src_xml_path = str(ROOT / cfg.mujoco_src_xml_path)
    cfg.mujoco_dst_xml_path = str(tmp_path / "calibrated.xml")
    model = mujoco.MjModel.from_xml_path(cfg.mujoco_src_xml_path)
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    scale = np.array(cfg.mocap.scale)
    rotation = np.array(cfg.mocap.rotation).reshape(3, 3)
    offset = np.array(cfg.mocap.offset)

    def calibration(loader, *, export=False):
        return process_mocap_sites(
            cfg,
            model,
            data,
            loader,
            scale,
            rotation,
            str(profile if subject_specific else tmp_path / "amass.yaml"),
            x_offset=offset[0],
            y_offset=offset[1],
            z_offset=offset[2],
            save=export,
            compute_offsets=export,
            subject_transform=mocap_transform.resolve_mocap_transform(cfg),
        )[0]

    cfg.upper_body_offset = None
    loader = prepare_motion(cfg, amass_loader.file_path)
    unshifted_motion = loader.get_frames().copy()
    unshifted_preview = calibration(loader)
    adjustment = np.array([0.03, -0.02, 0.065])
    if subject_specific:
        # A selected subject task overrides the inherited AMASS value before loading.
        from omegaconf import OmegaConf

        cfg = OmegaConf.merge(cfg, {"upper_body_offset": adjustment.tolist()})
    else:
        cfg.upper_body_offset = adjustment.tolist()
    loader = prepare_motion(cfg, amass_loader.file_path)
    shifted_motion = loader.get_frames().copy()
    shifted_preview = calibration(loader)
    for name in ["Spine3", "Head", "L_Wrist", "Pelvis", "L_Hip", "L_Ankle"]:
        expected = adjustment if name in ["Spine3", "Head", "L_Wrist"] else np.zeros(3)
        idx = loader.joint_names_ordered.index(name)
        np.testing.assert_allclose(
            shifted_motion[:, idx] - unshifted_motion[:, idx],
            np.tile(expected, (loader.num_frames, 1)),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            shifted_preview[name] - unshifted_preview[name], expected, atol=1e-7
        )

    exported_positions = calibration(loader, export=True)
    if subject_specific:
        saved = yaml.safe_load(profile.read_text())
        assert saved["upper_body_offset"] == adjustment.tolist()
    for name in shifted_preview:
        np.testing.assert_array_equal(exported_positions[name], shifted_preview[name])
    tree = ET.parse(cfg.mujoco_dst_xml_path)
    for name, body_name in [
        ("L_Wrist", "left_wrist_yaw_link"),
        ("L_Hip", "left_hip_pitch_link"),
    ]:
        site = tree.find(f".//body[@name='{body_name}']/site[@name='mocap_{name}']")
        assert site is not None
        local = np.fromstring(site.get("pos"), sep=" ")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        world = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ local
        np.testing.assert_allclose(world, shifted_preview[name], atol=2e-6)


@pytest.mark.parametrize(
    "data_type", ["accad", "omomo", "soma", "optitrack", "unknown"]
)
def test_unavailable_source_is_rejected_before_opening_input(data_type):
    with pytest.raises(ValueError, match="Unsupported data_type"):
        load_mocap("does-not-exist", {"data_type": data_type})


@pytest.mark.parametrize(
    ("data_type", "filename"),
    [("lafan1", "lafan.yaml"), ("amass", "amass.yaml")],
)
def test_alignment_is_saved_to_its_own_dataset_config(data_type, filename):
    assert task_config_filename({"data_type": data_type}) == filename


def test_default_config_uses_g1_and_a_bundled_motion():
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config")
    assert cfg.robot == "unitree_g1"
    assert cfg.data_type == "lafan1"
    assert cfg.mocap.ground is True
    assert (ROOT / cfg.mocap_src_path).is_file()


def test_amass_model_path_uses_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPLH_MODEL_PATH", str(tmp_path))
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["task=unitree_g1/amass"])
    assert cfg.smplh_model_path == str(tmp_path)
    assert OmegaConf.to_container(cfg, resolve=True)["smplh_model_path"] == str(
        tmp_path
    )


@pytest.mark.parametrize("task", ["lafan", "amass"])
def test_retained_g1_models_load_without_other_robot_assets(task):
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"task=unitree_g1/{task}"])
    for xml_path in [cfg.mujoco_src_xml_path, cfg.mujoco_dst_xml_path]:
        model = mujoco.MjModel.from_xml_path(str(ROOT / xml_path))
        assert model.nq == 36  # 29 joints plus the floating base.


def test_batch_wrapper_invokes_a_registered_command(monkeypatch):
    import re
    from types import SimpleNamespace
    from tools import batch_retarget

    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout="Total Frames Retargeted: 3\n", stderr="")

    monkeypatch.setattr(batch_retarget.subprocess, "run", run)
    result = batch_retarget.retarget_file(
        batch_retarget.RetargetJob(str(SAMPLE), "unitree_g1/lafan")
    )
    project_metadata = (ROOT / "pyproject.toml").read_text()
    assert result[0]
    assert re.search(
        rf'^{re.escape(calls[0][2])}\s*=\s*"mocap_retargeting\.[^"]+:main"',
        project_metadata,
        re.MULTILINE,
    )
    task = next(
        arg.removeprefix("task=") for arg in calls[0] if arg.startswith("task=")
    )
    assert (ROOT / "conf/task" / (task + ".yaml")).is_file()
