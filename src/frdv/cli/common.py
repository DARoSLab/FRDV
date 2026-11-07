"""Glue between the Hydra/OmegaConf configs and the solver (shared by all CLI tools)."""

from __future__ import annotations

import glob as _glob
import os
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from ..retarget import IKConfig
from ..robot_model import load_model
from ..sqp import SQPOptions

PACKAGE_DIR = Path(__file__).resolve().parents[1]
# Repository root when frdv is used from a checkout (robots/ lives there); else the package dir.
REPO_ROOT = PACKAGE_DIR.parents[1] if (PACKAGE_DIR.parents[1] / "robots").is_dir() else PACKAGE_DIR

try:
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("frdv_root"):
        OmegaConf.register_new_resolver("frdv_root", lambda: str(REPO_ROOT))
    if not OmegaConf.has_resolver("menagerie"):
        # ${menagerie:unitree_g1} -> local path of the MuJoCo Menagerie model, downloaded on first use
        def _menagerie(robot, model=None):
            from ..assets import menagerie
            return menagerie(robot, model)
        OmegaConf.register_new_resolver("menagerie", _menagerie)
except ImportError:  # pragma: no cover - only the CLI needs omegaconf
    OmegaConf = None

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def to_dict(cfg):
    """Plain, resolved dict (picklable for worker processes)."""
    if OmegaConf is not None and OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def resolve_path(p: str) -> str:
    p = os.path.expanduser(str(p))
    if os.path.isabs(p) or os.path.exists(p):
        return os.path.abspath(p)
    cand = REPO_ROOT / p
    return str(cand) if cand.exists() else os.path.abspath(p)


def find_files(data: str, pattern: str = "*") -> list[str]:
    data = resolve_path(data)
    if os.path.isfile(data):
        return [data]
    if not os.path.isdir(data):
        raise FileNotFoundError(f"data path not found: {data}")
    files = sorted(_glob.glob(os.path.join(data, pattern), recursive=True))
    return [f for f in files if os.path.isfile(f)]


def validate_subject_input(cfg: dict, files):
    """A selected AMASS profile is for one fitted body shape, regardless of collection."""
    if cfg["dataset"]["format"] == "amass" and (cfg.get("subject") or {}).get("name"):
        from ..motion.amass import validate_subject_files
        validate_subject_files(files)


def target_offset(robot: dict, dataset: dict):
    """Dataset alignments can replace an offset already included in a V1 profile."""
    return dataset.get("robot_target_offset", robot.get("target_offset") or (0, 0, 0))


def nominal_qpos(model: mujoco.MjModel, key: str | None) -> np.ndarray:
    if key:
        k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key)
        if k < 0:
            raise KeyError(f"robot.nominal_key '{key}' is not a keyframe in the model")
        return model.key_qpos[k].copy()
    return model.qpos0.copy()


def robot_nominal(model: mujoco.MjModel, robot: dict) -> np.ndarray:
    """Nominal pose: ``nominal_key`` keyframe (or qpos0), then ``nominal_joints: {joint: value}``."""
    q = nominal_qpos(model, robot.get("nominal_key"))
    for name, v in (robot.get("nominal_joints") or {}).items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise KeyError(f"robot.nominal_joints: no joint named '{name}'")
        q[model.jnt_qposadr[j]] = float(v)
    return q


def joint_limits(robot: dict) -> dict:
    """``joint_limits: {joint: [lo, hi]}`` overrides (e.g. tighter than the model's)."""
    return {k: (float(v[0]), float(v[1])) for k, v in (robot.get("joint_limits") or {}).items()}


def model_height(model: mujoco.MjModel, qpos: np.ndarray) -> float:
    """Standing height: top of the highest geom minus bottom of the lowest, at ``qpos``."""
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(model, d)
    g = np.flatnonzero(model.geom_bodyid > 0)               # skip world geoms (floor)
    R = d.geom_xmat[g].reshape(-1, 3, 3)
    center = model.geom_aabb[g, :3]
    half = model.geom_aabb[g, 3:]
    cz = d.geom_xpos[g, 2] + np.einsum("gj,gj->g", R[:, 2, :], center)
    hz = np.einsum("gj,gj->g", np.abs(R[:, 2, :]), half)     # world z half-extent of each local AABB
    return float((cz + hz).max() - (cz - hz).min())


def robot_keypoints(robot: dict, dataset: dict) -> dict:
    """canonical -> {link, weight}, with dataset-specific weight overrides applied."""
    kps = {k: dict(v) for k, v in (robot.get("keypoints") or {}).items()}
    for k, w in (dataset.get("weights") or {}).items():
        if k in kps:
            kps[k]["weight"] = float(w)
    return kps


def robot_height(robot: dict) -> float:
    if robot.get("height"):
        return float(robot["height"])
    model = load_model(resolve_path(robot["xml"]))
    return model_height(model, robot_nominal(model, robot))


def robot_points(robot: dict) -> dict:
    """{canonical keypoint: world position} at the robot's nominal pose (for limb-wise scaling)."""
    import json
    kps = {k: v for k, v in (robot.get("keypoints") or {}).items() if not v.get("calibrate")}
    nominal = json.dumps({"nominal_key": robot.get("nominal_key"), "nominal_joints": robot.get("nominal_joints")})
    return _robot_points(resolve_path(robot["xml"]), nominal, json.dumps(
        {k: [v["link"], list(v.get("offset") or (0.0, 0.0, 0.0))] for k, v in kps.items()}, sort_keys=True))


@lru_cache(maxsize=16)
def _robot_points(xml: str, nominal: str, spec: str) -> dict:
    import json
    from ..kinematics import Kinematics
    kin = Kinematics(xml)
    q = robot_nominal(kin.model, json.loads(nominal))
    kin.forward(q, with_jac=False)
    items = json.loads(spec)
    pts = kin.point_set([(link, off) for link, off in items.values()])
    pos = kin.positions(pts)
    return {k: pos[i].tolist() for i, k in enumerate(items)}


def calibrated_keypoints(robot: dict) -> list[str]:
    return [k for k, v in (robot.get("keypoints") or {}).items() if v.get("calibrate")]


def calibration_paths(robot: dict, dataset: dict, profile=None) -> list[Path]:
    """User and package paths within this robot's calibration directory."""
    from .profiles import _component
    robot_name, dataset_name = _component(robot['name']), _component(dataset.get('name', 'unknown'))
    relative = Path(profile) if profile else Path(robot_name, dataset_name, 'base.yaml')
    if relative.is_absolute() or '..' in relative.parts or relative.parts[:2] != (robot_name, dataset_name):
        raise ValueError('Calibration destination must belong to the selected robot and dataset')
    user = Path(os.environ.get("FRDV_CONFIG_DIR", "frdv_conf")).resolve()
    return [user / 'calibration' / relative, PACKAGE_DIR / 'conf' / 'calibration' / relative]


def load_calibration(robot: dict, dataset: dict, calibration=None) -> tuple[dict | None, Path | None]:
    """(offsets {keypoint: [x, y, z]}, path) or (None, None) when no calibration exists."""
    if calibration is None:
        from .profiles import load_profile
        calibration = load_profile(robot['name'], dataset.get('name', 'unknown')).calibration
    for key, expected in (('robot', robot['name']), ('dataset', dataset.get('name', 'unknown'))):
        if calibration.get(key, expected) != expected:
            raise ValueError(f'Marker calibration belongs to another {key}')
    data = calibration.get('markers') or {}
    if not data.get('offsets'):
        return None, None
    for key, expected in (('robot', robot['name']), ('dataset', dataset.get('name', 'unknown'))):
        if data.get(key, expected) != expected:
            raise ValueError(f'Marker offsets belong to another {key}')
    paths = calibration.get('sources') or []
    return {k: list(map(float, v)) for k, v in data['offsets'].items()}, Path(paths[-1]) if paths else None


def ik_config(robot: dict, solver: dict, keypoints: list[str], dataset: dict | None = None,
              offsets: dict | None = None, calibration=None) -> IKConfig:
    """Build the solver config for the tracked ``keypoints`` (canonical names, in target order).

    Keypoints with ``calibrate: true`` take their offset from ``offsets`` if given, else from the
    robot x dataset calibration file, else from their own ``offset`` (default: the link origin).
    """
    xml = resolve_path(robot["xml"])
    model = load_model(xml)
    kps = robot_keypoints(robot, dataset or {})
    calib = offsets if offsets is not None else (load_calibration(robot, dataset or {}, calibration)[0] or {})
    for k in keypoints:
        if kps[k].get("calibrate") and k in calib:
            kps[k]["offset"] = calib[k]
    reg = np.full(model.nq - 7, float(robot.get("joint_reg_weight", 1.0)))
    for name, w in (robot.get("joint_reg_weights") or {}).items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise KeyError(f"robot.joint_reg_weights: no joint named '{name}'")
        reg[model.jnt_qposadr[j] - 7] = float(w)
    sc = robot.get("self_collision") or {}
    return IKConfig(
        xml_path=xml,
        targets=[(kps[k]["link"], list(kps[k]["offset"])) if kps[k].get("offset") is not None else kps[k]["link"]
                 for k in keypoints],
        target_weights=[float(kps[k].get("weight", 1000.0)) for k in keypoints],
        foot_points=foot_points(robot) + auto_ground_points(robot, model),
        q_nominal=robot_nominal(model, robot),
        joint_limits=joint_limits(robot),
        joint_reg_weights=reg,
        smoothing_weight=float(solver.get("smoothing_weight", 10.0)),
        accel_weight=float(solver.get("accel_weight", 0.0)),
        rot_smoothing_scale=float(solver.get("rot_smoothing_scale", 0.25)),
        capsule_ground=bool(robot.get("capsule_ground", True)),
        self_collision_pairs=[tuple(p) for p in (sc.get("pairs") or [])],
        self_collision_margin=float(sc.get("margin", 0.05)),
        constraint_activation=solver.get("constraint_activation", 0.10),
        use_jit=solver.get("jit", None),
        sqp=SQPOptions(max_iter=int(solver.get("max_iter", 1)), line_search=bool(solver.get("line_search", True)),
                       mu_init=float(solver.get("mu_init", 10.0))),
        warmup_iters=int(solver.get("warmup_iters", 200)),
        boost_frames=int(solver.get("boost_frames", 20)),
        boost_iters=int(solver.get("boost_iters", 100)),
    )


def foot_points(robot: dict) -> list:
    """foot_points entries: a body/site name, or {link, offset} (offset in the link frame)."""
    out = []
    for f in robot.get("foot_points") or []:
        if isinstance(f, dict):
            out.append((f["link"], list(map(float, f.get("offset") or (0.0, 0.0, 0.0)))))
        else:
            out.append(f)
    return out


def auto_ground_points(robot: dict, model) -> list:
    """``ground_points: auto`` -> extreme surface points of every body (kept above the floor)."""
    gp = robot.get("ground_points", None)
    if gp != "auto":
        return []
    from ..robot_model import surface_points
    feet = {f["link"] if isinstance(f, dict) else f for f in robot.get("foot_points") or []}
    return [(b, list(o)) for b, o in surface_points(model, exclude_bodies=feet)]


def validate_robot(robot: dict) -> list[str]:
    """Human-readable problems with a robot config (missing links, keyframe, ...)."""
    problems = []
    xml = resolve_path(robot["xml"])
    if not os.path.exists(xml):
        return [f"robot.xml not found: {xml}"]
    model = load_model(xml)
    if model.njnt == 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        problems.append("the first joint must be a free joint (floating base)")

    def exists(n):
        return (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0
                or mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) >= 0)

    for k, v in (robot.get("keypoints") or {}).items():
        if v.get("offset") is not None and len(v["offset"]) != 3:
            problems.append(f"keypoint '{k}': offset must be [x, y, z]")
        if not exists(v["link"]):
            problems.append(f"keypoint '{k}': no body or site named '{v['link']}'")
    for f in robot.get("foot_points") or []:
        f = f["link"] if isinstance(f, dict) else f
        if not exists(f):
            problems.append(f"foot point '{f}': no body or site with that name")
    key = robot.get("nominal_key")
    if key and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key) < 0:
        problems.append(f"nominal_key '{key}' is not a keyframe")
    return problems


def per_clip_setup(retargeter, solver: dict, fps: float):
    """Settings that depend on the clip's frame rate (velocity limits are given in rad/s)."""
    v = solver.get("max_joint_velocity")
    retargeter.cfg.max_joint_step = None if v is None else float(v) / float(fps)


def postprocess(retargeter, solver: dict, targets, q, fps: float):
    """Optional non-causal clean-up; returns a short report string (or None)."""
    ds = solver.get("despike") or {}
    if not ds.get("enabled", False):
        return None
    from ..smooth import despike
    rep = despike(retargeter, targets, q, threshold=float(ds.get("threshold", 7.5)) / float(fps),
                  ratio=float(ds.get("ratio", 4.0)), half_window=int(ds.get("half_window", 5)),
                  sigma=float(ds.get("sigma", 1.5)), pull_weight=float(ds.get("pull_weight", 200.0)),
                  passes=int(ds.get("passes", 3)))
    return str(rep)
