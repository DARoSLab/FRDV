"""Original AMASS SMPL+H NPZ motions, with V1 validation and compact NumPy FK.

Requires poses (T, 156), trans (T, 3), betas (16,), gender, and a positive frame rate.
Body models are <smplh_model_path>/<gender>/model.npz with 16 shape coefficients.
Only the 22 body joints are reconstructed; finger poses and DMPL mesh deformations are ignored.
"""

from __future__ import annotations

import os

import numpy as np

from ._smplh import _load_smplh_model_data
from .amass_joints import BODY_JOINT_NAMES, NUM_BODY_JOINTS
from .base import Motion


def _identity(gender, betas):
    if gender.size != 1:
        raise ValueError("AMASS gender must be a single male, female, or neutral value")
    value = gender.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value).strip().lower()
    if value not in {"male", "female", "neutral"}:
        raise ValueError(f"Unsupported AMASS gender: {value!r}")
    if betas.shape != (16,):
        raise ValueError(f"AMASS betas must have shape (16,); got {betas.shape}")
    if betas.dtype.kind not in "fiu" or not np.isfinite(betas).all():
        raise ValueError("AMASS betas must contain finite numeric values")
    return value, np.asarray(betas, dtype=np.float64)


def read_motion(path):
    """Read and validate V1's input contract; never invent missing motion metadata."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"poses", "trans", "betas", "gender"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"AMASS motion {path} is missing: {sorted(missing)}")
        data = {k: archive[k] for k in required}
        rates = {k: archive[k] for k in ("mocap_framerate", "mocap_frame_rate") if k in archive}
    poses, trans = data["poses"], data["trans"]
    if poses.ndim != 2 or poses.shape[1] != 156 or len(poses) == 0:
        raise ValueError(f"AMASS poses must have shape (frames, 156), with at least one frame; got {poses.shape}")
    if trans.shape != (len(poses), 3):
        raise ValueError(f"AMASS trans must have shape ({len(poses)}, 3); got {trans.shape}")
    for key, value in (("poses", poses), ("trans", trans)):
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"AMASS {key} must contain finite numeric values")
    gender, betas = _identity(data["gender"], data["betas"])
    if not rates:
        raise ValueError("AMASS motion requires mocap_framerate or mocap_frame_rate")
    for key, value in rates.items():
        if value.size != 1 or value.dtype.kind not in "fiu":
            raise ValueError(f"AMASS {key} must be a positive finite frame rate")
        rates[key] = float(value.item())
        if not np.isfinite(rates[key]) or rates[key] <= 0:
            raise ValueError(f"AMASS {key} must be a positive finite frame rate")
    if len(rates) == 2 and not np.isclose(*rates.values()):
        raise ValueError("AMASS frame-rate metadata fields disagree")
    return np.asarray(poses, dtype=np.float64), np.asarray(trans, dtype=np.float64), betas, gender, next(iter(rates.values()))


def validate_subject_files(paths):
    """A selected body-profile requires the same gender and fitted shape in every clip."""
    identity = None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if not {"gender", "betas"} <= set(archive.files):
                raise ValueError(f"AMASS motion {path} is missing subject gender/betas")
            current = _identity(archive["gender"], archive["betas"])
        if identity is None:
            identity = current
        elif current[0] != identity[0] or not np.array_equal(current[1], identity[1]):
            raise ValueError(f"Multiple body-shape fits in selected motions ({path}); use separate subject profiles/runs")
    return identity


def _rodrigues(rotvec):
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = rotvec / np.maximum(angle, 1e-12)
    c, s = np.cos(angle)[..., None], np.sin(angle)[..., None]
    K = np.zeros(rotvec.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -axis[..., 2], axis[..., 1]
    K[..., 1, 0], K[..., 1, 2] = axis[..., 2], -axis[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -axis[..., 1], axis[..., 0]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def skeleton_fk(poses, trans, betas, model):
    """Vectorized over time; fixed subject shape once, rotations and positions only.

    Avoids the V1 T-by-52 shaped-joint copies and T-by-22 homogeneous 4x4 matrices.
    """
    J = model["J_template"] + np.einsum("jcb,b->jc", model["J_shapedirs"], betas)
    T = len(poses)
    rotations = _rodrigues(poses[:, :66].reshape(T, NUM_BODY_JOINTS, 3))
    pos = np.empty((T, NUM_BODY_JOINTS, 3))
    world = np.empty((T, NUM_BODY_JOINTS, 3, 3))
    world[:, 0], pos[:, 0] = rotations[:, 0], J[0]
    for j in range(1, NUM_BODY_JOINTS):
        parent = model["parents"][j]
        world[:, j] = world[:, parent] @ rotations[:, j]
        pos[:, j] = pos[:, parent] + np.einsum("tij,j->ti", world[:, parent], J[j] - J[parent])
    return pos + trans[:, None], J


def load(path: str, smplh_model_path: str | None = None, **_) -> Motion:
    model_path = smplh_model_path or os.environ.get("SMPLH_MODEL_PATH")
    if not model_path:
        raise ValueError("AMASS needs the original 16-beta SMPL+H models: set dataset.smplh_model_path "
                         "or SMPLH_MODEL_PATH to the folder containing <gender>/model.npz")
    poses, trans, betas, gender, fps = read_motion(path)
    model = _load_smplh_model_data(model_path, gender)
    positions, J = skeleton_fk(poses, trans, betas, model)
    # The zero-angle SMPL+H body is Y-up; recorded AMASS positions are already world Z-up.
    rest = (J - J[0]) @ np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]).T
    return Motion(names=list(BODY_JOINT_NAMES), positions=positions, fps=fps,
                  rest=rest, parents=model["parents"].tolist())
