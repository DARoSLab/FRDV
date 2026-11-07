"""Small array checks at clip boundaries, outside the per-frame solver loops."""

import numpy as np


class ValidationError(ValueError):
    """Unusable motion/solver data; stop instead of writing an invalid trajectory."""


def targets_array(targets, points):
    try:
        array = np.asarray(targets, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValidationError("Targets must be numeric XYZ coordinates") from error
    if array.ndim != 3 or array.shape[1:] != (points, 3) or len(array) == 0:
        raise ValidationError(f"Targets must have shape (frames, {points}, 3) with at least one frame; "
                              f"got {array.shape}. Check the skeleton and keypoint mapping.")
    if not np.isfinite(array).all():
        raise ValidationError("Targets contain NaN or infinity; check the source motion, units and scaling")
    return np.ascontiguousarray(array)


def pose_array(pose, nq):
    array = np.asarray(pose, dtype=float)
    if array.shape != (nq,) or not np.isfinite(array).all():
        raise ValidationError(f"Starting pose must contain {nq} finite values; got shape {array.shape}")
    _quaternions(array[None])
    return np.ascontiguousarray(array)


def trajectory_array(qpos, frames, nq=None):
    array = np.asarray(qpos, dtype=float)
    if frames < 1 or array.ndim != 2 or array.shape[0] != frames or array.shape[1] < 7 or (nq is not None and array.shape[1] != nq):
        raise ValidationError(f"Output trajectory must have shape ({frames}, {nq or 'nq >= 7'}) "
                              f"with at least one frame; got {array.shape}")
    if not np.isfinite(array).all():
        raise ValidationError("Output trajectory contains NaN or infinity; no motion will be saved")
    _quaternions(array)
    return array


def _quaternions(array):
    quat = array[:, 3:7]
    norm2 = np.einsum('ij,ij->i', quat, quat)
    if np.any(np.abs(norm2 - 1.0) > 2e-4):
        raise ValidationError("Root quaternion is not normalized; check the pose/quaternion convention")


def qp_succeeded(failures):
    if failures:
        raise ValidationError(f"QP solver failed {failures} time(s); check the mapping, calibration and constraints")
