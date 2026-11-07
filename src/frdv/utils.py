"""Quaternion-convention helpers for the free joint (MuJoCo wxyz <-> Pinocchio xyzw)."""

import numpy as np


def mj_to_pin(q):
    """MuJoCo qpos (..., nq) -> Pinocchio q (root quaternion xyzw)."""
    q = np.array(q, dtype=float, copy=True)
    q[..., 3:7] = q[..., [4, 5, 6, 3]]
    return q


def pin_to_mj(q):
    """Pinocchio q (..., nq) -> MuJoCo qpos (root quaternion wxyz)."""
    q = np.array(q, dtype=float, copy=True)
    q[..., 3:7] = q[..., [6, 3, 4, 5]]
    return q


def continuous_quaternions(qpos, start=3):
    """Flip root quaternions (in place) so consecutive frames lie in the same hemisphere.

    q and -q are the same rotation, but a sign flip between frames looks like a jump to anything
    that differentiates raw qpos (velocities, controllers). Returns the number of flips fixed."""
    quat = qpos[:, start:start + 4]
    neg = np.sum(quat[1:] * quat[:-1], axis=1) < 0
    if not neg.any():
        return 0
    sign = np.cumprod(np.where(np.r_[False, neg], -1.0, 1.0))
    quat *= sign[:, None]
    return int(neg.sum())
