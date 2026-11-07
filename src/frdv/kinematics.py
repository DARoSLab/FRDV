"""Minimal MuJoCo kinematics: batched point positions and Jacobians.

All Jacobians are w.r.t. the MuJoCo tangent space (``nv``): for the free joint
that is [world linear velocity, body-frame angular velocity], which matches
``mj_integratePos`` so steps can be applied without any quaternion
normalisation constraint.

Every quantity is produced by one ``mj_jacBody`` per *unique* body, then
shifted to the requested points with a vectorised cross product:

    J_point = Jp_body - [p - x_body]_x Jr_body
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .robot_model import load_model


@dataclass
class Capsule:
    body: int
    p1: np.ndarray  # endpoint in body frame
    p2: np.ndarray
    radius: float
    name: str


class PointSet:
    """A set of points rigidly attached to bodies (sites, body origins, offsets)."""

    def __init__(self, body_ids, local_pos):
        self.body_ids = np.asarray(body_ids, dtype=np.int64)
        self.local_pos = np.asarray(local_pos, dtype=np.float64).reshape(-1, 3)


class Kinematics:
    def __init__(self, xml_path: str | None = None, model: mujoco.MjModel | None = None):
        self.model = model if model is not None else load_model(xml_path)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.nq, self.nv = m.nq, m.nv
        if m.njnt == 0 or m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("frdv expects a floating-base model (first joint free).")
        if self.nq - 7 != self.nv - 6:
            raise ValueError("Only hinge/slide joints are supported after the free joint (this model has ball joints, e.g. a parallel-linkage ankle). Use the robot's serial-chain model instead, e.g. its URDF.")

        # Joint limits on q[7:] (= dofs 6:).
        lo = np.full(self.nq - 7, -np.inf)
        hi = np.full(self.nq - 7, np.inf)
        for j in range(1, m.njnt):
            if m.jnt_limited[j]:
                a = m.jnt_qposadr[j] - 7
                lo[a], hi[a] = m.jnt_range[j]
        self.q_min, self.q_max = lo, hi
        self._jacp = np.zeros((3, self.nv))
        self._jacr = np.zeros((3, self.nv))

    # ---------------------------------------------------------------- lookups
    def frame_point(self, name: str) -> tuple[int, np.ndarray]:
        """Resolve a name to (body id, local offset). Sites, then bodies."""
        m = self.model
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            # Site pos/quat are relative to the body frame.
            return int(m.site_bodyid[sid]), m.site_pos[sid].copy()
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return bid, np.zeros(3)
        raise KeyError(f"No site or body named '{name}' in model.")

    def point_set(self, names) -> PointSet:
        """Points from names, or ``(name, offset)`` pairs (offset in that body's/site's body frame)."""
        bodies, local = [], []
        for n in names:
            if isinstance(n, (tuple, list)):
                b, p = self.frame_point(n[0])
                p = p + np.asarray(n[1], dtype=float)
            else:
                b, p = self.frame_point(n)
            bodies.append(b)
            local.append(p)
        return PointSet(bodies, local)

    def capsules(self, bodies=None) -> list[Capsule]:
        m = self.model
        out = []
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                continue
            b = int(m.geom_bodyid[g])
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            if bodies is not None and bname not in bodies:
                continue
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            axis = R.reshape(3, 3)[:, 2] * m.geom_size[g][1]
            pos = m.geom_pos[g]
            out.append(Capsule(b, pos - axis, pos + axis, float(m.geom_size[g][0]), bname))
        return out

    # ------------------------------------------------------------- evaluation
    def forward(self, qpos, with_jac: bool = True):
        d = self.data
        d.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, d)
        if with_jac:
            mujoco.mj_comPos(self.model, d)

    def body_jacobians(self, body_ids):
        """(B,3,nv) translational and rotational Jacobians at body origins."""
        B = len(body_ids)
        Jp = np.empty((B, 3, self.nv))
        Jr = np.empty((B, 3, self.nv))
        m, d = self.model, self.data
        for i, b in enumerate(body_ids):
            mujoco.mj_jacBody(m, d, Jp[i], Jr[i], int(b))
        return Jp, Jr

    def positions(self, ps: PointSet):
        """World positions of a point set (call after ``forward``)."""
        d = self.data
        R = d.xmat[ps.body_ids].reshape(-1, 3, 3)
        return d.xpos[ps.body_ids] + np.einsum("bij,bj->bi", R, ps.local_pos)


def shift_jacobian(Jp, Jr, offset):
    """Jacobian of points at ``x_body + offset``: Jp - offset x Jr  (all batched).

    Jp, Jr: (P,3,nv); offset: (P,3) world-frame offset from body origin.
    """
    ox, oy, oz = offset[:, 0, None], offset[:, 1, None], offset[:, 2, None]
    rx, ry, rz = Jr[:, 0], Jr[:, 1], Jr[:, 2]
    # v = v_b + w x o  ->  w x o = -o x w
    out = Jp.copy()
    out[:, 0] += ry * oz - rz * oy
    out[:, 1] += rz * ox - rx * oz
    out[:, 2] += rx * oy - ry * ox
    return out


def segment_closest_points(a0, a1, b0, b1, eps=1e-12):
    """Vectorised closest points between segments [a0,a1] and [b0,b1]. Shapes (P,3)."""
    d1 = a1 - a0
    d2 = b1 - b0
    r = a0 - b0
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    denom = a * e - b * b
    s = np.where(denom > eps, np.clip((b * f - c * e) / np.maximum(denom, eps), 0.0, 1.0), 0.0)
    t = np.where(e > eps, (b * s + f) / np.maximum(e, eps), 0.0)
    # Clamp t and recompute s where needed.
    t_c = np.clip(t, 0.0, 1.0)
    s = np.where(t != t_c, np.clip(np.where(a > eps, (b * t_c - c) / np.maximum(a, eps), 0.0), 0.0, 1.0), s)
    t = t_c
    pa = a0 + s[:, None] * d1
    pb = b0 + t[:, None] * d2
    return pa, pb
