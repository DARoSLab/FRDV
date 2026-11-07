"""Motion loading and conversion to robot targets.

Every format loads to a :class:`Motion` (raw units/axes). :func:`build_targets` turns it into
world-frame keypoint targets for a robot, driven by two small configs:

* dataset: file format, units, rotation to the robot's Z-up world, how to measure the performer
  and ground the clip, optional synthetic markers, and ``keypoints`` mapping
  canonical keypoint names -> joint names in this skeleton;
* robot: ``height`` and ``keypoints`` mapping canonical names -> robot links (+ weights).

The tracked set is the intersection of the two keypoint maps, so any robot works with any dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import amass, bvh
from .base import Motion

LOADERS = {"bvh": bvh.load, "amass": amass.load}

__all__ = ["Motion", "LOADERS", "load", "Targets", "build_targets", "enabled_markers", "keypoint_map"]


def load(path: str, format: str, **opts) -> Motion:
    if format not in LOADERS:
        raise ValueError(f"unknown motion format '{format}'; available: {', '.join(LOADERS)}")
    return LOADERS[format](path, **opts)


@dataclass
class Targets:
    X: np.ndarray                 # (T, K, 3) world targets, metres, Z-up, grounded
    keypoints: list[str]          # canonical names (K,)
    source_joints: list[str]      # joint name in the source skeleton for each keypoint
    fps: float
    scale: np.ndarray             # applied scale; source XYZ in manual mode, world XYZ otherwise
    human_height: float
    skeleton: np.ndarray | None = None        # (T, J, 3) every source joint, same transform
    skeleton_parents: list[int] = field(default_factory=list)
    segment_scale: dict | None = None         # limbs mode: {"root": leg ratio, canonical: segment ratio}


def _get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default) if not hasattr(cfg, "get") else cfg.get(key, default)


def enabled_markers(dataset) -> list:
    """Synthetic hand/arm orientation points are opt-in for dataset-driven loading."""
    return list(_get(dataset, "markers", []) or []) if _get(dataset, "augmentation", False) else []


def keypoint_map(dataset) -> dict:
    """Exclude synthetic targets as well as synthetic geometry when augmentation is off."""
    mapping = dict(_get(dataset, "keypoints", {}) or {})
    if not _get(dataset, "augmentation", False):
        synthetic = {m["name"] for m in (_get(dataset, "markers", []) or [])}
        mapping = {k: name for k, name in mapping.items() if name not in synthetic}
    return mapping


# Canonical keypoint tree used for per-segment scaling: child -> parent.
TREE = {
    "left_hip": "pelvis", "right_hip": "pelvis", "spine": "pelvis",
    "left_knee": "left_hip", "left_ankle": "left_knee", "left_toe": "left_ankle",
    "right_knee": "right_hip", "right_ankle": "right_knee", "right_toe": "right_ankle",
    "left_shoulder": "spine", "left_elbow": "left_shoulder", "left_wrist": "left_elbow",
    "right_shoulder": "spine", "right_elbow": "right_shoulder", "right_wrist": "right_elbow",
}
_ORDER = ["pelvis", "left_hip", "right_hip", "spine", "left_knee", "right_knee", "left_ankle", "right_ankle",
          "left_toe", "right_toe", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist"]


def _rotate(P, R, scale):
    """scale * (R @ p) for every point of P (..., 3), as one compiled pass / one GEMM."""
    from .. import _jit, _native
    P = np.ascontiguousarray(P, dtype=float)
    if _native.available() or _jit.AVAILABLE:
        out = np.empty_like(P)
        (_native if _native.available() else _jit).rotate_points(P.reshape(-1, 3), np.ascontiguousarray(R, dtype=float), float(scale), out.reshape(-1, 3))
        return out
    return (P.reshape(-1, 3) @ R.T).reshape(P.shape) * scale


def _matmul_const(A, B):
    """(T, 3, 3) @ (3, 3) as one GEMM (numpy's stacked matmul loops over T tiny products)."""
    if A.ndim == 2:
        return A @ B
    return (np.ascontiguousarray(A).reshape(-1, 3) @ B).reshape(A.shape)


def _run_plan(P, plan, rots):
    """Apply limb_rescale's joint assignments in order (compiled when numba is available)."""
    from .. import _jit, _native
    T, J, _ = P.shape
    out = np.full_like(P, np.nan)
    if _native.available() or _jit.AVAILABLE:
        n = len(plan)
        joint = np.array([p[0] for p in plan], np.int64)
        anchor = np.array([p[1] for p in plan], np.int64)
        kind = np.array([p[2] for p in plan], np.int64)
        ratio = np.array([p[3] for p in plan], float)
        rot_id = np.array([p[4] for p in plan], np.int64)
        off = np.array([p[5] if p[5] is not None else np.zeros(3) for p in plan], float).reshape(n, 3)
        R = np.ascontiguousarray(np.stack(rots) if rots else np.zeros((1, T, 3, 3)))
        (_native if _native.available() else _jit).limb_plan(np.ascontiguousarray(P), joint, anchor, kind, ratio,
                                                             rot_id, off, R, out)
        return out
    for j, a, kind, r, rid, off in plan:
        if kind == 2:
            out[:, j] = r * P[:, j]
        elif kind == 1:
            out[:, j] = out[:, a] + np.einsum("tij,j->ti", rots[rid], off)
        else:
            out[:, j] = out[:, a] + r * (P[:, j] - P[:, a])
    return out


def _finish(P, fac, idx, shift, dz, offset, extra):
    """((P[:, idx] * fac - shift), z -= dz) + offset + extra, elementwise in that order."""
    from .. import _jit, _native
    idx = np.asarray(idx, np.int64)
    if _native.available() or _jit.AVAILABLE:
        X = np.empty((P.shape[0], len(idx), 3))
        (_native if _native.available() else _jit).finish_targets(np.ascontiguousarray(P), np.asarray(fac, float), idx, shift, float(dz), offset, extra, X)
        return X
    X = P[:, idx] * fac - shift
    X[..., 2] -= dz
    return X + offset + extra


def limb_rescale(P, names, parents, kp_map, robot_points, fallback: float, clamp=(0.4, 2.5)):
    """Rescale a skeleton bone by bone to the robot's proportions.

    ``P`` (T, J, 3) metres; ``robot_points`` {canonical: world position at the robot's nominal pose}.
    * pelvis trajectory: scaled by the leg-length ratio (stride and pelvis height match the legs);
    * torso (hips, spine, shoulders): the robot's own nominal offsets, rotated with a pelvis frame
      (hip line + hip->shoulder line) and a chest frame (shoulder line), so hip width, waist and
      shoulder placement are the robot's, not a scaled human's;
    * limbs (hip->knee->ankle->toe, shoulder->elbow->wrist): performer's bone directions with the
      robot's bone lengths;
    * every other joint / marker: carried rigidly by its nearest canonical ancestor, scaled with
      that ancestor's segment.  Returns (P', info).
    """
    T, J, _ = P.shape
    have = {k: names.index(kp_map[k]) for k in _ORDER
            if k in kp_map and kp_map[k] in names and k in robot_points}
    if "pelvis" not in have:
        return P * fallback, {"root": fallback}

    def parent_of(k):
        p = TREE.get(k)
        while p is not None and p not in have:
            p = TREE.get(p)
        return p

    rp = {k: np.asarray(v, dtype=float) for k, v in robot_points.items()}
    from .. import _native
    pairs = [(have[k], have[parent_of(k)]) for k in _ORDER[1:] if k in have and parent_of(k) is not None]
    if _native.available():   # every bone length median in one GIL-free pass (same values)
        med = dict(zip(pairs, _native.bone_medians(P, pairs).tolist()))
    else:
        med = {}

    def bone_median(a, b):
        if (a, b) not in med:
            med[(a, b)] = float(np.median(np.linalg.norm(P[:, a] - P[:, b], axis=-1)))
        return med[(a, b)]

    ratio = {}
    for k in _ORDER[1:]:
        p = parent_of(k)
        if k not in have or p is None:
            continue
        h = bone_median(have[k], have[p])
        r = float(np.linalg.norm(rp[k] - rp[p]))
        ratio[k] = fallback if h < 0.02 else float(np.clip(r / h, clamp[0] * fallback, clamp[1] * fallback))
    legs = []
    for side in ("left", "right"):
        ch = [k for k in (f"{side}_hip", f"{side}_knee", f"{side}_ankle") if k in have]
        if len(ch) == 3:
            h = sum(bone_median(have[b], have[a]) for a, b in zip(ch, ch[1:]))
            r = sum(np.linalg.norm(rp[b] - rp[a]) for a, b in zip(ch, ch[1:]))
            legs.append(r / h)
    root = float(np.mean(legs)) if legs else fallback

    # Every output joint is one of: scaled root (kind 2), parent + ratio * human bone (kind 0), or
    # parent + rigid rotation @ robot offset (kind 1). Collect the assignments in order, then run them.
    plan = []                                   # (joint, anchor, kind, ratio, rot id, offset)
    plan.append((have["pelvis"], -1, 2, root, -1, None))
    joint_ratio = {have["pelvis"]: root}

    # Torso: rigid offsets taken from the robot (its hip width, waist and shoulder placement),
    # carried by a pelvis frame (hips) and a chest frame (shoulders) computed the same way on the
    # performer and on the robot.  Limbs: performer's bone directions with the robot's lengths.
    def frame(pts, lat_a, lat_b, up_a, up_b):
        lat = pts(lat_a) - pts(lat_b)
        up = pts(up_a) - pts(up_b)
        lat = lat / np.maximum(np.linalg.norm(lat, axis=-1, keepdims=True), 1e-9)
        up = up - (up * lat).sum(-1, keepdims=True) * lat
        up = up / np.maximum(np.linalg.norm(up, axis=-1, keepdims=True), 1e-9)
        return np.stack([np.cross(lat, up), lat, up], axis=-1)       # columns: forward, left, up

    def mid(get, a, b):
        return 0.5 * (get(a) + get(b))

    hum = lambda k: P[:, have[k]]                                    # noqa: E731
    rob = lambda k: rp[k]                                            # noqa: E731
    hips_ok = all(k in have for k in ("left_hip", "right_hip"))
    sh_ok = all(k in have for k in ("left_shoulder", "right_shoulder"))
    rigid = {}
    rots = []
    if hips_ok:
        up_pts = (("left_shoulder", "right_shoulder") if sh_ok else None)
        if up_pts:
            hu = lambda k: mid(hum, *up_pts) if k == "_sh" else (mid(hum, "left_hip", "right_hip") if k == "_hp" else hum(k))  # noqa: E731
            ru = lambda k: mid(rob, *up_pts) if k == "_sh" else (mid(rob, "left_hip", "right_hip") if k == "_hp" else rob(k))  # noqa: E731
            Fr = frame(ru, "left_hip", "right_hip", "_sh", "_hp")
            Fh = None if _native.available() else frame(hu, "left_hip", "right_hip", "_sh", "_hp")
        else:
            up_h = np.broadcast_to([0.0, 0.0, 1.0], P[:, 0].shape)
            hu = lambda k: up_h if k == "_up" else (0 * up_h if k == "_0" else hum(k))  # noqa: E731
            ru = lambda k: np.array([0.0, 0.0, 1.0]) if k == "_up" else (np.zeros(3) if k == "_0" else rob(k))  # noqa: E731
            Fr = frame(ru, "left_hip", "right_hip", "_up", "_0")
            Fh = None if _native.available() else frame(hu, "left_hip", "right_hip", "_up", "_0")
        if Fh is None:
            sh = (have["left_shoulder"], have["right_shoulder"]) if up_pts else (-1, -1)
            rots.append(_native.torso_frames(P, have["left_hip"], have["right_hip"], *sh, have["left_hip"],
                                             have["right_hip"], Fr))
        else:
            rots.append(_matmul_const(Fh, Fr.T))                     # (T, 3, 3) robot nominal -> now
        for k in ("left_hip", "right_hip", "spine"):
            if k in have:
                rigid[k] = 0
        if sh_ok:
            hu2 = lambda k: mid(hum, "left_shoulder", "right_shoulder") if k == "_sh" else (  # noqa: E731
                mid(hum, "left_hip", "right_hip") if k == "_hp" else hum(k))
            ru2 = lambda k: mid(rob, "left_shoulder", "right_shoulder") if k == "_sh" else (  # noqa: E731
                mid(rob, "left_hip", "right_hip") if k == "_hp" else rob(k))
            Fr2 = frame(ru2, "left_shoulder", "right_shoulder", "_sh", "_hp")
            if _native.available():
                rots.append(_native.torso_frames(P, have["left_shoulder"], have["right_shoulder"],
                                                 have["left_shoulder"], have["right_shoulder"], have["left_hip"],
                                                 have["right_hip"], Fr2))
            else:
                rots.append(_matmul_const(frame(hu2, "left_shoulder", "right_shoulder", "_sh", "_hp"), Fr2.T))
            rigid["left_shoulder"] = rigid["right_shoulder"] = 1

    for k in _ORDER[1:]:
        if k not in have or parent_of(k) is None:
            continue
        i, kp = have[k], parent_of(k)
        ip = have[kp]
        if k in rigid:
            plan.append((i, ip, 1, 0.0, rigid[k], rp[k] - rp[kp]))
            joint_ratio[i] = root
        elif k in ratio:
            plan.append((i, ip, 0, ratio[k], -1, None))
            joint_ratio[i] = ratio[k]
    # every other joint: rigidly attached to its nearest canonical ancestor (or the nearest one)
    canon = set(joint_ratio)
    for j in range(J):
        if j in canon:
            continue
        a = j
        while a >= 0 and a not in canon:
            a = parents[a] if parents and a < len(parents) else -1
        if a < 0:
            cj = sorted(canon)
            a = cj[int(np.argmin(np.linalg.norm(P[0, cj] - P[0, j], axis=-1)))]
        plan.append((j, a, 0, joint_ratio[a], -1, None))
    out = _run_plan(P, plan, rots)
    return out, {"root": root, **ratio}


def build_targets(motion: Motion, dataset, robot_height: float, robot_keypoints,
                  extra_offset=(0.0, 0.0, 0.0), keep_skeleton: bool = False,
                  robot_points: dict | None = None) -> Targets:
    """Convert a loaded motion to world-frame robot targets.

    ``dataset`` keys (dict / OmegaConf):
      units: metres per source unit (0.01 for cm)
      rotation: 3x3, source axes -> world (Z-up)
      keypoints: {canonical: source joint}
      height: {source: frame0|rest, exclude: [joints], margin: m}   human height estimate
      scale: {mode: manual|limbs|height|fixed, value: [sx,sy,sz] (manual/fixed), per_axis: [kx,ky,kz],
              add: [ax,ay,az] (metres per source unit, added to the per-axis factor)}
      ground: {mode: none|frame0|clip, joints: [...], margin: m}
      upper_body_offset: world-metre translation for upper_body_joints, after scaling/rotation
        manual: fixed source-XYZ factors after unit conversion, before rotation (V1 convention).
        fixed: legacy fixed world-XYZ factors, after rotation.
        limbs: every bone scaled to the robot's segment lengths (needs ``robot_points``,
               canonical keypoint -> world position at the robot's nominal pose); falls back to
               ``height`` without them.  height: one factor robot_height / performer_height.
      root: joint used to zero the XY position at frame 0
      offset: [x,y,z] added at the end (metres)
    """
    names = motion.names
    kp_map = keypoint_map(dataset)
    robot_kp = list(robot_keypoints)
    keypoints = [k for k in robot_kp if k in kp_map and kp_map[k] in names]
    if not keypoints:
        raise ValueError("no keypoint is shared by the robot and dataset configs "
                         f"(robot: {robot_kp}; dataset: {sorted(kp_map)})")
    src = [kp_map[k] for k in keypoints]
    idx = [names.index(s) for s in src]

    units = float(_get(dataset, "units", 1.0))
    R = np.asarray(_get(dataset, "rotation", np.eye(3)), dtype=float).reshape(3, 3)
    P = _rotate(motion.positions, R, units)                  # (T, J, 3) metres, world axes

    # --- performer height -> scale ----------------------------------------------------------
    hcfg = _get(dataset, "height", {}) or {}
    exclude = set(_get(hcfg, "exclude", []) or [])
    live = [i for i, n in enumerate(names) if n not in exclude]
    source = _get(hcfg, "source", "frame0")
    ref = (motion.rest * units @ R.T) if (source == "rest" and motion.rest is not None) else P[0]
    human_height = float(np.ptp(ref[live, 2]) + float(_get(hcfg, "margin", 0.0)))

    scfg = _get(dataset, "scale", {}) or {}
    mode = _get(scfg, "mode", "height")
    segment_scale = None
    if mode == "limbs" and not robot_points:
        mode = "height"
    if mode == "limbs":
        s = np.full(3, robot_height / human_height)
    elif mode == "height":
        s = np.full(3, robot_height / human_height)
    elif mode in ("manual", "fixed"):
        value = _get(scfg, "value", None if mode == "manual" else 1.0)
        if value is None:
            raise ValueError("manual scaling requires dataset.scale.value=[sx,sy,sz]; select scaling=manual")
        try:
            s = np.broadcast_to(np.asarray(value, dtype=float), (3,)).copy()
        except (TypeError, ValueError) as e:
            raise ValueError("dataset.scale.value must be a positive scalar or three XYZ factors") from e
    else:
        raise ValueError(f"scale.mode must be 'manual', 'limbs', 'height' or 'fixed', got {mode}")
    s = s * np.asarray(_get(scfg, "per_axis", [1.0, 1.0, 1.0]), dtype=float)
    # Optional additive correction to the per-axis factor, in metres per *source unit*
    # (k = s * units * per_axis + add), for reproducing calibrations expressed that way.
    add = np.asarray(_get(scfg, "add", [0.0, 0.0, 0.0]), dtype=float)
    s = s + add / units
    if not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError("scale factors must be finite and positive")
    if mode == "limbs":
        P, segment_scale = limb_rescale(P, names, motion.parents, kp_map, robot_points, float(s.mean()))
        fac = s / s.mean()              # optional per-axis / add corrections stay relative
        s = np.full(3, segment_scale["root"])
    elif mode == "manual":
        # V1 scales along source axes before rotating into world coordinates. Folding the
        # diagonal scale into R preserves that order without copying the full raw skeleton.
        P = _rotate(motion.positions, R @ np.diag(s), units)
        fac = np.ones(3)
    else:
        fac = s
    # P * fac is applied lazily (only to the joints that are needed) on the fast path below

    # V1 AMASS alignment shifts the upper torso/arms after scaling and rotation. Keep the
    # offset in world metres even when the optional axis factors are applied lazily below.
    upper = np.asarray(_get(dataset, "upper_body_offset", (0.0, 0.0, 0.0)), dtype=float)
    if upper.shape != (3,) or not np.isfinite(upper).all():
        raise ValueError("upper_body_offset must contain three finite world-metre values")
    if np.any(upper):
        upper_names = set(_get(dataset, "upper_body_joints", []) or [])
        upper_idx = [i for i, name in enumerate(names) if name in upper_names]
        if not upper_idx:
            raise ValueError("upper_body_offset requires matching upper_body_joints")
        P = P.copy()
        P[:, upper_idx] += upper / fac

    # --- zero root XY at frame 0, ground -------------------------------------------------------
    root = _get(dataset, "root", None)
    r = names.index(root) if root in names else 0
    gcfg = _get(dataset, "ground", {}) or {}
    gmode = _get(gcfg, "mode", "frame0")
    if gmode not in ("none", "frame0", "clip"):
        raise ValueError(f"ground.mode must be 'none', 'frame0' or 'clip', got {gmode}")
    gj = [names.index(j) for j in (_get(gcfg, "joints", []) or []) if j in names] or live
    P0 = P[0] * fac
    shift = np.array([P0[r, 0], P0[r, 1], 0.0])
    offset = np.asarray(_get(dataset, "offset", [0.0, 0.0, 0.0]), dtype=float)
    extra = np.asarray(extra_offset, dtype=float)
    margin = float(_get(gcfg, "margin", 0.0))
    if gmode in ("none", "frame0") and not keep_skeleton:
        # Only frame 0 is needed for the floor; transform just the tracked joints (same ops, in the same order).
        floor = (P0[gj] - shift)[:, 2].min() if gmode == "frame0" else 0.0
        if gmode == "none":
            margin = 0.0
        X = _finish(P, fac, idx, shift, floor - margin, offset, extra)
        return Targets(X=X, keypoints=keypoints, source_joints=src, fps=motion.fps, scale=s,
                       human_height=human_height, skeleton=None,
                       skeleton_parents=list(motion.parents or []), segment_scale=segment_scale)
    P = P * fac - shift
    floor = (P[0, gj, 2].min() if gmode == "frame0" else P[:, gj, 2].min()) if gmode != "none" else 0.0
    if gmode == "none":
        margin = 0.0
    P[..., 2] -= floor - margin
    P = P + offset + extra

    return Targets(X=P[:, idx], keypoints=keypoints, source_joints=src, fps=motion.fps, scale=s,
                   human_height=human_height, skeleton=P if keep_skeleton else None,
                   skeleton_parents=list(motion.parents or []), segment_scale=segment_scale)
