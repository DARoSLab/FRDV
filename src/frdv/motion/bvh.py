"""Vectorised BVH reader and forward kinematics for LAFAN1 motion clips."""

from __future__ import annotations

import numpy as np

from .base import Motion
from ..validation import ValidationError


def _parse(path: str):
    with open(path, "rb") as f:
        blob = f.read()
    cut = blob.find(b"MOTION")
    if cut < 0:
        raise ValueError(f"{path}: no MOTION section")
    hierarchy = blob[:cut].decode()
    names, parents, offsets, channels, starts = [], [], [], [], []
    stack: list[int] = []
    n_chan = 0
    cur = -1
    for raw in hierarchy.splitlines():
        tok = raw.split()
        if not tok:
            continue
        key = tok[0]
        if key in ("ROOT", "JOINT", "End"):
            name = tok[1] if key != "End" else f"{names[stack[-1]]}_EndSite"
            names.append(name)
            parents.append(stack[-1] if stack else -1)
            offsets.append(np.zeros(3))
            channels.append([])
            starts.append(0)
            cur = len(names) - 1
        elif key == "OFFSET":
            offsets[cur] = np.array([float(v) for v in tok[1:4]])
        elif key == "CHANNELS":
            channels[cur] = tok[2:]
            starts[cur] = n_chan
            n_chan += int(tok[1])
        elif key == "{":
            stack.append(cur)
        elif key == "}":
            stack.pop()
            cur = stack[-1] if stack else -1
    ft = blob.find(b"Frame Time:", cut)
    if ft < 0:
        raise ValueError(f"{path}: no 'Frame Time:' line")
    head = blob[cut:ft].split()
    n_frames = int(head[head.index(b"Frames:") + 1]) if b"Frames:" in head else -1
    i = ft + len(b"Frame Time:")
    while i < len(blob) and blob[i:i + 1].isspace():
        i += 1
    j = i
    while j < len(blob) and not blob[j:j + 1].isspace():
        j += 1
    frame_time = float(blob[i:j])
    if not np.isfinite(frame_time) or frame_time <= 0:
        raise ValidationError(f"{path}: Frame Time must be finite and positive; got {frame_time}")
    data = _parse_numbers(blob, j, n_frames * n_chan)
    if data is None:  # something the fast reader does not handle (nan, very long mantissas, ...)
        data = np.fromstring(blob[j:].decode(), sep=" ")
    if n_chan <= 0 or data.size == 0 or data.size % n_chan:
        raise ValidationError(f"{path}: motion values do not form complete {n_chan}-channel frames")
    if not np.isfinite(data).all() or not np.isfinite(offsets).all():
        raise ValidationError(f"{path}: BVH motion or skeleton offsets contain NaN or infinity")
    return names, parents, offsets, channels, starts, data.reshape(-1, n_chan), frame_time


def _parse_numbers(raw: bytes, start: int, expected: int):
    """Whitespace-separated decimals from raw[start:], exactly like np.fromstring(sep=" "), or None."""
    from .. import _jit, _native
    if _native.available():
        return _native.parse_decimals(raw, start, expected)
    if not _jit.AVAILABLE:
        return None
    buf = np.frombuffer(raw, dtype=np.uint8)
    out = np.empty(expected if expected > 0 else (len(raw) - start) // 2 + 1)
    n = _jit.parse_decimals(buf, start, out)
    if n < 0:
        if expected > 0:  # the Frames: header was wrong; retry without trusting it
            return _parse_numbers(raw, start, -1)
        return None
    return out[:n]


def _axis_rot(axis, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    i, j = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[axis]
    k = 3 - i - j
    R[:, k, k] = 1.0
    R[:, i, i] = c
    R[:, j, j] = c
    R[:, i, j] = -s
    R[:, j, i] = s
    return R


def _channel_codes(channels, starts, n):
    kind = np.zeros(n, np.int64)
    axis = np.zeros(n, np.int64)
    for j, ch in enumerate(channels):
        for c, name in enumerate(ch):
            kind[starts[j] + c] = 0 if name.endswith("position") else 1
            axis[starts[j] + c] = "XYZ".index(name[0])
    return kind, axis


def _fk(parents, offsets, channels, starts, data, rot_for=()):
    """World positions (T, J, 3) and world rotations for the joints in ``rot_for``."""
    from .. import _jit, _native
    if _native.available() or _jit.AVAILABLE:
        J = len(parents)
        kind = np.zeros(data.shape[1], np.int64)
        axis = np.zeros(data.shape[1], np.int64)
        for j in range(J):
            for c, ch in enumerate(channels[j]):
                kind[starts[j] + c] = 0 if ch.endswith("position") else 1
                axis[starts[j] + c] = "XYZ".index(ch[0])
        sel = sorted(rot_for)
        slot = np.full(J, -1, np.int64)
        slot[sel] = np.arange(len(sel))
        pos = np.empty((data.shape[0], J, 3))
        rot = np.empty((data.shape[0], max(len(sel), 1), 3, 3))
        fk = _native.bvh_fk_sel if _native.available() else _jit.bvh_fk_sel
        fk(np.asarray(parents, np.int64), np.asarray(offsets, float), np.asarray(starts, np.int64),
           np.array([len(c) for c in channels], np.int64), kind, axis, np.ascontiguousarray(data), slot, pos, rot)
        return pos, {j: rot[:, slot[j]] for j in sel}
    return _fk_numpy(parents, offsets, channels, starts, data, rot_for)


def _fk_numpy(parents, offsets, channels, starts, data, rot_for=()):
    T, J = data.shape[0], len(parents)
    pos = np.zeros((T, J, 3))
    rot = [None] * J
    eye = np.broadcast_to(np.eye(3), (T, 3, 3))
    for j in range(J):  # parents precede children in BVH order
        t = np.broadcast_to(offsets[j], (T, 3)).copy()
        R = None
        for c, ch in enumerate(channels[j]):
            col = data[:, starts[j] + c]
            if ch.endswith("position"):
                t[:, "XYZ".index(ch[0])] += col
            else:
                Rc = _axis_rot(ch[0], col)
                R = Rc if R is None else R @ Rc
        p = parents[j]
        if p < 0:
            rot[j] = R if R is not None else eye
            pos[:, j] = t
        else:
            pos[:, j] = pos[:, p] + np.einsum("tij,tj->ti", rot[p], t)
            rot[j] = rot[p] if R is None else rot[p] @ R
    return pos, {j: rot[j] for j in rot_for}


def load(path: str, markers=(), **_) -> Motion:
    """Load a BVH clip in raw units/axes.

    ``markers``: synthetic points rigidly attached to a joint,
    ``[{name, parent, offset: [x, y, z]}]`` with the offset in the parent joint's local frame
    (BVH units). Useful when the skeleton lacks points the robot should track (finger tips,
    points on the upper/lower arm for roll, ...).
    """
    names, parents, offsets, channels, starts, data, frame_time = _parse(path)
    index = {n: i for i, n in enumerate(names)}
    for m in markers:
        if m["parent"] not in index:
            raise KeyError(f"marker '{m['name']}': parent joint '{m['parent']}' not in {path}")
    from .. import _native
    fused_markers = _native.available()
    if fused_markers:  # joints and markers in one compiled pass
        kind, axis = _channel_codes(channels, starts, data.shape[1])
        pos = _native.bvh_fk_markers(parents, offsets, starts, [len(c) for c in channels], kind, axis, data,
                                     [index[m["parent"]] for m in markers],
                                     [np.asarray(m["offset"], dtype=float) for m in markers] or np.zeros((0, 3)))
    else:
        rot_for = {index[m["parent"]] for m in markers}
        pos, rots = _fk(parents, offsets, channels, starts, data, rot_for)

    # Rest pose (all rotations zero, root at origin): used for body-size estimation.
    rest = np.zeros((len(names), 3))
    for j, p in enumerate(parents):
        rest[j] = offsets[j] if p < 0 else rest[p] + offsets[j]
    rest[0] = 0.0

    extra_names, extra_pos, extra_rest = [], [], []
    for m in markers:
        pj = index[m["parent"]]
        off = np.asarray(m["offset"], dtype=float)
        extra_names.append(m["name"])
        if not fused_markers:
            extra_pos.append(pos[:, pj] + np.einsum("tij,j->ti", rots[pj], off))
        extra_rest.append(rest[pj] + off)
    if markers:
        names = names + extra_names
        if not fused_markers:
            pos = np.concatenate([pos, np.stack(extra_pos, axis=1)], axis=1)
        rest = np.concatenate([rest, np.array(extra_rest)])
    return Motion(names=names, positions=pos, fps=1.0 / frame_time, rest=rest,
                  parents=parents + [index[m["parent"]] for m in markers])
