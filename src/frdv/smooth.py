"""Non-causal clean-up of joint "snaps" in a retargeted trajectory.

The per-frame solver is causal (each frame is solved from the previous one), so when the best
pose jumps to another branch -- a joint leaving a limit it was pinned against, the foot being
pushed out of the floor for one frame -- the result snaps within a frame or two. Offline we know
the whole clip, so we can fix it after the fact:

1. detect isolated spikes: a joint step much larger than the steps around it;
2. around each spike, smooth the trajectory with a zero-phase Gaussian (tangent space), tapered
   to zero change at the window edges so the clean-up blends in continuously;
3. re-project every frame of the window onto the constraints with the same IK, pulled strongly
   towards the smoothed pose -- joint limits, ground and collision constraints stay satisfied and
   the targets are still tracked as well as the smoothness allows.

Only the windows around spikes are touched, so the cost is small.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .kinematics import Kinematics
from .retarget import IKProblem, Retargeter
from .sqp import SQPOptions


@dataclass
class DespikeReport:
    spikes_before: int = 0
    spikes_after: int = 0
    windows: list = field(default_factory=list)
    worst_before: float = 0.0
    worst_after: float = 0.0
    reverted: int = 0

    def __str__(self):
        return (f"spikes {self.spikes_before} -> {self.spikes_after}, worst step "
                f"{self.worst_before:.2f} -> {self.worst_after:.2f} rad, {len(self.windows)} window(s) "
                f"repaired, {self.reverted} reverted")


def joint_steps(problem, q):
    """(T-1, nv) tangent-space steps between consecutive frames (= mj_differentiatePos, vectorised).

    Free joint: world-frame position change and the body-frame rotation vector log(q0^-1 q1);
    hinge / slide joints: plain differences.
    """
    q = np.asarray(q, dtype=float)
    if len(q) < 2:
        return np.zeros((0, problem.nv))
    a, b = q[:-1], q[1:]
    out = np.empty((len(q) - 1, problem.nv))
    out[:, :3] = b[:, :3] - a[:, :3]
    out[:, 6:] = b[:, 7:] - a[:, 7:]
    out[:, 3:6] = _quat_log_diff(a[:, 3:7], b[:, 3:7])
    return out


def _quat_log_diff(qa, qb):
    """Rotation vectors of conj(qa) * qb for (N, 4) wxyz quaternions, like mju_subQuat."""
    qa = qa / np.linalg.norm(qa, axis=1, keepdims=True)
    qb = qb / np.linalg.norm(qb, axis=1, keepdims=True)
    w0, v0 = qa[:, :1], -qa[:, 1:]                     # conj(qa)
    w1, v1 = qb[:, :1], qb[:, 1:]
    w = w0 * w1 - np.sum(v0 * v1, axis=1, keepdims=True)
    v = w0 * v1 + w1 * v0 + np.cross(v0, v1)
    flip = w < 0                                      # shortest rotation
    w, v = np.where(flip, -w, w), np.where(flip, -v, v)
    sin_a2 = np.linalg.norm(v, axis=1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_a2, w)
    scale = np.where(sin_a2 > 1e-15, angle / np.maximum(sin_a2, 1e-300), 2.0)
    return v * scale


def find_spikes(steps, threshold=0.25, ratio=4.0, k=7):
    """Step indices t (frame t -> t+1) where some dof moves > threshold and > ratio x the median
    step of that dof over the surrounding +-k steps (vectorised rolling median)."""
    a = np.abs(steps)
    T, n = a.shape
    if T < 3:
        return [], 0.0
    # only steps above the threshold can be spikes: take the rolling median just for those
    # (edge-clamped neighbours = the "edge" padding of a full rolling window)
    ti, di = np.nonzero(a > threshold)
    if ti.size == 0:
        return [], 0.0
    off = np.r_[np.arange(-k, 0), np.arange(1, k + 1)]
    nb = a[np.clip(ti[:, None] + off, 0, T - 1), di[:, None]]
    base = np.maximum(np.median(nb, axis=1), 0.02)
    sel = a[ti, di] > ratio * base
    rows = np.unique(ti[sel])
    worst = float(a[ti[sel], di[sel]].max()) if rows.size else 0.0
    return rows.tolist(), worst


def _window_score(problem, q, a, b):
    """(peak joint acceleration over the window incl. its borders, worst ground violation)."""
    lo, hi = max(0, a - 2), min(len(q), b + 2)
    st = joint_steps(problem, q[lo:hi])[:, 6:]
    acc = float(np.abs(np.diff(st, axis=0)).max()) if len(st) > 1 else 0.0
    viol = 0.0
    for t in range(a, b):
        problem.kin.forward(q[t], with_jac=False)
        viol = max(viol, float(-problem._ground(problem._positions()).min()))
    return acc, max(viol, 0.0)


def _windows(hits, half, T):
    out = []
    for t in hits:
        a, b = max(1, t - half), min(T - 1, t + half + 1)
        if out and a <= out[-1][1] + 2:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _smooth_window(problem, q, a, b, sigma):
    """Zero-phase Gaussian smoothing of q[a:b] in the tangent space at q[a-1], tapered to the edges."""
    anchor = q[a - 1]
    pad = int(np.ceil(3 * sigma))
    lo, hi = max(0, a - pad), min(len(q), b + pad)
    V = np.array([problem.difference(anchor, q[t]) for t in range(lo, hi)])
    k = np.arange(-pad, pad + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    Vp = np.pad(V, ((pad, pad), (0, 0)), mode="edge")
    Vs = np.stack([np.convolve(Vp[:, i], w, mode="valid") for i in range(V.shape[1])], axis=1)
    n = b - a
    taper = np.sin(np.pi * (np.arange(n) + 1) / (n + 1)) ** 2  # 0 at the window edges, 1 in the middle
    out = []
    for i, t in enumerate(range(a, b)):
        v = V[t - lo] + taper[i] * (Vs[t - lo] - V[t - lo])
        out.append(problem.integrate(anchor, v))
    return np.array(out)


def despike(retargeter: Retargeter, targets, q, threshold=0.25, ratio=4.0, half_window=5, sigma=1.5,
            pull_weight=200.0, iters=2, passes=3):
    """Remove isolated joint snaps from ``q`` (in place) and return a :class:`DespikeReport`.

    Each pass re-processes only the spikes that remain, with a wider window, longer smoothing and
    a stronger pull towards the smoothed pose, so stubborn branch switches get spread out further.
    ``pull_weight`` sets how strongly the re-projection follows the smoothed pose relative to the
    keypoint weights (higher = smoother, lower = tighter tracking).
    """
    p = retargeter.problem
    rep = DespikeReport()
    hits, rep.worst_before = find_spikes(joint_steps(p, q), threshold, ratio)
    rep.spikes_before = len(hits)
    for k in range(passes):
        if not hits:
            break
        # A solver that pulls towards a given pose instead of the previous frame.
        cfg = replace(retargeter.cfg, smoothing_weight=pull_weight * 4 ** k, accel_weight=0.0,
                      rot_smoothing_scale=1.0, max_joint_step=None, sqp=SQPOptions(max_iter=iters, line_search=True))
        proj = Retargeter(cfg, IKProblem(cfg, Kinematics(model=p.kin.model)))   # reuse the loaded model
        for a, b in _windows(hits, half_window + 4 * k, len(q)):
            qs = _smooth_window(p, q, a, b, sigma * (1 + k))
            old = q[a:b].copy()
            score0 = _window_score(p, q, a, b)
            for i, t in enumerate(range(a, b)):
                q[t] = proj.step(qs[i], targets[t])
            # Keep the repair only if it is smoother and no less feasible; otherwise revert.
            score1 = _window_score(p, q, a, b)
            if score1[0] < score0[0] and score1[1] <= score0[1] + 1e-3:
                rep.windows.append((a, b))
            else:
                q[a:b] = old
                rep.reverted += 1
        hits, _ = find_spikes(joint_steps(p, q), threshold, ratio)
    hits, rep.worst_after = find_spikes(joint_steps(p, q), threshold, ratio)
    rep.spikes_after = len(hits)
    return rep
