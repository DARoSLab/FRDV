"""Parallel-in-time retargeting with artifact-free seam handoff.

Why blending chunk overlaps causes artifacts
--------------------------------------------
Sequential retargeting is a deterministic causal filter, q_t = Phi(q_{t-1}, x_t):
the smoothing term makes each frame depend on the previous solution. Chunks
solved in parallel start from a guessed state, so near a seam two chunks
generally hold *different* solutions. Cross-fading them (smoothstep/slerp in
joint space) produces poses no IK solve ever produced: tracking error rises
mid-blend, feet can slide or dip below the ground (constraints are nonconvex,
so a blend of two feasible poses can be infeasible), and when the chunks sit
in different IK branches (e.g. knee/elbow/yaw) the blend sweeps through an
implausible pose and adds a velocity/acceleration bump.

Handoff stitching (Parareal-style correction)
---------------------------------------------
1. Parallel: chunk k solves frames [s_k - B, s_{k+1}) from a cold start. The
   B burn-in frames are discarded; they only let the filter "forget" its
   initial guess so the chunk usually already agrees with the sequential
   solution at s_k.
2. Seam handoff, for k = 1..K-1 in order: re-run the *same* per-frame
   solver from the true predecessor state q_{s_k - 1} and overwrite chunk
   k frame by frame until two consecutive re-solved frames match chunk k to within
   ``seam_tol`` (and the chunk is past its boosted cold-start frames). The solver is
   contractive, so the remaining chunk then converges onto the sequential solution;
   the residual difference is smooth but can grow to ~25x seam_tol before decaying,
   hence the tight default (1e-6: matches sequential to ~5e-6 rad on all of LAFAN1).

Every output frame is a genuine constrained IK solution from its true
predecessor, so there is no blending, no constraint violation and no
discontinuity larger than ``seam_tol``. If the chunks landed in different IK
branches the handoff simply runs longer (in the worst case it degenerates to
the sequential solve, which is still correct). Measured on LAFAN1 -> G1 with
burn_in=30: median ~30-100 frames per seam, p90 ~110-190, and a few percent of
seams (chunks on another wrist/shoulder branch) run 500+ frames or to the clip end.
Don't set ``max_handoff``: stopping before a match leaves a discontinuity.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from .retarget import IKConfig, Retargeter
from .utils import continuous_quaternions

_WORKER: dict = {}
# Per-frame problems are tiny (n ~ 20-60); multi-threaded BLAS in N worker processes only
# oversubscribes the cores. Children inherit these at spawn time, before numpy loads.
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS")


def _init_worker(cfg: IKConfig, ready):
    r = _WORKER["r"] = Retargeter(cfg)
    _prime(r)
    with ready.get_lock():
        ready.value += 1


def _prime(r: Retargeter):
    """Run a tiny solve so JIT kernels are loaded/compiled before real work arrives."""
    p = r.problem
    p.kin.forward(p.q_nominal, with_jac=False)
    x = p.kin.positions(p.pts)[: p.S]
    r.run(np.repeat(x[None], 3, axis=0))
    p.qp_active = None


def _solve_clip(args):
    i, targets = args
    return i, _WORKER["r"].run(targets)


def _solve_chunk(args):
    k, start, stop, targets = args
    t0 = time.perf_counter()
    q = _WORKER["r"].run(targets)
    return k, start, stop, q, time.perf_counter() - t0


@dataclass
class StitchReport:
    chunk_bounds: list = field(default_factory=list)
    handoff_frames: list = field(default_factory=list)   # frames re-solved per seam
    seam_residual: list = field(default_factory=list)    # pose gap where the handoff stopped
    parallel_time: float = 0.0
    handoff_time: float = 0.0
    ran_to_end: int = 0      # handoffs that re-solved to the end of the clip (correct, just serial)
    cut_short: int = 0       # handoffs stopped by max_handoff before matching (discontinuity!)

    def __str__(self):
        return (f"chunks={len(self.chunk_bounds)} handoff_frames={self.handoff_frames} "
                f"seam_residual_max={max(self.seam_residual, default=0):.2e} "
                f"ran_to_end={self.ran_to_end} cut_short={self.cut_short} "
                f"parallel={self.parallel_time:.3f}s handoff={self.handoff_time:.3f}s")


def chunk_bounds(T: int, n_chunks: int, min_chunk: int = 1):
    n = max(1, min(n_chunks, T // max(1, min_chunk)))
    edges = np.linspace(0, T, n + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def handoff(retargeter: Retargeter, targets, q, bounds, seam_tol=1e-6, max_frames=None,
            report: StitchReport | None = None, burn_in: int = 0):
    """Reconcile seams in place. ``q`` holds the concatenated chunk solutions.

    ``burn_in`` must be the burn-in the chunks were solved with: chunk k started cold at
    ``s_k - burn_in`` and its first ``boost_frames`` frames were solved with extra iterations
    (a different map than ``step``), so a match there does not imply the rest of the chunk agrees.
    The handoff therefore never stops before ``s_k - burn_in + boost_frames`` (0 = conservative).
    """
    T = len(targets)
    prob = retargeter.problem
    boost = int(getattr(retargeter.cfg, "boost_frames", 0) or 0)
    for s, _ in bounds[1:]:
        # The solver state is the previous frame (and the one before it when accel_weight > 0),
        # so the handoff stops only after two consecutive re-solved frames match the chunk.
        q_prev, q_prev2 = q[s - 1], (q[s - 2] if s >= 2 else None)
        t, gap, matched = s, np.inf, 0
        earliest_stop = max(s, s - burn_in) + max(0, boost - burn_in)
        limit = T if max_frames is None else min(T, s + max_frames)
        while t < limit:
            q_new = retargeter.step(q_prev, targets[t], q_prev2=q_prev2)
            gap = float(np.max(np.abs(prob.difference(q[t], q_new))))
            q[t] = q_new
            q_prev2, q_prev = q_prev, q_new
            t += 1
            matched = matched + 1 if gap < seam_tol else 0
            if matched >= 2 and t >= earliest_stop:
                break
        cut_short = t < T and matched < 2
        if cut_short:
            warnings.warn(f"handoff at frame {s} hit max_frames={max_frames} with the chunk still "
                          f"{gap:.3g} away: the output has a discontinuity there", RuntimeWarning, stacklevel=2)
        if report is not None:
            report.handoff_frames.append(t - s)
            report.seam_residual.append(gap)
            report.ran_to_end += int(t >= T)
            report.cut_short += int(cut_short)
    return q


class ParallelRetargeter:
    """Reusable worker pool; construct once, call many times (one call per motion clip)."""

    def __init__(self, cfg: IKConfig, n_workers: int | None = None, burn_in: int = 30,
                 seam_tol: float = 1e-6, max_handoff: int | None = None, min_chunk: int = 120,
                 start_method: str = "spawn", startup_timeout: float = 120.0):
        self.cfg = cfg
        self.n_workers = n_workers or os.cpu_count() or 1
        self.burn_in = burn_in
        self.seam_tol = seam_tol
        self.max_handoff = max_handoff
        self.min_chunk = min_chunk
        self.local = Retargeter(cfg)
        _prime(self.local)  # compiles numba kernels once in the parent; workers then load from cache
        self._pool = None
        if self.n_workers > 1:
            ctx = mp.get_context(start_method)
            ready = ctx.Value("i", 0)
            saved = {v: os.environ.get(v) for v in _THREAD_VARS}
            os.environ.update({v: "1" for v in _THREAD_VARS})
            try:
                self._pool = ctx.Pool(self.n_workers, initializer=_init_worker, initargs=(cfg, ready))
            finally:
                for v, val in saved.items():
                    if val is None:
                        os.environ.pop(v, None)
                    else:
                        os.environ[v] = val
            # Wait until every worker has built its model and loaded the kernels, so start-up
            # cost is paid (and reported) here rather than inside the first run.
            t_end = time.perf_counter() + startup_timeout
            while ready.value < self.n_workers:
                if time.perf_counter() > t_end:
                    raise TimeoutError("frdv workers failed to start")
                time.sleep(0.005)

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def run_many(self, clips):
        """Retarget several clips, one exact sequential solve per worker (no seams at all).

        This is the best use of cores for batch processing; use ``run`` to split a single
        long clip across workers.
        """
        clips = [np.asarray(c, dtype=float) for c in clips]
        if self._pool is None:
            return [self.local.run(c) for c in clips]
        out = [None] * len(clips)
        # Longest first for better load balance.
        order = sorted(range(len(clips)), key=lambda i: -len(clips[i]))
        for i, q in self._pool.imap_unordered(_solve_clip, [(i, clips[i]) for i in order]):
            out[i] = q
        return out

    def run(self, targets, return_report: bool = False):
        targets = np.asarray(targets, dtype=float)
        T = len(targets)
        report = StitchReport()
        bounds = chunk_bounds(T, self.n_workers, self.min_chunk)
        report.chunk_bounds = bounds

        t0 = time.perf_counter()
        if len(bounds) == 1 or self._pool is None:
            q = self.local.run(targets)
            report.parallel_time = time.perf_counter() - t0
            return (q, report) if return_report else q

        tasks = []
        for k, (s, e) in enumerate(bounds):
            a = max(0, s - self.burn_in) if k > 0 else 0
            tasks.append((k, a, e, targets[a:e]))
        q = np.empty((T, self.local.nq))
        for k, a, e, qk, _ in self._pool.imap_unordered(_solve_chunk, tasks):
            s = bounds[k][0]
            q[s:e] = qk[s - a:]
        report.parallel_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        handoff(self.local, targets, q, bounds, self.seam_tol, self.max_handoff, report, burn_in=self.burn_in)
        continuous_quaternions(q)  # chunks may start in the other quaternion hemisphere
        report.handoff_time = time.perf_counter() - t0
        return (q, report) if return_report else q
