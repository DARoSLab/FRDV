"""Dense strictly-convex QP solver (Goldfarb-Idnani dual active set, warm-startable).

Solves

    min_x  0.5 x'Hx + g'x
    s.t.   lb <= x[bidx] <= ub         (simple bounds on a subset of variables)
           C x >= d                    (general inequalities)

with H symmetric positive definite.

Why this method for per-frame IK:
  * The QP is small and dense (n ~ 20-60), so sparse ADMM/IPM solvers spend most
    of their time on setup and data conversion rather than math.
  * The dual method starts from the unconstrained minimiser and only touches
    constraints that are violated; no feasible start or tuning is needed and it
    terminates finitely with an exact solution.
  * Warm start: consecutive frames share almost the same active set (feet on the
    ground, joints at limits). Passing the previous active set usually turns the
    solve into a single KKT solve plus a feasibility/sign check.

Because H is positive definite the solution is unique, so warm starting only
changes speed, never the answer.
"""

from __future__ import annotations

import numpy as np

from . import _jit

__all__ = ["solve_qp", "QPResult"]

_INF = np.inf


class QPResult:
    __slots__ = ("x", "status", "iters", "active")

    def __init__(self, x, status, iters, active):
        self.x = x
        self.status = status  # 0 = solved, 1 = infeasible, 2 = max iterations
        self.iters = iters
        self.active = active  # indices into the stacked rows [C; I_b; -I_b]

    @property
    def solved(self) -> bool:
        return self.status == 0


_EYE_CACHE: dict = {}


def _eye(n):
    e = _EYE_CACHE.get(n)
    if e is None:
        e = _EYE_CACHE[n] = np.eye(n)
    return e


_BOUND_CACHE: dict = {}


def _bound_block(n, bidx):
    key = (n, bidx.tobytes() if hasattr(bidx, "tobytes") else tuple(bidx))
    E = _BOUND_CACHE.get(key)
    if E is None:
        I = np.eye(n)[np.asarray(bidx)]
        E = _BOUND_CACHE[key] = np.concatenate((I, -I))
    return E


_EMPTY_I = np.zeros(0, dtype=np.int64)
_EMPTY_F = np.zeros(0)


def solve_qp(H, g, C=None, d=None, bidx=None, lb=None, ub=None,
             tol=1e-9, max_iter=200, active0=None, use_jit=None) -> QPResult:
    """See module docstring. Dispatches to the numba kernel when available."""
    if use_jit is None:
        use_jit = _jit.AVAILABLE
    if not use_jit:
        return _solve_qp_numpy(H, g, C, d, bidx, lb, ub, tol, max_iter, active0)
    n = g.shape[0]
    if C is None or C.shape[0] == 0:
        C, d = np.zeros((0, n)), _EMPTY_F
    if bidx is None or len(bidx) == 0:
        bidx, lb, ub = _EMPTY_I, _EMPTY_F, _EMPTY_F
    a0 = np.asarray(active0, dtype=np.int64) if active0 else _EMPTY_I
    if bidx.dtype != np.int64:
        bidx = bidx.astype(np.int64)
    x, status, iters, act, na = _jit.gi_qp(H, g, C, d, bidx, lb, ub, a0, tol, max_iter)
    return QPResult(x, int(status), int(iters), act[:na].tolist())


def _solve_qp_numpy(H, g, C=None, d=None, bidx=None, lb=None, ub=None,
                    tol=1e-9, max_iter=200, active0=None) -> QPResult:
    """Solve the QP above. Inputs are not modified.

    Args:
        H: (n, n) SPD Hessian.  g: (n,) linear term.
        C, d: (m, n), (m,) rows of ``C x >= d`` (optional).
        bidx: (k,) indices of bounded variables; lb, ub: (k,) bounds (may be +-inf).
        active0: optional warm-start active set (``QPResult.active`` of a previous
            solve with the same constraint layout).
    """
    n = g.shape[0]
    Hinv = np.linalg.inv(H)  # n is small; one LAPACK call beats repeated triangular solves from Python

    # Stack every constraint as A x >= b  (bound block [I_b; -I_b] is cached per layout).
    has_c = C is not None and C.shape[0] > 0
    has_b = bidx is not None and len(bidx) > 0
    if has_b:
        Eb = _bound_block(n, bidx)
        bb = np.concatenate((lb, -ub))
        A = np.concatenate((C, Eb)) if has_c else Eb
        b = np.concatenate((d, bb)) if has_c else bb
    elif has_c:
        A, b = C, d
    else:
        return QPResult(-(Hinv @ g), 0, 0, [])
    mtot = A.shape[0]

    HinvAT = None  # lazily computed Hinv @ A.T (only needed if constraints become active)
    x = -(Hinv @ g)
    active: list[int] = []
    u = np.zeros(0)
    it = 0

    # ---- warm start: re-impose the previous active set, shed wrong-sign multipliers.
    if active0:
        W = [i for i in active0 if i < mtot]
        if W:
            W = [i for i, ok in zip(W, np.isfinite(b[W]).tolist()) if ok]
        HinvAT = Hinv @ A.T
        Hg = -x  # Hinv @ g
        while W:
            N = A[W]
            HN = HinvAT[:, W]
            M = N @ HN
            try:
                lam = np.linalg.solve(M, b[W] + N @ Hg)
            except np.linalg.LinAlgError:
                W = []
                break
            it += 1
            neg = lam < 0.0
            if not neg.any():
                x = HN @ lam - Hg
                active, u = W, lam
                break
            W = [w for w, ng in zip(W, neg) if not ng]

    s = A @ x - b
    p = int(np.argmin(s))
    if s[p] >= -tol:
        return QPResult(x, 0, it, active)
    if HinvAT is None:
        HinvAT = Hinv @ A.T

    while True:
        if it >= max_iter:
            return QPResult(x, 2, it, active)
        n_p = A[p]
        Hn = HinvAT[:, p]
        u_p = 0.0
        while True:
            it += 1
            if active:
                N = A[active]
                HN = HinvAT[:, active]
                r = np.linalg.solve(N @ HN, N @ Hn)
                z = Hn - HN @ r
                # Dual step length: keep active multipliers nonnegative.
                ratio = np.where(r > 1e-14, u / np.where(r > 1e-14, r, 1.0), _INF)
                kdrop = int(np.argmin(ratio))
                t1 = ratio[kdrop]
            else:
                r = u
                z = Hn
                t1, kdrop = _INF, -1

            nz = float(z @ n_p)
            t2 = (b[p] - float(n_p @ x)) / nz if nz > 1e-14 else _INF
            t = t1 if t1 < t2 else t2
            if t == _INF:
                return QPResult(x, 1, it, active)  # infeasible

            if t2 != _INF:
                x = x + t * z
            if active:
                u = u - t * r
            u_p += t

            if t2 != _INF and t == t2:
                active = active + [p]
                u = np.append(u, u_p)
                break
            # Drop the blocking constraint and retry with the same p.
            del active[kdrop]
            u = np.delete(u, kdrop)

        s = A @ x - b
        p = int(np.argmin(s))
        if s[p] >= -tol or p in active:
            return QPResult(x, 0, it, active)
