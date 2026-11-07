"""Gauss-Newton SQP on a manifold with an L1-merit backtracking line search.

The problem object supplies:

    linearize(q) -> Linearization     # GN model + linearised constraints at q
    merit_terms(q) -> (cost, viol)    # cheap: forward kinematics only
    integrate(q, dv) -> q'            # retraction (handles the free-joint quaternion)

Steps live in the tangent space (dim nv), so there is no quaternion-norm
equality constraint and joint limits become plain variable bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qp import solve_qp


@dataclass
class SQPOptions:
    max_iter: int = 1          # GN iterations per call (per frame)
    tol: float = 1e-4          # stop when ||dv||_inf < tol (step is still applied)
    line_search: bool = True
    mu_init: float = 10.0      # L1 merit penalty lower bound
    mu_mult: float = 1.1       # penalty = max(mu_init, mu_mult * max|lambda|)
    c1: float = 1e-4           # Armijo constant
    tau: float = 0.5           # backtracking factor
    max_ls_iter: int = 10
    qp_tol: float = 1e-9


@dataclass
class Linearization:
    H: np.ndarray
    g: np.ndarray
    cost: float
    C: np.ndarray | None = None   # C dv >= d
    d: np.ndarray | None = None
    bidx: np.ndarray | None = None
    lb: np.ndarray | None = None
    ub: np.ndarray | None = None
    viol: float = 0.0
    row_ids: np.ndarray | None = None  # global id of each C row (stable across calls, for warm starts)
    n_rows: int = 0                    # number of global constraint ids


@dataclass
class SQPInfo:
    iters: int = 0
    ls_iters: int = 0
    qp_iters: int = 0
    qp_failures: int = 0
    converged: bool = False
    step_norm: float = 0.0


def sqp_solve(problem, q, opts: SQPOptions, info: SQPInfo | None = None):
    info = info if info is not None else SQPInfo()
    if getattr(problem, "fast", False):        # fused compiled path (IKProblem.solve_fast)
        return problem.solve_fast(q, opts, info)
    for _ in range(opts.max_iter):
        info.iters += 1
        lin = problem.linearize(q)
        active0 = _to_local(getattr(problem, "qp_active", None), lin)
        res = solve_qp(lin.H, lin.g, lin.C, lin.d, lin.bidx, lin.lb, lin.ub, tol=opts.qp_tol,
                       active0=active0)
        problem.qp_active = _to_global(res.active, lin)  # warm start for the next solve
        info.qp_iters += res.iters
        if not res.solved:
            info.qp_failures += 1
        dv = res.x
        step = float(np.max(np.abs(dv))) if dv.size else 0.0
        info.step_norm = step

        alpha = 1.0
        if opts.line_search:
            mu = opts.mu_init
            phi0 = lin.cost + mu * lin.viol
            dphi = float(lin.g @ dv) - mu * lin.viol
            if dphi < 0.0:
                for _ in range(opts.max_ls_iter):
                    c, v = problem.merit_terms(problem.integrate(q, alpha * dv))
                    if c + mu * v <= phi0 + opts.c1 * alpha * dphi:
                        break
                    alpha *= opts.tau
                    info.ls_iters += 1
                else:
                    alpha = 1.0  # line search failed: take the full step
        q = problem.integrate(q, alpha * dv)
        if step < opts.tol:
            info.converged = True
            break
    return q, info


# Warm-start bookkeeping. QP rows are stacked as [C ; +bounds ; -bounds]; C rows may be a
# varying subset of the problem's constraints, so active sets are stored as global ids:
#   C row i -> row_ids[i];  bound row j -> n_rows + j.
def _to_global(active, lin):
    if not active or lin.row_ids is None:
        return active
    m = len(lin.row_ids)
    ids = lin.row_ids
    return [int(ids[a]) if a < m else lin.n_rows + (a - m) for a in active]


def _to_local(active, lin):
    if not active or lin.row_ids is None:
        return active
    m = len(lin.row_ids)
    a = np.asarray(active)
    out = []
    bnd = a >= lin.n_rows
    if bnd.any():
        out = (m + a[bnd] - lin.n_rows).tolist()
    g = a[~bnd]
    if g.size:
        i = np.searchsorted(lin.row_ids, g)   # row_ids are sorted (np.flatnonzero)
        i = np.minimum(i, m - 1)
        out = i[lin.row_ids[i] == g].tolist() + out
    return out
