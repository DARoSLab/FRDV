"""numba kernels must reproduce the numpy reference implementation."""
import numpy as np
import pytest
from frdv import IKConfig, IKProblem, solve_qp
from frdv import _jit

pytestmark = pytest.mark.skipif(not _jit.AVAILABLE, reason="numba not installed")


@pytest.mark.parametrize("seed", range(20))
def test_qp_backends_agree(seed):
    rng = np.random.default_rng(seed)
    n, m, k = 20, 12, 14
    A = rng.standard_normal((n, n))
    H = A @ A.T + 0.1 * np.eye(n)
    g = rng.standard_normal(n) * 5
    C = rng.standard_normal((m, n))
    d = C @ (rng.standard_normal(n) * 0.1) - rng.uniform(0, 0.3, m)
    bidx = rng.choice(n, k, replace=False).astype(np.int64)
    lb, ub = -rng.uniform(0.05, 0.3, k), rng.uniform(0.05, 0.3, k)
    a = solve_qp(H, g, C, d, bidx, lb, ub, use_jit=False)
    b = solve_qp(H, g, C, d, bidx, lb, ub, use_jit=True)
    assert a.solved and b.solved
    assert np.allclose(a.x, b.x, atol=1e-8)
    # warm start from the solution's own active set must give the same answer
    c = solve_qp(H, g, C, d, bidx, lb, ub, use_jit=True, active0=b.active)
    assert np.allclose(c.x, b.x, atol=1e-8)


def test_fused_linearization_matches_mujoco(ik_cfg):
    from dataclasses import replace
    pn = IKProblem(replace(ik_cfg, use_jit=False, constraint_activation=None))
    pj = IKProblem(replace(ik_cfg, use_jit=True))
    rng = np.random.default_rng(3)
    q = pn.integrate(pn.q_nominal, rng.standard_normal(pn.nv) * 0.3)
    tg, qp = rng.standard_normal((pn.S, 3)), pn.integrate(q, rng.standard_normal(pn.nv) * 0.05)
    pn.set_frame(tg, qp)
    pj.set_frame(tg, qp)
    a, b = pn.linearize(q), pj.linearize(q)
    assert np.allclose(a.H, b.H, rtol=1e-10, atol=1e-8)
    assert np.allclose(a.g, b.g, rtol=1e-10, atol=1e-8)
    assert np.allclose(a.C, b.C, atol=1e-10) and np.allclose(a.d, b.d, atol=1e-12)
    assert np.isclose(a.cost, b.cost) and np.isclose(a.viol, b.viol)
    assert np.allclose(pn.merit_terms(q), pj.merit_terms(q))
