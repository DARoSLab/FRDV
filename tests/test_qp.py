import numpy as np
import pytest
from frdv import solve_qp

scipy_opt = pytest.importorskip("scipy.optimize")


def _random_qp(rng, n, m, k):
    A = rng.standard_normal((n, n))
    H = A @ A.T + 0.1 * np.eye(n)
    g = rng.standard_normal(n) * 5
    C = rng.standard_normal((m, n))
    x_feas = rng.standard_normal(n) * 0.1
    d = C @ x_feas - rng.uniform(0, 0.5, m)       # feasible by construction, many active
    bidx = rng.choice(n, k, replace=False)
    lb = np.minimum(x_feas[bidx], -rng.uniform(0, 0.3, k))
    ub = np.maximum(x_feas[bidx], rng.uniform(0, 0.3, k))
    return H, g, C, d, bidx, lb, ub


@pytest.mark.parametrize("seed", range(30))
def test_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 30))
    H, g, C, d, bidx, lb, ub = _random_qp(rng, n, int(rng.integers(0, 15)), int(rng.integers(0, n)))
    res = solve_qp(H, g, C, d, bidx, lb, ub)
    assert res.solved
    x = res.x
    assert np.all(C @ x - d >= -1e-7)
    assert np.all(x[bidx] >= lb - 1e-7) and np.all(x[bidx] <= ub + 1e-7)

    bounds = [(None, None)] * n
    for i, j in enumerate(bidx):
        bounds[j] = (lb[i], ub[i])
    ref = scipy_opt.minimize(lambda z: 0.5 * z @ H @ z + g @ z, np.zeros(n), jac=lambda z: H @ z + g,
                             constraints=[{"type": "ineq", "fun": lambda z: C @ z - d, "jac": lambda z: C}],
                             bounds=bounds, method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
    f = lambda z: 0.5 * z @ H @ z + g @ z
    assert f(x) <= f(ref.x) + 1e-6 * max(1, abs(f(ref.x)))


def test_unconstrained_fast_path():
    H = np.diag([1.0, 2.0])
    res = solve_qp(H, np.array([1.0, -2.0]))
    assert res.iters == 0 and np.allclose(res.x, [-1.0, 1.0])


def test_infeasible_reported():
    H = np.eye(1)
    C = np.array([[1.0], [-1.0]])
    d = np.array([1.0, 0.0])  # x >= 1 and x <= 0
    assert solve_qp(H, np.zeros(1), C, d).status == 1
