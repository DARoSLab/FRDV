from dataclasses import replace

import numpy as np
from frdv import IKProblem


def test_jacobians_match_finite_differences(ik_cfg):
    cfg = replace(ik_cfg, constraint_activation=None,     # keep every row, near the floor or not
                  self_collision_pairs=[("L_foot_link", "R_foot_link"), ("L_shank_link", "R_shank_link")])
    p = IKProblem(cfg)
    rng = np.random.default_rng(0)
    q = p.integrate(p.q_nominal, rng.standard_normal(p.nv) * 0.2)
    p.set_frame(rng.standard_normal((p.S, 3)), p.integrate(q, rng.standard_normal(p.nv) * 0.05))
    lin = p.linearize(q)

    # Gradient of the cost.
    eps = 1e-6
    fd = np.array([(p.merit_terms(p.integrate(q, eps * e))[0] - p.merit_terms(p.integrate(q, -eps * e))[0])
                   / (2 * eps) for e in np.eye(p.nv)])
    # Smoothing uses the GN identity Jacobian for the rotation part; allow small mismatch there.
    assert np.allclose(lin.g, fd, rtol=1e-3, atol=1e-2 * max(1, np.abs(fd).max()) * 1e-2)

    # Constraint Jacobian (C dv >= d with d = -c(q)).
    def cons(qq):
        return -p.linearize(qq).d
    fdC = np.stack([(cons(p.integrate(q, eps * e)) - cons(p.integrate(q, -eps * e))) / (2 * eps)
                    for e in np.eye(p.nv)], axis=1)
    assert np.allclose(lin.C, fdC, atol=1e-5)
