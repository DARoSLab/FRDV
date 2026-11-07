import numpy as np
from frdv import IKProblem, ParallelRetargeter, Retargeter, chunk_bounds, handoff


def _targets(cfg, T=240, seed=0):
    """Synthetic but kinematically valid targets: FK of a smooth random joint trajectory."""
    p = IKProblem(cfg)
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)[:, None]
    phase = rng.uniform(0, 2 * np.pi, p.nv)
    amp = np.r_[0.3, 0.3, 0.0, 0.0, 0.0, 1.0, np.full(p.nv - 6, 0.3)]
    out = np.empty((T, p.S, 3))
    for i in range(T):
        dv = amp * np.sin(2 * np.pi * 1.5 * t[i] + phase)
        dv[5] = 1.0 * t[i, 0]  # turn
        q = p.integrate(p.q_nominal, dv)
        q[7:] = np.clip(q[7:], p.kin.q_min, p.kin.q_max)
        p.kin.forward(q, with_jac=False)
        out[i] = p.kin.positions(p.pts)[: p.S]
    return out


def test_chunk_bounds_cover():
    b = chunk_bounds(1000, 7)
    assert b[0][0] == 0 and b[-1][1] == 1000
    assert all(x[1] == y[0] for x, y in zip(b, b[1:]))


def test_handoff_reproduces_sequential(ik_cfg):
    targets = _targets(ik_cfg)
    r = Retargeter(ik_cfg)
    seq = r.run(targets)

    # Emulate the parallel phase in-process with deliberately short burn-in.
    bounds = chunk_bounds(len(targets), 4)
    q = np.empty_like(seq)
    for k, (s, e) in enumerate(bounds):
        a = max(0, s - 5) if k else 0
        q[s:e] = r.run(targets[a:e])[s - a:]
    gap_before = max(np.abs(r.problem.difference(seq[s], q[s])).max() for s, _ in bounds[1:])
    handoff(r, targets, q, bounds, seam_tol=1e-6)
    err = max(np.abs(r.problem.difference(a, b)).max() for a, b in zip(seq, q))
    assert gap_before > 1e-4          # the seams really were inconsistent
    assert err < 1e-4                 # ...and the handoff made it match the sequential solve


def test_parallel_pool(ik_cfg):
    targets = _targets(ik_cfg, T=300, seed=1)
    seq = Retargeter(ik_cfg).run(targets)
    with ParallelRetargeter(ik_cfg, n_workers=3, min_chunk=50, seam_tol=1e-6) as pr:
        q, rep = pr.run(targets, return_report=True)
    assert len(rep.chunk_bounds) == 3
    assert np.abs(q - seq).max() < 1e-3


def test_run_many_matches_sequential(ik_cfg):
    clips = [_targets(ik_cfg, T=80 + 20 * i, seed=i) for i in range(4)]
    r = Retargeter(ik_cfg)
    with ParallelRetargeter(ik_cfg, n_workers=2) as pr:
        out = pr.run_many(clips)
    for c, q in zip(clips, out):
        assert np.abs(q - r.run(c)).max() < 1e-9


def test_continuous_quaternions():
    from frdv import continuous_quaternions
    q = np.zeros((5, 7))
    q[:, 3] = 1.0
    q[2:4, 3:7] *= -1                    # frames 2-3 in the other hemisphere
    assert continuous_quaternions(q) == 2
    assert np.all(q[:, 3] > 0)
