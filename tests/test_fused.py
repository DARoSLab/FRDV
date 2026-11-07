"""The whole-clip solvers (native C build and numba build) reproduce Retargeter.run / despike."""
from dataclasses import replace

import numpy as np
import pytest

from frdv import Retargeter, _jit, _native
from frdv.smooth import despike
from test_motion import _walk_targets


def _backends(problem):
    out = []
    if _native.supported(problem):
        out.append(("native", _native.Solver(problem)))
    if _jit.AVAILABLE:
        from frdv import _fused
        if _fused.supported(problem):
            out.append(("numba", _fused.Fused(problem)))
    if not out:
        pytest.skip("neither a C compiler nor numba is available")
    return out


@pytest.mark.parametrize("cap", [None, 0.3])
def test_clip_solver_matches_retargeter(ik_cfg, cap):
    cfg = replace(ik_cfg, accel_weight=20.0, max_joint_step=cap)
    X = _walk_targets(cfg, 240)
    r = Retargeter(cfg)
    ref = r.run(X)
    ref_warm = r.run(X[100:], q_prev=ref[99], cold_start=False)
    for name, s in _backends(r.problem):
        w = s.workspace()
        q = s.run(cfg, X, w)
        assert np.abs(q - ref).max() < 1e-8, name
        q2 = s.run(cfg, X[100:], w, q_prev=ref[99], cold_start=False)
        assert np.abs(q2 - ref_warm).max() < 1e-8, name


def test_clip_despike_matches(ik_cfg):
    cfg = replace(ik_cfg, accel_weight=20.0)
    X = _walk_targets(cfg)
    r = Retargeter(cfg)
    q = r.run(X)
    q[100, 7 + 3] += 0.8          # one-frame snap
    q[150, 7 + 1] -= 0.6
    ref = q.copy()
    rep_ref = despike(r, X, ref)
    assert rep_ref.windows
    for name, s in _backends(r.problem):
        got = q.copy()
        rep = s.despike(cfg, X, got, s.workspace())
        assert (rep.spikes_before, rep.spikes_after, rep.reverted) == \
            (rep_ref.spikes_before, rep_ref.spikes_after, rep_ref.reverted), name
        assert rep.windows == rep_ref.windows, name
        assert np.abs(got - ref).max() < 1e-8, name


def test_native_decimal_parser_is_exact():
    if not _native.available():
        pytest.skip("no C compiler")
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(0, 200, 5000).round(6), rng.normal(0, 1, 1000), [0.0, -0.0, 1e-7, 123456.5]])
    text = " ".join(f"{v:.6f}" for v in vals[:5000]) + "\n" + " ".join(f"{v:.9e}" for v in vals[5000:])
    blob = ("Frame Time: 0.1\n" + text + "\n").encode()
    start = blob.index(b"0.1") + 3
    got = _native.parse_decimals(blob, start, -1)
    ref = np.fromstring(text, sep=" ")
    assert got is not None and np.array_equal(got.view(np.int64), ref.view(np.int64))
    # not handled (caller falls back to numpy): nan, mantissas beyond 2^53
    assert _native.parse_decimals(b" 1.0 nan 2.0", 0, -1) is None
    assert _native.parse_decimals(b" 0.12345678901234567", 0, -1) is None
