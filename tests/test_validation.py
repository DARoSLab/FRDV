"""Invalid clip arrays must fail before a compiled call or output write."""
from types import SimpleNamespace

import numpy as np
import pytest

from frdv import Retargeter, _jit, _native
from frdv.validation import ValidationError


@pytest.fixture
def solvers(ik_cfg):
    retargeter = Retargeter(ik_cfg)
    choices = [('python', lambda x, **kw: retargeter.run(x, **kw))]
    if _native.supported(retargeter.problem):
        solver = _native.Solver(retargeter.problem)
        choices.append(('native', lambda x, **kw: solver.run(ik_cfg, x, **kw)))
    if _jit.AVAILABLE:
        from frdv import _fused
        fused = _fused.Fused(retargeter.problem)
        choices.append(('numba', lambda x, **kw: fused.run(ik_cfg, x, **kw)))
    return retargeter, choices


def test_wrong_point_count_stops_before_solving(solvers):
    retargeter, choices = solvers
    targets = np.zeros((2, retargeter.problem.S - 1, 3))
    for name, run in choices:
        with pytest.raises(ValidationError, match='shape.*keypoint mapping'):
            run(targets)


@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_nonfinite_targets_stop(solvers, value):
    retargeter, choices = solvers
    targets = np.zeros((2, retargeter.problem.S, 3))
    targets[1, 0, 0] = value
    for name, run in choices:
        with pytest.raises(ValidationError, match='NaN or infinity'):
            run(targets)


def test_invalid_starting_pose_stops(solvers):
    retargeter, choices = solvers
    targets = np.zeros((2, retargeter.problem.S, 3))
    for name, run in choices:
        with pytest.raises(ValidationError, match='Starting pose'):
            run(targets, q_prev=np.zeros(3))


def test_invalid_output_is_not_written(tmp_path):
    from frdv.cli.retarget import _save
    cfg = {'data':str(tmp_path/'source.npz'),'robot':{'name':'g1'},'dataset':{'name':'amass'}}
    tg = SimpleNamespace(X=np.zeros((2,1,3)))
    q = np.zeros((2,7)); q[:,3] = 1; q[1,0] = np.nan
    with pytest.raises(ValidationError, match='Output trajectory contains NaN'):
        _save(str(tmp_path/'source.npz'),str(tmp_path/'out'),tg,q,cfg,SimpleNamespace(),[])
    assert not (tmp_path/'out').exists()


def test_changed_keypoints_stop_batch_job(monkeypatch):
    from frdv.cli import retarget
    monkeypatch.setattr(retarget,'_W',dict(cfg={},height=1.,keypoints=['pelvis','left_ankle']))
    monkeypatch.setattr(retarget,'prepare',lambda *args:SimpleNamespace(keypoints=['pelvis']))
    with pytest.raises(ValidationError,match='bad.bvh.*keypoints differ') as error:
        retarget._job('bad.bvh')
    assert str(error.value).count('bad.bvh') == 1


def test_thread_validation_failure_stops_pending_files(monkeypatch,ik_cfg):
    from frdv.cli import retarget
    r = Retargeter(ik_cfg)
    class FakeSolver:
        def __init__(self,problem): self.p0 = np.zeros((problem.S,3))
        def workspace(self): return None
        def run(self,cfg,targets,wsp): return np.tile(r.problem.q_nominal,(len(targets),1))
    calls = []
    def prepare(path,*args):
        calls.append(path)
        raise ValidationError('targets contain NaN')
    monkeypatch.setattr(retarget,'_W',dict(r=r,ik=ik_cfg,keypoints=[]))
    monkeypatch.setattr(_native,'supported',lambda p:True)
    monkeypatch.setattr(_native,'Solver',FakeSolver)
    monkeypatch.setattr(retarget,'prepare',prepare)
    cfg = dict(solver={'despike':{'enabled':False}})
    with pytest.raises(ValidationError,match='bad.bvh.*NaN'):
        retarget._run_threads(['bad.bvh','later.bvh'],cfg,1.,1)
    assert calls == ['bad.bvh']


def test_bad_bvh_frame_time_and_nan_are_rejected(tmp_path):
    from frdv.motion.bvh import load
    from test_motion import BVH
    path = tmp_path/'bad.bvh'
    path.write_text(BVH.replace('Frame Time: 0.0333333','Frame Time: 0'))
    with pytest.raises(ValidationError,match='Frame Time must be finite and positive'):
        load(str(path))
    path.write_text(BVH.replace('5 95 7','nan 95 7'))
    with pytest.raises(ValidationError,match='NaN or infinity'):
        load(str(path))


def test_finite_native_output_with_reported_solver_failure_is_rejected(monkeypatch,ik_cfg):
    r = Retargeter(ik_cfg)
    if not _native.supported(r.problem):
        pytest.skip('native solver unavailable')
    solver = _native.Solver(r.problem)
    targets = np.zeros((2,r.problem.S,3))
    def failed_call(*args):
        np.ctypeslib.as_array(args[-2],shape=(2*r.nq,))[:] = np.tile(r.problem.q_nominal,2)
        np.ctypeslib.as_array(args[-1],shape=(8,))[3] = 2
    monkeypatch.setattr(solver,'_lib',SimpleNamespace(frdv_run_clip=failed_call))
    with pytest.raises(ValidationError,match='QP solver failed 2 time'):
        solver.run(ik_cfg,targets,wsp=SimpleNamespace(ptr=0))


def test_invalid_model_array_lengths_are_rejected(ik_cfg):
    from dataclasses import replace
    with pytest.raises(ValidationError,match='Target weights must contain'):
        Retargeter(replace(ik_cfg,target_weights=[1.]))
    with pytest.raises(ValidationError,match='Starting pose must contain'):
        Retargeter(replace(ik_cfg,q_nominal=np.zeros(3)))
