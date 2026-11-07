"""Native (C) build of the fused clip solver and the motion helpers.

``_native.c`` holds the same kernels as ``_fused.py`` (numba) and the compiled BVH / target
helpers of ``_jit.py``. It is compiled once with the system C compiler (``cc``; on macOS the Xcode
command line tools) into ``~/.cache/frdv/native/`` and loaded with ctypes. ctypes releases the GIL
during a call, so whole clips run on plain threads, and nothing here imports numba (which saves
~0.5 s of start-up: its import, type registries and kernel cache).

``available()`` is False when there is no compiler (or ``FRDV_NO_NATIVE=1``); callers then fall
back to the numba kernels, then to numpy.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np

from .validation import ValidationError, pose_array, qp_succeeded, targets_array, trajectory_array

_SRC = Path(__file__).with_name("_native.c")
_LIB = None
_LOCK = threading.Lock()
_FAILED = False

i64 = C.c_int64
dbl = C.c_double
P_I = C.POINTER(C.c_int64)
P_D = C.POINTER(C.c_double)


def _flags():
    flags = ["-O3", "-ffast-math", "-fno-finite-math-only", "-fPIC", "-shared"]
    if sys.platform == "darwin":
        flags += ["-mcpu=native"] if os.uname().machine == "arm64" else ["-march=native"]
    else:
        flags += ["-march=native"]
    return flags


def _cache_dir() -> Path:
    base = os.environ.get("FRDV_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "frdv")
    return Path(base) / "native"


def _build() -> Path:
    src = _SRC.read_bytes()
    cc = os.environ.get("CC", "cc")
    key = hashlib.sha256(src + " ".join([cc] + _flags()).encode() + sys.platform.encode()).hexdigest()[:16]
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    out = _cache_dir() / f"libfrdv_{key}{ext}"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=ext, dir=out.parent)
    os.close(fd)
    try:
        subprocess.run([cc, *_flags(), "-o", tmp, str(_SRC), "-lm"], check=True, capture_output=True)
        os.replace(tmp, out)   # atomic: concurrent builders are harmless
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return out


REASON = ""   # why the native solver is unavailable (empty when it is available or untried)


def _bundled():
    """The library prebuilt into the wheel (setup.py: frdv._libfrdv), if present."""
    import importlib.machinery
    for suf in importlib.machinery.EXTENSION_SUFFIXES:
        p = Path(__file__).with_name("_libfrdv" + suf)
        if p.exists():
            return p
    return None


def _reason(e) -> str:
    if isinstance(e, FileNotFoundError):
        return f"no C compiler found ('{os.environ.get('CC', 'cc')}')"
    if isinstance(e, subprocess.CalledProcessError):
        err = (e.stderr or b"").decode(errors="replace").strip().splitlines()
        return "C build failed" + (f": {err[-1]}" if err else "")
    return f"{type(e).__name__}: {e}"


def lib():
    """The loaded library, or None (no compiler / disabled / build failed)."""
    global _LIB, _FAILED, REASON
    if _LIB is not None or _FAILED:
        return _LIB
    with _LOCK:
        if _LIB is not None or _FAILED:
            return _LIB
        if os.environ.get("FRDV_NO_NATIVE", "0") in ("1", "true", "True"):
            REASON = "disabled by FRDV_NO_NATIVE"
            _FAILED = True
            return None
        L = None
        bundled = _bundled()
        if bundled is not None and os.environ.get("FRDV_NATIVE_BUILD", "0") not in ("1", "true", "True"):
            try:
                L = C.CDLL(str(bundled))
            except OSError:
                L = None
        if L is None:
            try:
                L = C.CDLL(str(_build()))
            except Exception as e:  # no compiler, unsupported platform, ...
                REASON = _reason(e)
                _FAILED = True
                return None
        L.frdv_work_new.restype = C.c_void_p
        L.frdv_work_new.argtypes = [C.c_void_p]
        L.frdv_work_free.argtypes = [C.c_void_p]
        L.frdv_work_nact.restype = i64
        L.frdv_work_nact.argtypes = [C.c_void_p]
        L.frdv_run_clip.restype = None
        L.frdv_run_clip.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
                                    i64, dbl, i64, dbl, P_D, i64, P_D, i64, P_D, P_I]
        L.frdv_despike.restype = None
        L.frdv_despike.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, P_D, i64, P_D, dbl, dbl, i64,
                                   dbl, i64, P_I, P_D, P_I]
        L.frdv_parse_decimals.restype = i64
        L.frdv_parse_decimals.argtypes = [C.c_char_p, i64, i64, P_D, i64]
        L.frdv_bvh_fk_sel.restype = None
        L.frdv_bvh_fk_sel.argtypes = [P_I, P_D, P_I, P_I, P_I, P_I, P_D, i64, i64, i64, P_I, i64, P_D, P_D]
        L.frdv_rotate_points.restype = None
        L.frdv_rotate_points.argtypes = [P_D, i64, P_D, dbl, P_D]
        L.frdv_limb_plan.restype = None
        L.frdv_limb_plan.argtypes = [P_D, i64, i64, P_I, P_I, P_I, P_D, P_I, P_D, i64, P_D, P_D]
        L.frdv_bvh_fk_markers.restype = None
        L.frdv_bvh_fk_markers.argtypes = [P_I, P_D, P_I, P_I, P_I, P_I, P_D, i64, i64, i64, P_I, P_D, i64, P_D]
        L.frdv_bone_medians.restype = None
        L.frdv_bone_medians.argtypes = [P_D, i64, i64, P_I, P_I, i64, P_D]
        L.frdv_torso_frames.restype = None
        L.frdv_torso_frames.argtypes = [P_D, i64, i64, i64, i64, i64, i64, i64, i64, P_D, P_D]
        L.frdv_finish_targets.restype = None
        L.frdv_finish_targets.argtypes = [P_D, i64, i64, P_D, P_I, i64, P_D, dbl, P_D, P_D, P_D]
        _LIB = L
        return _LIB


def available() -> bool:
    return lib() is not None


def _pd(a):
    return a.ctypes.data_as(P_D)


def _pi(a):
    return a.ctypes.data_as(P_I)


# --------------------------------------------------------------------------- motion helpers
def parse_decimals(blob: bytes, start: int, expected: int):
    """np.fromstring(blob[start:], sep=" ") (bit-identical) or None if the fast reader can't."""
    L = lib()
    if L is None:
        return None
    cap = expected if expected > 0 else (len(blob) - start) // 2 + 1
    out = np.empty(cap)
    n = L.frdv_parse_decimals(blob, len(blob), start, _pd(out), cap)
    if n < 0:
        return parse_decimals(blob, start, -1) if expected > 0 else None
    return out[:n]


def bvh_fk_sel(parents, offsets, chan_start, chan_n, kind, axis, data, slot, pos, rot):
    L = lib()
    a = [np.ascontiguousarray(x, dtype=np.int64) for x in (parents, chan_start, chan_n, kind, axis, slot)]
    offsets = np.ascontiguousarray(offsets, dtype=float)
    data = np.ascontiguousarray(data, dtype=float)
    L.frdv_bvh_fk_sel(_pi(a[0]), _pd(offsets), _pi(a[1]), _pi(a[2]), _pi(a[3]), _pi(a[4]), _pd(data),
                      data.shape[0], len(a[0]), data.shape[1], _pi(a[5]), rot.shape[1], _pd(pos), _pd(rot))


def bvh_fk_markers(parents, offsets, chan_start, chan_n, kind, axis, data, mk_parent, mk_off):
    """BVH FK plus rigid markers: (T, J + n_markers, 3) positions, in one GIL-free call."""
    a = [np.ascontiguousarray(x, dtype=np.int64) for x in (parents, chan_start, chan_n, kind, axis, mk_parent)]
    offsets = np.ascontiguousarray(offsets, dtype=float)
    mk_off = np.ascontiguousarray(mk_off, dtype=float).reshape(-1, 3)
    data = np.ascontiguousarray(data, dtype=float)
    T, J = data.shape[0], len(a[0])
    pos = np.empty((T, J + len(a[5]), 3))
    lib().frdv_bvh_fk_markers(_pi(a[0]), _pd(offsets), _pi(a[1]), _pi(a[2]), _pi(a[3]), _pi(a[4]), _pd(data), T, J,
                              data.shape[1], _pi(a[5]), _pd(mk_off), len(a[5]), _pd(pos))
    return pos


def bone_medians(P, pairs):
    """[np.median(np.linalg.norm(P[:, a] - P[:, b], axis=-1)) for a, b in pairs] (bit-identical)."""
    P = np.ascontiguousarray(P, dtype=float)
    ia = np.array([a for a, _ in pairs], np.int64)
    ib = np.array([b for _, b in pairs], np.int64)
    out = np.empty(len(pairs))
    lib().frdv_bone_medians(_pd(P), P.shape[0], P.shape[1], _pi(ia), _pi(ib), len(pairs), _pd(out))
    return out


def torso_frames(P, la, lb, u1, u2, v1, v2, Fr):
    """motion.limb_rescale's frame(...) @ Fr.T for every frame (u1 < 0: up = +z)."""
    P = np.ascontiguousarray(P, dtype=float)
    Fr = np.ascontiguousarray(Fr, dtype=float)
    out = np.empty((P.shape[0], 3, 3))
    lib().frdv_torso_frames(_pd(P), P.shape[0], P.shape[1], la, lb, u1, u2, v1, v2, _pd(Fr), _pd(out))
    return out


def rotate_points(P, R, s, out):
    R = np.ascontiguousarray(R, dtype=float)
    lib().frdv_rotate_points(_pd(P), P.shape[0], _pd(R), float(s), _pd(out))


def limb_plan(P, joint, anchor, kind, ratio, rot_id, off, R, out):
    T, J, _ = P.shape
    lib().frdv_limb_plan(_pd(P), T, J, _pi(joint), _pi(anchor), _pi(kind), _pd(ratio), _pi(rot_id), _pd(off),
                         len(joint), _pd(R), _pd(out))


def finish_targets(P, fac, idx, shift, dz, offset, extra, X):
    T, J, _ = P.shape
    f = [np.ascontiguousarray(x, dtype=float) for x in (fac, shift, offset, extra)]
    idx = np.ascontiguousarray(idx, dtype=np.int64)
    lib().frdv_finish_targets(_pd(P), T, J, _pd(f[0]), _pi(idx), len(idx), _pd(f[1]), float(dz), _pd(f[2]),
                              _pd(f[3]), _pd(X))


# --------------------------------------------------------------------------- solver
class _Model(C.Structure):
    _fields_ = ([(n, i64) for n in ("nbody", "nv", "nq", "nj", "S", "G", "Lc", "ngb")]
                + [(n, P_I) for n in ("bparent", "bjnt0", "bjntn", "jtype", "jqadr", "jdadr", "dparent", "dbody",
                                      "bchain", "bchain_len", "pb", "anc", "nanc", "gbody", "gofs", "gpt", "gpb",
                                      "bdof0", "bdofn")]
                + [(n, P_D) for n in ("bpos", "bquat", "jaxis", "jpos", "qpos0", "pl", "grad", "wpt", "q_nom", "qmin",
                                      "qmax", "reach", "lmax", "p0")]
                + [("h", dbl), ("bmat", P_D), ("bflag", P_I), ("maxr", i64), ("arng", P_I), ("narng", P_I),
                   ("brng", P_I), ("nbrng", P_I)])


class _Prm(C.Structure):
    _fields_ = [("ws", P_D), ("D", P_D), ("wreg", P_D), ("cap", dbl), ("bidx", P_I), ("nb", i64)]


class _Opt(C.Structure):
    _fields_ = [("max_iter", i64), ("tol", dbl), ("line_search", i64), ("mu_init", dbl), ("c1", dbl), ("tau", dbl),
                ("max_ls", i64), ("qp_tol", dbl), ("act_lim", dbl)]


def supported(problem) -> bool:
    """Same coverage as the numba fused path: free root + hinge/slide joints, no self-collision."""
    m = problem.kin.model
    if problem.pairs or m.jnt_type[0] != 0 or any(t not in (2, 3) for t in m.jnt_type[1:]):
        return False
    if any(m.body_jntnum[b] > 1 and m.jnt_type[m.body_jntadr[b]] == 0 for b in range(m.nbody)):
        return False
    return available()


class Solver:
    """Native fused solver for one IKProblem. ``workspace()`` gives per-thread state (it also
    carries the QP warm start between calls); ``run`` / ``despike`` release the GIL."""

    def __init__(self, problem):
        import mujoco
        from .retarget import _dof_chain
        p = problem
        m = p.kin.model
        self.problem = p
        nv, nq, nb = m.nv, m.nq, m.nbody
        self.nv, self.nq = nv, nq
        I = lambda a: np.ascontiguousarray(a, dtype=np.int64)  # noqa: E731,E741
        D = lambda a: np.ascontiguousarray(a, dtype=np.float64)  # noqa: E731
        chains = [_dof_chain(m, b) if b else [] for b in range(nb)]
        Lc = max(1, max(len(c) for c in chains))
        bchain = np.zeros((nb, Lc), np.int64)
        for b, ch in enumerate(chains):
            bchain[b, : len(ch)] = ch
        anc = np.zeros((nv, nv), np.int64)
        nanc = []
        for k in range(nv):
            a, i = [], m.dof_parentid[k]
            while i >= 0:
                a.append(i)
                i = m.dof_parentid[i]
            anc[k, : len(a)] = a
            nanc.append(len(a))
        grad = np.zeros(p.n_ground)
        grad[p.F:] = p.cap_radius
        gb = p._pb[p.S:]
        gbody = np.unique(gb)
        gpt = np.concatenate([np.flatnonzero(gb == b) for b in gbody]) if len(gb) else np.zeros(0, np.int64)
        gofs = np.r_[0, np.cumsum([np.sum(gb == b) for b in gbody])]
        ln = np.linalg.norm(p._pl[p.S:], axis=1) if len(gb) else np.zeros(0)
        reach = np.array([np.max(ln[gb == b] + grad[gb == b]) for b in gbody]) * (1 + 1e-9) + 1e-12
        lmax = np.array([np.max(ln[gb == b]) for b in gbody]) * (1 + 1e-9) + 1e-12
        d = mujoco.MjData(m)
        d.qpos[:] = p.q_nominal
        mujoco.mj_kinematics(m, d)
        R = d.xmat[p._pb[: p.S]].reshape(-1, 3, 3)
        self.p0 = d.xpos[p._pb[: p.S]] + np.einsum("bij,bj->bi", R, p._pl[: p.S])
        arr = dict(
            bparent=I(m.body_parentid), bjnt0=I(m.body_jntadr), bjntn=I(m.body_jntnum), jtype=I(m.jnt_type),
            jqadr=I(m.jnt_qposadr), jdadr=I(m.jnt_dofadr), dparent=I(m.dof_parentid), dbody=I(m.dof_bodyid),
            bchain=bchain, bchain_len=I([len(c) for c in chains]), pb=I(p._pb), anc=anc, nanc=I(nanc),
            gbody=I(gbody), gofs=I(gofs), gpt=I(gpt), gpb=I(p._pb[p.S:]), bdof0=I(m.body_dofadr.clip(0)),
            bdofn=I(m.body_dofnum), bpos=D(m.body_pos), bquat=D(m.body_quat), jaxis=D(m.jnt_axis),
            jpos=D(m.jnt_pos), qpos0=D(m.qpos0), pl=D(p._pl), grad=D(grad), wpt=D(p.w_tgt), q_nom=D(p.q_nominal),
            qmin=D(p.kin.q_min), qmax=D(p.kin.q_max), reach=D(reach), lmax=D(lmax), p0=D(self.p0))
        # FK matrix path: every non-root body has no joint or a single hinge at its origin
        bmat = np.zeros((nb, 9))
        for b in range(nb):
            mujoco.mju_quat2Mat(bmat[b], m.body_quat[b])
        simple = all(b == 0 or (m.body_jntnum[b] == 1 and m.jnt_type[m.body_jntadr[b]] == 0)
                     or m.body_jntnum[b] == 0
                     or (m.body_jntnum[b] == 1 and m.jnt_type[m.body_jntadr[b]] == 3
                         and not np.any(m.jnt_pos[m.body_jntadr[b]]))
                     for b in range(nb))
        ident = np.all(m.body_quat == [1.0, 0.0, 0.0, 0.0], axis=1)
        bflag = ident.astype(np.int64) | (2 * (simple & (m.body_jntnum == 0) | simple & (m.jnt_type[m.body_jntadr.clip(0)] == 3) & (m.body_jntnum == 1))).astype(np.int64)
        bflag[0] = 0
        arr["bmat"], arr["bflag"] = bmat, bflag
        # runs of consecutive indices in each dof's ancestor list and each body's dof chain
        def runs(lst):
            out = []
            for i in lst:                       # descending
                if out and out[-1][0] == i + 1:
                    out[-1][0] = i
                else:
                    out.append([i, i])
            return out
        ar = [runs([int(a) for a in anc[k, : nanc[k]]]) for k in range(nv)]
        br = [runs(ch) for ch in chains]
        maxr = max(1, max(len(r) for r in ar + br))
        arng = np.zeros((nv, maxr, 2), np.int64)
        brng = np.zeros((nb, maxr, 2), np.int64)
        for k, r in enumerate(ar):
            if r:
                arng[k, : len(r)] = r
        for b, r in enumerate(br):
            if r:
                brng[b, : len(r)] = r
        arr.update(arng=arng, narng=I([len(r) for r in ar]), brng=brng, nbrng=I([len(r) for r in br]))
        self._maxr = maxr
        self._keep = arr      # the struct holds raw pointers into these
        st = _Model(nbody=nb, nv=nv, nq=nq, nj=nv - 6, S=p.S, G=p.n_ground, Lc=Lc, ngb=len(gbody),
                    h=float(p.cfg.ground_height), maxr=self._maxr)
        for k, v in arr.items():
            setattr(st, k, _pi(v) if v.dtype == np.int64 else _pd(v))
        self._model = st
        self.bidx = I(p.bidx)
        self._lib = lib()

    def workspace(self):
        return _Work(self._lib, self._model)

    # IKProblem.__init__'s weights for a (possibly modified) config
    def params(self, cfg, max_joint_step=None):
        nv = self.nv
        reg = np.broadcast_to(np.asarray(cfg.joint_reg_weights, float), (nv - 6,)).copy()
        ws = np.full(nv, float(cfg.smoothing_weight) + float(cfg.accel_weight))
        ws[3:6] *= cfg.rot_smoothing_scale
        Dg = ws.copy()
        Dg[6:] += reg
        Dg += cfg.hessian_reg
        keep = (ws, Dg, reg)
        prm = _Prm(ws=_pd(ws), D=_pd(Dg), wreg=_pd(reg), cap=-1.0 if max_joint_step is None else float(max_joint_step),
                   bidx=_pi(self.bidx), nb=len(self.bidx))
        prm._keep = keep
        alpha = cfg.accel_weight / max(cfg.smoothing_weight + cfg.accel_weight, 1e-12)
        return prm, alpha

    @staticmethod
    def options(o, act_lim):
        return _Opt(max_iter=int(o.max_iter), tol=float(o.tol), line_search=int(bool(o.line_search)),
                    mu_init=float(o.mu_init), c1=float(o.c1), tau=float(o.tau), max_ls=int(o.max_ls_iter),
                    qp_tol=float(o.qp_tol), act_lim=-1.0 if act_lim is None else float(act_lim))

    def run(self, cfg, targets, wsp=None, q_prev=None, cold_start=True, info=None):
        """Retargeter.run(targets, q_prev, cold_start) in one GIL-free call."""
        targets = targets_array(targets, self.problem.S)
        if q_prev is not None:
            q_prev = pose_array(q_prev, self.nq)
        wsp = wsp if wsp is not None else self.workspace()
        prm, alpha = self.params(cfg, cfg.max_joint_step)
        act = cfg.constraint_activation
        o = self.options(cfg.sqp, act)
        ob = self.options(replace(cfg.sqp, max_iter=cfg.boost_iters), act)
        ow = self.options(replace(cfg.sqp, max_iter=1), act)
        X = np.ascontiguousarray(targets, dtype=float).reshape(len(targets), -1, 3)
        out = np.empty((len(X), self.nq))
        q0 = np.zeros(self.nq) if q_prev is None else np.ascontiguousarray(q_prev, dtype=float)
        stats = np.zeros(8, np.int64)
        A = C.addressof
        self._lib.frdv_run_clip(A(self._model), wsp.ptr, A(prm), A(o), A(ob), A(ow),
                                int(cfg.warmup_iters), float(cfg.warmup_tol),
                                int(cfg.boost_frames) if cold_start else 0, float(alpha), _pd(X), len(X), _pd(q0),
                                int(q_prev is None), _pd(out), _pi(stats))
        if info is not None:
            info.iters += int(stats[0])
            info.ls_iters += int(stats[1])
            info.qp_iters += int(stats[2])
            info.qp_failures += int(stats[3])
        qp_succeeded(int(stats[3]))
        return trajectory_array(out, len(X), self.nq)

    def despike(self, cfg, targets, q, wsp=None, threshold=0.25, ratio=4.0, half_window=5, sigma=1.5,
                pull_weight=200.0, iters=2, passes=3):
        """smooth.despike (same arguments, q modified in place) in one GIL-free call."""
        from .smooth import DespikeReport
        from .sqp import SQPOptions
        targets = targets_array(targets, self.problem.S)
        trajectory_array(q, len(targets), self.nq)
        if not isinstance(q, np.ndarray) or not q.flags.c_contiguous or q.dtype != np.float64:
            raise ValidationError('Despike requires a C-contiguous float64 qpos array for in-place updates')
        wsp = wsp if wsp is not None else self.workspace()
        n = max(passes, 1)
        prms = (_Prm * n)()
        keep = []
        for k in range(n):
            pk, _ = self.params(replace(cfg, smoothing_weight=pull_weight * 4 ** k, accel_weight=0.0,
                                        rot_smoothing_scale=1.0), None)
            prms[k] = pk
            keep.append(pk)
        opt = self.options(SQPOptions(max_iter=iters, line_search=True), cfg.constraint_activation)
        X = np.ascontiguousarray(targets, dtype=float).reshape(len(targets), -1, 3)
        assert q.flags.c_contiguous and q.dtype == np.float64
        irep, frep = np.zeros(4, np.int64), np.zeros(2)
        wins = np.zeros((max(len(q), 1), 2), np.int64)
        self._lib.frdv_despike(C.addressof(self._model), wsp.ptr, C.addressof(prms), C.addressof(opt), _pd(X),
                               len(X), _pd(q),
                               float(threshold), float(ratio), int(half_window), float(sigma), int(passes),
                               _pi(irep), _pd(frep), _pi(wins))
        trajectory_array(q, len(X), self.nq)
        return DespikeReport(spikes_before=int(irep[0]), spikes_after=int(irep[1]),
                             windows=[tuple(map(int, w)) for w in wins[: irep[2]]], worst_before=float(frep[0]),
                             worst_after=float(frep[1]), reverted=int(irep[3]))


class _Work:
    def __init__(self, L, model):
        self._lib = L
        self.ptr = C.c_void_p(L.frdv_work_new(C.addressof(model)))

    def __del__(self):
        try:
            self._lib.frdv_work_free(self.ptr)
        except Exception:
            pass
