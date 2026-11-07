"""frdv-retarget: retarget a motion file or a whole dataset onto a robot.

    frdv-retarget robot=g1 dataset=lafan1 data=/path/to/lafan1
    frdv-retarget robot=g1 dataset=lafan1 data=/path/to/lafan1 workers=8
    frdv-retarget robot=g1 scaling=manual data=/path/to/lafan1 dataset.scale.value=[0.76,0.78,0.74]
    frdv-retarget robot=g1 data=/path/to/lafan1 dataset.augmentation=true  # optional hand/arm markers
    frdv-retarget dataset=amass data=/path/to/AMASS dataset.smplh_model_path=/path/to/smplh
    frdv-retarget dataset=amass subject=amass/accad/Female1Walking_c3d data=/path/to/subject
    frdv-retarget robot=myrobot dataset=lafan1 data=walk.bvh solver.smoothing_weight=20 vis=true
    frdv-retarget --cfg robot=g1 dataset=lafan1            # print the composed config

Outputs ``<out>/<clip>.npz`` (qpos (T, nq) MuJoCo convention, fps, targets, keypoints,
robot_links, joint_names, robot_xml, source) and ``<out>/frdv_config.yaml``.
"""

from __future__ import annotations

import os
import sys
import time

from .common import THREAD_VARS

for _v in THREAD_VARS:  # tiny per-frame problems: single-threaded BLAS everywhere
    os.environ.setdefault(_v, "1")

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402

from .. import motion as M  # noqa: E402
from ..parallel import ParallelRetargeter  # noqa: E402
from ..retarget import Retargeter  # noqa: E402
from ..validation import ValidationError, targets_array, trajectory_array  # noqa: E402
from .common import (calibrated_keypoints, find_files, ik_config, load_calibration, per_clip_setup,  # noqa: E402
                     postprocess, resolve_path, robot_height, robot_points, target_offset, to_dict, validate_robot,
                     validate_subject_input)

_W: dict = {}


def _loader_opts(ds: dict) -> dict:
    opts = {"markers": M.enabled_markers(ds)}
    if ds.get("smplh_model_path") is not None:
        opts["smplh_model_path"] = ds["smplh_model_path"]
    return opts


def _clip_error(path, error):
    message = str(error)
    return ValidationError(message if message.startswith(f"{path}:") else f"{path}: {message}")


def prepare(path: str, cfg: dict, height: float, keep_skeleton=False):
    ds, rb = cfg["dataset"], cfg["robot"]
    try:
        mot = M.load(path, ds["format"], **_loader_opts(ds))
        tg = M.build_targets(mot, ds, height, list((rb.get("keypoints") or {}).keys()),
                             extra_offset=target_offset(rb, ds), keep_skeleton=keep_skeleton,
                             robot_points=robot_points(rb))
    except ValueError as error:
        raise _clip_error(path, error) from error
    if not np.isfinite(tg.fps) or tg.fps <= 0:
        raise ValidationError(f"{path}: frame rate must be finite and positive; got {tg.fps}")
    s, e = int(cfg.get("start") or 0), cfg.get("end")
    tg.X = tg.X[s:e]
    if tg.skeleton is not None:
        tg.skeleton = tg.skeleton[s:e]
    tg.X = targets_array(tg.X, len(tg.keypoints))
    from .diagnostics import warn_measurements
    warn_measurements(tg, ds, height, path, s)
    return tg


def _seg_report(tg) -> str:
    sc = tg.segment_scale
    pick = lambda *ks: np.mean([sc[k] for k in ks if k in sc]) if any(k in sc for k in ks) else float("nan")  # noqa: E731
    return (f"(limb scaling: pelvis/stride {sc['root']:.3f}, thigh {pick('left_knee', 'right_knee'):.3f}, "
            f"shin {pick('left_ankle', 'right_ankle'):.3f}, torso {pick('spine', 'left_shoulder', 'right_shoulder'):.3f}, "
            f"upper arm {pick('left_elbow', 'right_elbow'):.3f}, forearm {pick('left_wrist', 'right_wrist'):.3f})")


def _save(path, out_dir, tg, q, cfg, ik, joint_names):
    from ..utils import continuous_quaternions
    q = trajectory_array(q, len(tg.X), len(ik.q_nominal) if getattr(ik, 'q_nominal', None) is not None else None)
    continuous_quaternions(q)
    data_root = resolve_path(cfg["data"])
    relative = os.path.relpath(path, data_root) if os.path.isdir(data_root) else os.path.basename(path)
    name = os.path.splitext(relative)[0]
    destination = os.path.join(out_dir, f"{name}.npz")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    extra = {}
    if tg.skeleton is not None:
        extra = dict(skeleton=tg.skeleton.astype(np.float32), skeleton_parents=np.array(tg.skeleton_parents))
    subject = cfg.get("subject") or {}
    np.savez(destination, qpos=q, fps=tg.fps, targets=tg.X.astype(np.float32),
             keypoints=np.array(tg.keypoints), source_joints=np.array(tg.source_joints),
             robot_links=np.array([t[0] if isinstance(t, (tuple, list)) else t for t in ik.targets]),
             robot_link_offsets=np.array([t[1] if isinstance(t, (tuple, list)) else (0.0, 0.0, 0.0) for t in ik.targets],
                                         dtype=float),
             joint_names=np.array(joint_names), robot_xml=ik.xml_path,
             robot=cfg["robot"]["name"], dataset=cfg["dataset"]["name"], source=path,
             subject=subject.get("name") or "", collection=subject.get("collection") or "", **extra)
    return name


def _joint_names(r: Retargeter):
    import mujoco
    m = r.problem.kin.model
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]


def _init(cfg, height, keypoints, warm=True):
    ik = ik_config(cfg["robot"], cfg["solver"], keypoints, cfg["dataset"], calibration=cfg.get('calibration'))
    r = Retargeter(ik)
    p = r.problem
    if warm:
        p.kin.forward(p.q_nominal, with_jac=False)
        r.run(np.repeat(p.kin.positions(p.pts)[None, : p.S], 2, axis=0))  # load JIT kernels
    _W.update(cfg=cfg, height=height, r=r, ik=ik, joints=_joint_names(r), keypoints=list(keypoints))


def _check_keypoints(tg, expected, path):
    if tg.keypoints != list(expected):
        raise ValidationError(f"{path}: tracked keypoints differ from the first clip: expected {list(expected)}, "
                              f"got {tg.keypoints}. Check the skeleton and dataset keypoint mapping.")


def _job(path):
    t0 = time.perf_counter()
    try:
        tg = prepare(path, _W["cfg"], _W["height"], _W["cfg"].get("save_skeleton", False))
        _check_keypoints(tg, _W['keypoints'], path)
        t1 = time.perf_counter()
        per_clip_setup(_W["r"], _W["cfg"]["solver"], tg.fps)
        q = _W["r"].run(tg.X)
        postprocess(_W["r"], _W["cfg"]["solver"], tg.X, q, tg.fps)
        t2 = time.perf_counter()
        name = _save(path, _W["cfg"]["out"], tg, q, _W["cfg"], _W["ik"], _W["joints"])
        return name, len(q), t1 - t0, t2 - t1, None
    except ValidationError as error:
        raise _clip_error(path, error) from error
    except Exception as e:  # keep the batch going for other file errors
        return os.path.basename(path), 0, 0.0, 0.0, f"{type(e).__name__}: {e}"


def _fused_ok(c, height, keypoints) -> bool:
    """True if the GIL-free whole-clip solver (native C build, else numba) covers this problem."""
    from .. import _jit, _native
    if c.get("threads") is False:
        return False
    native = _native.available()
    if not native and not _jit.AVAILABLE:
        _notice(f"native solver unavailable ({_native.REASON}) and numba is not installed: using the "
                "pure-Python per-frame solver, which is much slower. Install a C compiler (macOS: "
                "xcode-select --install) or numba (pip install 'frdv[fast]').")
        return False
    _init(c, height, keypoints, warm=False)
    p = _W["r"].problem
    if native:
        ok = _native.supported(p)
    else:
        from .. import _fused
        ok = _fused.supported(p)
        if ok:
            _notice(f"native solver unavailable ({_native.REASON}): using the numba build, which is slower. "
                    "Install a C compiler (macOS: xcode-select --install) for the fast path.")
    if not ok:
        _notice("this robot config (e.g. self-collision pairs) is not covered by the whole-clip "
                "solver: using the per-frame solver in worker processes, which is slower.")
    return ok


_NOTICED = False


def _notice(msg):
    global _NOTICED
    if not _NOTICED:
        _NOTICED = True
        print(f"[frdv] note: {msg}", flush=True)


def _run_threads(files, c, height, workers, first=(None, None)):
    """files mode on threads with the GIL-free whole-clip solver (frdv._native, else frdv._fused).
    ``first`` = (path, targets) of a clip that is already loaded."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    from .. import _native
    r, ik = _W["r"], _W["ik"]
    if _native.supported(r.problem):
        fz = _native.Solver(r.problem)
    else:
        from .. import _fused
        fz = _fused.Fused(r.problem)
    solver = c["solver"]
    ds = solver.get("despike") or {}
    vmax = solver.get("max_joint_velocity")
    keep = c.get("save_skeleton", False)
    tls = threading.local()
    abort = threading.Event()
    failures = []

    def check_abort():
        if abort.is_set():
            raise failures[0]

    def solve(tg, wsp):
        cfg = replace(ik, max_joint_step=None if vmax is None else float(vmax) / float(tg.fps))
        q = fz.run(cfg, tg.X, wsp)
        if ds.get("enabled", False):
            fps = float(tg.fps)
            fz.despike(cfg, tg.X, q, wsp, threshold=float(ds.get("threshold", 7.5)) / fps,
                       ratio=float(ds.get("ratio", 4.0)), half_window=int(ds.get("half_window", 5)),
                       sigma=float(ds.get("sigma", 1.5)), pull_weight=float(ds.get("pull_weight", 200.0)),
                       passes=int(ds.get("passes", 3)))
        return q

    _tick("solver set up")
    # load the compiled kernels once, before the threads start
    p = r.problem
    x0 = fz.p0[None].repeat(4, axis=0)
    from types import SimpleNamespace
    solve(SimpleNamespace(X=x0, fps=30.0), fz.workspace())

    _tick("kernels loaded")

    def job(path):
        check_abort()
        t0 = time.perf_counter()
        c0 = time.thread_time()
        try:
            wsp = getattr(tls, "wsp", None)
            if wsp is None:
                wsp = tls.wsp = fz.workspace()
            tg = first[1] if path == first[0] else prepare(path, c, height, keep)
            _check_keypoints(tg, _W['keypoints'], path)
            check_abort()
            t1 = time.perf_counter()
            q = solve(tg, wsp)
            check_abort()
            t2 = time.perf_counter()
            name = _save(path, c["out"], tg, q, c, ik, _W["joints"])
            t3 = time.perf_counter()
            return name, len(q), t1 - t0, t2 - t1, None, t3 - t2, time.thread_time() - c0, t3, threading.get_ident()
        except ValidationError as error:
            if not failures:
                failures.append(_clip_error(path, error))
            abort.set()
            raise failures[0]
        except Exception as e:  # keep the batch going for other file errors
            return os.path.basename(path), 0, 0.0, 0.0, f"{type(e).__name__}: {e}", 0.0, 0.0, time.perf_counter(), 0

    results = []
    # Workers alternate short GIL-holding Python (loading, saving) with long GIL-free native calls;
    # with the default 5 ms switch interval a worker coming back from native code can wait that long
    # for the GIL, so hand it over sooner.
    sw = sys.getswitchinterval()
    sys.setswitchinterval(float(os.environ.get("FRDV_SWITCH_INTERVAL", 2e-4)))
    try:
        with ThreadPoolExecutor(max(1, min(workers, len(files)))) as ex:
            for i, res in enumerate(ex.map(job, files), 1):
                results.append(res[:5])
                if os.environ.get("FRDV_TIMING"):
                    _TIMING.append(res)
    finally:
        sys.setswitchinterval(sw)
    if os.environ.get("FRDV_TIMING"):
        load = sum(x[2] for x in _TIMING)
        solve_t = sum(x[3] for x in _TIMING)
        save = sum(x[5] for x in _TIMING)
        cpu = sum(x[6] for x in _TIMING)
        last = {}
        for x in _TIMING:
            last[x[8]] = max(last.get(x[8], 0.0), x[7])
        ends = sorted(last.values())
        print(f"[frdv] workers finished between t={ends[0] - _T0:.3f}s and t={ends[-1] - _T0:.3f}s")
        print(f"[frdv] thread time: load {load:.2f}s solve+despike {solve_t:.2f}s save {save:.2f}s "
              f"(sum over workers; CPU {cpu:.2f}s)")
    return results


_TIMING: list = []
_T0 = time.perf_counter()


def _tick(label):
    if os.environ.get("FRDV_TIMING"):
        print(f"[frdv] t={time.perf_counter() - _T0:6.3f}s  {label}")


def run(cfg) -> int:
    from omegaconf import OmegaConf

    _tick("config composed")
    c = to_dict(cfg)
    problems = validate_robot(c["robot"])
    if problems:
        print("robot config problems:\n  - " + "\n  - ".join(problems))
        return 2
    files = find_files(c["data"], c["dataset"].get("glob", "*"))[: c.get("limit")]
    if not files:
        print(f"no files matching '{c['dataset'].get('glob')}' in {c['data']}")
        return 2
    validate_subject_input(c, files)
    out = os.path.abspath(os.path.expanduser(c["out"]))
    from .diagnostics import warn_source_overwrite
    warn_source_overwrite(files, out, resolve_path(c['data']))
    os.makedirs(out, exist_ok=True)
    c["out"] = out
    height = robot_height(c["robot"])

    # Orientation-marker offsets: use the robot x dataset calibration, fit it now if missing.
    needs = [k for k in calibrated_keypoints(c["robot"]) if k in M.keypoint_map(c["dataset"])]
    if needs:
        calib, path = load_calibration(c["robot"], c["dataset"], c.get('calibration'))
        if calib is None and (c.get("calibration") or {}).get("auto", True):
            print(f"[frdv] no marker calibration for {c['robot']['name']} x {c['dataset']['name']}: fitting one "
                  f"(augment; disable with calibration.auto=false)")
            from .calibrate import calibrate
            calibrate(c, files)
        elif calib is None:
            print(f"[frdv] warning: no calibration for {len(needs)} marker(s); using their configured offsets")
        else:
            print(f"[frdv] marker calibration: {path}")

    _tick("robot validated, calibration found")
    # Tracked keypoints = robot keypoints present in the dataset map (checked on the first clip).
    files_in = list(files)
    tg0 = prepare(files[0], c, height, c.get("save_skeleton", False))
    _tick("first clip loaded")
    keypoints = tg0.keypoints
    skipped = [k for k, v in c["robot"]["keypoints"].items()
               if k not in keypoints and not v.get("calibrate")]
    print(f"[frdv] robot={c['robot']['name']} dataset={c['dataset']['name']}  {len(files)} file(s) -> {out}")
    scale_label = "manual source XYZ scale" if c["dataset"].get("scale", {}).get("mode") == "manual" else "scale"
    print(f"[frdv] tracking {len(keypoints)} keypoints; robot height {height:.2f} m, "
          f"performer {tg0.human_height:.2f} m " + (_seg_report(tg0) if tg0.segment_scale else
                                                     f"({scale_label} {tg0.scale.round(3).tolist()})"))
    if skipped:
        print(f"[frdv] robot keypoints not in this dataset (ignored): {', '.join(skipped)}")
    with open(os.path.join(out, "frdv_config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(c)))

    workers = c.get("workers") or os.cpu_count() or 1
    mode = c.get("mode", "auto")
    st = c.get("stitch") or {}
    if mode == "auto":
        mode = "files" if len(files) > 1 else "sequential"
    t_start = time.perf_counter()
    _tick("start of retargeting (tool timer)")
    results = []
    if mode == "chunks":
        ik = ik_config(c["robot"], c["solver"], keypoints, c["dataset"], calibration=c.get('calibration'))
        v = c["solver"].get("max_joint_velocity")  # workers are built once, so fix the cap from the first clip
        ik.max_joint_step = None if v is None else float(v) / tg0.fps
        with ParallelRetargeter(ik, n_workers=workers, burn_in=int(st.get("burn_in", 30)),
                                seam_tol=float(st.get("seam_tol", 1e-3)), min_chunk=int(st.get("min_chunk", 300))) as pr:
            joints = _joint_names(pr.local)
            for path in files:
                t0 = time.perf_counter()
                tg = prepare(path, c, height, c.get("save_skeleton", False))
                _check_keypoints(tg, keypoints, path)
                t1 = time.perf_counter()
                if abs(tg.fps - tg0.fps) > 1e-6:
                    print(f"[frdv] warning: {os.path.basename(path)} is {tg.fps:g} fps but the joint velocity cap "
                          f"was set for {tg0.fps:g} fps")
                q, rep = pr.run(tg.X, return_report=True)
                ds = postprocess(pr.local, c["solver"], tg.X, q, tg.fps)
                if ds:
                    print(f"[frdv] {os.path.basename(path)} despike: {ds}")
                results.append((_save(path, out, tg, q, c, ik, joints), len(q), t1 - t0, time.perf_counter() - t1, None))
                print(f"[frdv] {results[-1][0]}: {rep}")
    elif mode in ("sequential", "files") and (workers == 1 or len(files) == 1 or mode == "sequential") \
            and _fused_ok(c, height, keypoints):
        results = _run_threads(files, c, height, 1, first=(files_in[0], tg0))
    elif mode == "sequential" or workers == 1 or len(files) == 1:
        _init(c, height, keypoints)
        for path in files:
            results.append(_job(path))
    elif mode == "files" and _fused_ok(c, height, keypoints):
        # One process, one thread per worker: the whole clip (solve + despike) runs in one GIL-free
        # compiled call, so there is no process start-up and nothing to pickle.
        files.sort(key=os.path.getsize, reverse=True)  # longest first: better load balance
        results = _run_threads(files, c, height, workers, first=(files_in[0], tg0))
    elif mode == "files":
        files.sort(key=os.path.getsize, reverse=True)  # longest first: better load balance
        # Compile the numba kernels once here; the workers then load them from numba's on-disk
        # cache instead of all compiling at the same time (on a fresh install that would cost the
        # full compile time in every worker, in parallel).
        _init(c, height, keypoints)
        from ..smooth import despike
        despike(_W["r"], tg0.X[:60], _W["r"].run(tg0.X[:60]))
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(workers, len(files)), initializer=_init, initargs=(c, height, keypoints)) as pool:
            for i, res in enumerate(pool.imap_unordered(_job, files), 1):
                results.append(res)
                print(f"\r[frdv] {i}/{len(files)} {res[0]:<40.40s}", end="", flush=True)
        print()
    else:
        print(f"unknown mode '{mode}' (auto | sequential | files | chunks)")
        return 2
    wall = time.perf_counter() - t_start
    _tick("all clips done")

    errors = [(n, e) for n, *_, e in results if e]
    for n, e in errors:
        print(f"[frdv] FAILED {n}: {e}")
    frames = sum(r[1] for r in results)
    solve = sum(r[3] for r in results)
    # Wall time measures throughput; per-clip solver times overlap across workers.
    print(f"[frdv] {len(results) - len(errors)}/{len(results)} clips, {frames} frames in {wall:.1f}s "
          f"({frames / max(wall, 1e-9):,.0f} frames/s; wall {wall / max(frames, 1) * 1e6:.2f} us/frame; "
          f"solver sum {solve / max(frames, 1) * 1e6:.2f} us/frame)")
    if c.get("vis") and frames:
        from .view import serve
        serve(out, port=int(c.get("port", 8080)))
    return 1 if errors else 0


def main():
    from .config import cli
    try:
        status = cli(__doc__, run) or 0
    except ValidationError as error:
        print(f"[frdv] ERROR: {error}", file=sys.stderr)
        status = 1
    sys.exit(status)


if __name__ == "__main__":
    main()
