"""augment: fit the robot-side offsets of orientation markers from data.

    augment robot=g1 data=/path/to/lafan1 dataset.augmentation=true
    augment robot=myrobot data=/path/to/lafan1 dataset.augmentation=true calibration.clips=40

Robot keypoints marked ``calibrate: true`` (finger tips, upper-arm/forearm surface points, ...)
are rigidly attached to a link, but where exactly the performer's marker sits on that link differs
between human skeletons and robots. Instead of hand-placing them, or measuring them from one T-pose
frame, frdv fits them over many poses:

1. solve the clips tracking only the uncalibrated keypoints (joint centres);
2. each marker's offset = its mean position in its link's frame over all solved frames;
3. solve again with the markers at those offsets, refit; repeat (``calibration.iterations``).

Augmentation is off by default. This fits optional marker offsets, not manual body scaling.
The result updates ``<user config dir>/calibration/<robot>/<dataset>/base.yaml`` (or the selected
subject profile), preserving body scales and alignment. It is picked up
automatically by frdv-retarget (which also runs this on the fly when a calibration is missing).
"""

from __future__ import annotations

import datetime
import multiprocessing as mp
import os
import sys
from pathlib import Path

from .common import THREAD_VARS

for _v in THREAD_VARS:
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from ..kinematics import Kinematics  # noqa: E402
from .. import motion as M  # noqa: E402
from ..retarget import Retargeter  # noqa: E402
from .common import (calibrated_keypoints, calibration_paths, find_files, ik_config, per_clip_setup,  # noqa: E402
                     resolve_path, robot_height, to_dict, validate_robot, validate_subject_input)


def _solve_clip(args):
    from .retarget import prepare
    path, cfg, height, offsets, markers, with_markers, stride = args
    rb, ds, sv = cfg["robot"], cfg["dataset"], dict(cfg["solver"])
    sv["despike"] = {"enabled": False}
    tg = prepare(path, cfg, height)
    keep = [k for k in tg.keypoints if with_markers or k not in markers]
    idx = [tg.keypoints.index(k) for k in keep]
    r = Retargeter(ik_config(rb, sv, keep, ds, offsets=offsets, calibration=cfg.get('calibration')))
    per_clip_setup(r, sv, tg.fps)
    q = r.run(tg.X[:, idx])
    return q[::stride], {k: tg.X[::stride, tg.keypoints.index(k)] for k in markers if k in tg.keypoints}


def fit_step(kin: Kinematics, frames: dict, results) -> tuple[dict, dict, int]:
    """One fit: each marker's mean position in its link frame over all solved poses.

    ``frames``: {marker: (body id, local pos of the named link/site in that body)};
    ``results``: [(qpos (T, nq), {marker: targets (T, 3)}), ...].
    Returns (offsets {marker: (3,)}, spread {marker: max per-axis std}, frames used).
    """
    d = kin.data
    acc = {k: [] for k in frames}
    n = 0
    for q, X in results:
        for t in range(len(q)):
            kin.forward(q[t], with_jac=False)
            for k, (b, local) in frames.items():
                if k in X:
                    acc[k].append(d.xmat[b].reshape(3, 3).T @ (X[k][t] - d.xpos[b]) - local)
            n += 1
    offsets = {k: np.mean(v, axis=0) for k, v in acc.items() if v}
    spread = {k: float(np.std(v, axis=0).max()) for k, v in acc.items() if v}
    return offsets, spread, n


def fit_offsets(cfg: dict, files: list[str], iterations=3, stride=3, workers=None, log=print):
    """Returns (offsets {keypoint: [x, y, z]}, spread {keypoint: std (m)}, frames used)."""
    rb, ds = cfg["robot"], cfg["dataset"]
    kp_map = M.keypoint_map(ds)
    markers = [k for k in calibrated_keypoints(rb) if k in kp_map]
    if not markers:
        return {}, {}, 0
    height = robot_height(rb)
    kin = Kinematics(resolve_path(rb["xml"]))
    frames = {k: kin.frame_point(rb["keypoints"][k]["link"]) for k in markers}   # (body, local pos)
    offsets: dict = {}
    workers = max(1, min(workers or os.cpu_count() or 1, len(files)))
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for it in range(iterations):
            jobs = [(f, cfg, height, offsets, markers, it > 0, stride) for f in files]
            results = pool.map(_solve_clip, jobs)
            new, spread, n = fit_step(kin, frames, results)
            change = max((np.linalg.norm(new[k] - np.asarray(offsets[k])) for k in new if k in offsets), default=np.nan)
            offsets = {k: [float(x) for x in v] for k, v in new.items()}
            log(f"[augment] iteration {it + 1}/{iterations}: {n} frames, "
                f"max offset change {change * 100 if np.isfinite(change) else float('nan'):.2f} cm, "
                f"median spread {np.median(list(spread.values())) * 100:.1f} cm")
    return offsets, spread, n


def save(path: Path, cfg: dict, offsets: dict, spread: dict, n_files: int, n_frames: int):
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "robot": cfg["robot"]["name"], "dataset": cfg["dataset"]["name"],
        "fitted": datetime.datetime.now().isoformat(timespec="seconds"),
        "clips": n_files, "frames": n_frames,
        "offsets": {k: [round(x, 5) for x in v] for k, v in offsets.items()},
        "spread_cm": {k: round(v * 100, 2) for k, v in spread.items()},
    }
    document = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    calibration = document.setdefault('calibration', {})
    for key, value in [('robot', cfg['robot']['name']), ('dataset', cfg['dataset']['name'])]:
        if calibration.get(key, value) != value:
            raise ValueError(f'{path}: refusing to overwrite another {key} calibration')
        calibration[key] = value
    calibration['markers'] = data
    path.write_text('# @package _global_\n# Robot-owned calibration; marker offsets are in link-frame metres.\n'
                    + yaml.safe_dump(document, sort_keys=False))
    cfg.setdefault('calibration', {})['markers'] = data
    sources = cfg['calibration'].setdefault('sources', [])
    if str(path) not in sources:
        sources.append(str(path))


def calibrate(c: dict, files: list[str], out: Path | None = None, log=print) -> Path | None:
    cal = c.get("calibration") or {}
    n = int(cal.get("clips", 20))
    pick = files[:: max(1, len(files) // n)][:n] if len(files) > n else files
    n_markers = len([k for k in calibrated_keypoints(c["robot"]) if k in M.keypoint_map(c["dataset"])])
    if not n_markers:
        log("[augment] no active orientation markers; enable them with dataset.augmentation=true")
        return None
    log(f"[augment] fitting {n_markers} marker offsets for "
        f"robot={c['robot']['name']} dataset={c['dataset']['name']} on {len(pick)} clip(s)")
    offsets, spread, frames = fit_offsets(c, pick, int(cal.get("iterations", 3)), int(cal.get("stride", 3)),
                                          c.get("workers"), log)
    if not offsets:
        log("[augment] no marker offsets to fit (no 'calibrate: true' keypoint shared with this dataset)")
        return None
    out = out or (Path(cal["out"]) if cal.get("out") else
                  calibration_paths(c["robot"], c["dataset"], cal.get('profile'))[0])
    save(out, c, offsets, spread, len(pick), frames)
    log(f"[augment] wrote {out}")
    return out


def run(cfg) -> int:
    c = to_dict(cfg)
    problems = validate_robot(c["robot"])
    if problems:
        print("robot config problems:\n  - " + "\n  - ".join(problems))
        return 2
    files = find_files(c["data"], c["dataset"].get("glob", "*"))
    if not files:
        print(f"no files matching '{c['dataset'].get('glob')}' in {c['data']}")
        return 2
    validate_subject_input(c, files)
    out = calibrate(c, files)
    if out is not None:
        import yaml
        data = yaml.safe_load(out.read_text())['calibration']['markers']
        for k, v in data["offsets"].items():
            print(f"  {k:22s} on {c['robot']['keypoints'][k]['link']:26s} "
                  f"[{', '.join(f'{x * 100:6.1f}' for x in v)}] cm  (spread {data['spread_cm'][k]:.1f} cm)")
    return 0


def main():
    from .config import cli
    sys.exit(cli(__doc__, run, extra_help="calibration.clips=N calibration.iterations=K calibration.out=<path>") or 0)


if __name__ == "__main__":
    main()
