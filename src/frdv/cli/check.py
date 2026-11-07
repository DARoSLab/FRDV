"""frdv-check: sanity-check a robot + dataset config on one clip before a big run.

    frdv-check robot=myrobot dataset=lafan1 data=/path/to/clip.bvh
    frdv-check robot=g1 dataset=lafan1 data=walk1_subject1.bvh frames=600

Reports config problems, the joints available in the clip (when a mapping is missing),
performer vs robot height and scale, and per-keypoint tracking error over the first ``frames``
frames, which quickly reveals a wrong link mapping, axis rotation or offset.
"""

from __future__ import annotations

import sys

import numpy as np

from .. import motion as M
from ..retarget import Retargeter
from .common import (find_files, ik_config, per_clip_setup, robot_height, robot_points, target_offset,
                     to_dict, validate_robot, validate_subject_input)
from .retarget import _loader_opts


def run(cfg) -> int:
    c = to_dict(cfg)
    ds, rb = c["dataset"], c["robot"]
    ok = True
    problems = validate_robot(rb)
    print(f"robot '{rb['name']}': " + ("ok" if not problems else "PROBLEMS"))
    for p in problems:
        print(f"  - {p}")
        ok = False
    files = find_files(c["data"], ds.get("glob", "*"))
    if not files:
        print(f"no files matching '{ds.get('glob')}' in {c['data']}")
        return 2
    validate_subject_input(c, files)
    path = files[0]
    mot = M.load(path, ds["format"], **_loader_opts(ds))
    print(f"clip {path}: {mot.positions.shape[0]} frames @ {mot.fps:g} fps, {len(mot.names)} joints")

    kp_map = M.keypoint_map(ds)
    missing = {k: v for k, v in kp_map.items() if v not in mot.names}
    if missing:
        ok = False
        print("dataset keypoints whose joint is not in the clip:")
        for k, v in missing.items():
            print(f"  - {k}: '{v}'")
        print("joints in the clip:\n  " + ", ".join(mot.names))
    shared = [k for k in rb["keypoints"] if k in kp_map and kp_map[k] in mot.names]
    print(f"tracked keypoints ({len(shared)}): {', '.join(shared)}")
    unused_r = [k for k, v in rb["keypoints"].items() if k not in kp_map and not v.get("calibrate")]
    if unused_r:
        print(f"robot keypoints with no dataset joint (ignored): {', '.join(unused_r)}")
    if problems or not shared:
        return 1

    height = robot_height(rb)
    tg = M.build_targets(mot, ds, height, list(rb["keypoints"]), extra_offset=target_offset(rb, ds),
                         robot_points=robot_points(rb))
    from .diagnostics import warn_measurements
    warn_measurements(tg, ds, height, path)
    if tg.segment_scale:
        print(f"robot height {height:.2f} m, performer {tg.human_height:.2f} m -> per-segment scale "
              + ", ".join(f"{k} {v:.3f}" for k, v in tg.segment_scale.items()))
    else:
        label = "manual source XYZ scale" if ds.get("scale", {}).get("mode") == "manual" else "scale"
        print(f"robot height {height:.2f} m, performer {tg.human_height:.2f} m -> {label} {np.round(tg.scale, 4).tolist()}")
    if not 0.3 < tg.human_height < 2.5:
        print("  ! performer height looks unusual: check the first frame, units and axes (warning only)")
    z0 = tg.X[0, :, 2]
    print(f"frame 0 target height range {z0.min():.2f} .. {z0.max():.2f} m (should span roughly 0 .. robot height)")

    n = min(int(c.get("frames") or 300), len(tg.X))
    r = Retargeter(ik_config(rb, c["solver"], tg.keypoints, ds, calibration=c.get('calibration')))
    per_clip_setup(r, c["solver"], tg.fps)
    q = r.run(tg.X[:n])
    p = r.problem
    err = np.zeros((n, p.S))
    for t in range(n):
        p.kin.forward(q[t], with_jac=False)
        err[t] = np.linalg.norm(p.kin.positions(p.pts)[: p.S] - tg.X[t], axis=1)
    e = err.mean(0) * 1e3
    print(f"tracking error over {n} frames (mean {e.mean():.1f} mm):")
    for k, link, v in sorted(zip(tg.keypoints, r.cfg.targets, e), key=lambda x: -x[2]):
        flag = "  <-- check mapping/offset" if v > max(3 * np.median(e), 80) else ""
        print(f"  {k:22s} -> {link[0] if isinstance(link, (tuple, list)) else link:28s} {v:7.1f} mm{flag}")
    print("OK" if ok else "issues found (see above)")
    return 0 if ok else 1


def main():
    from .config import cli

    def _run(cfg):
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.setdefault("frames", 300)
        return run(cfg)
    sys.exit(cli(__doc__, _run, extra_help="extra: frames=N (frames retargeted for the check, default 300)") or 0)


if __name__ == "__main__":
    main()
