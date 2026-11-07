"""frdv demo: retarget LAFAN1 onto Unitree G1 and watch the motions in the browser.

    uv run --extra demo frdv-demo                                       # from this checkout
    frdv-demo --clips 10                                                # subset
    frdv-demo --clip walk1_subject1                                     # select one motion

The G1 model comes from MuJoCo Menagerie and LAFAN1 (Ubisoft La Forge, CC BY-NC-ND 4.0, 144 MB)
from its GitHub release; both are cached in ~/.cache/frdv ($FRDV_CACHE). Each run retargets
the clips on all cores, then opens a viser viewer with the G1 motions in one scene.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from ..assets import cache_dir

LAFAN1_URL = ("https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/"
              "master/lafan1/lafan1.zip")
LAFAN1_CLIPS = 77


def default_workers() -> int:
    """All cores, except on Apple silicon: performance cores + a third of the efficiency cores
    (more workers only add start-up cost there)."""
    n = os.cpu_count() or 1
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            sysctl = lambda k: int(subprocess.check_output(["sysctl", "-n", k], text=True))   # noqa: E731
            return max(1, sysctl("hw.perflevel0.physicalcpu") + sysctl("hw.perflevel1.physicalcpu") // 3)
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    return n


def _progress(done: int, total: int, t0: float):
    mb, tot = done / 2**20, total / 2**20
    rate = mb / max(time.perf_counter() - t0, 1e-6)
    print(f"\r[frdv-demo] downloading LAFAN1: {mb:6.1f} / {tot:.1f} MB ({rate:.1f} MB/s)", end="", flush=True)


def fetch_lafan1(root: Path) -> Path:
    """LAFAN1 BVH folder, downloaded and extracted once."""
    out = root / "lafan1"
    if len(list(out.glob("*.bvh"))) >= LAFAN1_CLIPS:
        return out
    out.mkdir(parents=True, exist_ok=True)
    tmp = root / "lafan1.zip.part"
    print(f"[frdv-demo] LAFAN1 (Ubisoft La Forge, CC BY-NC-ND 4.0) -> {out}")
    t0 = time.perf_counter()
    with urllib.request.urlopen(LAFAN1_URL) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            _progress(done, total, t0)
    print()
    with zipfile.ZipFile(tmp) as z:
        for name in z.namelist():
            if name.endswith(".bvh") and not name.startswith("__MACOSX"):
                with z.open(name) as src, open(out / os.path.basename(name), "wb") as dst:
                    shutil.copyfileobj(src, dst)
    tmp.unlink()
    return out


def _free_port(port: int) -> int:
    """``port`` if it is free, else the next free one."""
    import socket
    for p in range(port, port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", p))
                return p
            except OSError:
                continue
    return port


ROBOTS = ["g1"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--robots", nargs="+", choices=ROBOTS, default=ROBOTS, help="supported robot (default: g1)")
    ap.add_argument("--clips", type=int, default=None, help="only the first N clips (default: all 77)")
    ap.add_argument("--clip", default=None, help="show only this G1 motion (default: all motions)")
    ap.add_argument("--workers", type=int, default=None, help="parallel workers (default: auto)")
    ap.add_argument("--port", type=int, default=8080, help="viewer port (the next free one if taken)")
    ap.add_argument("--no-view", action="store_true", help="retarget only, don't start the viewer")
    ap.add_argument("--reuse", action="store_true", help="show the previous run's results instead of retargeting")
    args = ap.parse_args(argv)

    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    from .common import to_dict
    from .config import compose
    root = cache_dir()
    data = fetch_lafan1(root)                   # downloaded once
    workers = args.workers or default_workers()
    outs, summary = [], []
    for robot in args.robots:
        cfg = compose([f"robot={robot}", "dataset=lafan1"])
        name = cfg.robot.name
        to_dict(cfg.robot)                      # resolves ${menagerie:...}: downloads the model once
        out = root / "demo" / name
        outs.append(str(out))
        if args.reuse and any(out.glob("*.npz")):
            continue
        from .retarget import run
        shutil.rmtree(out, ignore_errors=True)
        n = args.clips or LAFAN1_CLIPS
        print(f"\n[frdv-demo] {name}: retargeting {n} LAFAN1 clip(s) with {workers} worker(s)")
        overrides = [f"robot={robot}", "dataset=lafan1", f"data={data}", f"out={out}", f"workers={workers}"]
        if args.clips:
            overrides.append(f"limit={args.clips}")
        t0 = time.perf_counter()
        if run(compose(overrides)):
            return 1
        frames = sum(int(__import__("numpy").load(f)["qpos"].shape[0]) for f in out.glob("*.npz"))
        summary.append((name, frames, time.perf_counter() - t0))
    if summary:
        tot_f, tot_t = sum(s[1] for s in summary), sum(s[2] for s in summary)
        print("\n[frdv-demo] robot                  frames     time")
        for name, f, t in summary:
            print(f"[frdv-demo] {name:20s} {f:9,d} {t:7.1f} s")
        print(f"[frdv-demo] {'total':20s} {tot_f:9,d} {tot_t:7.1f} s  ({tot_f / tot_t:,.0f} frames/s)")
    if args.no_view:
        return 0
    from .view import serve
    port = _free_port(args.port)
    what = f"motion {args.clip}" if args.clip else "all motions"
    print(f"[frdv-demo] open http://localhost:{port}  ({len(outs)} robots, {what}; Ctrl+C to stop)")
    serve(outs, port=port, clip=args.clip, labels=args.clip is not None)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
