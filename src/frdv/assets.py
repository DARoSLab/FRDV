"""Download public robot models on demand (nothing is bundled with frdv).

    from frdv.assets import menagerie
    xml = menagerie("unitree_g1")          # ~/.cache/frdv/mujoco_menagerie/refs/<ref-key>/unitree_g1/g1.xml

Only the model file, the files it includes and the meshes / textures it references are fetched
from github.com/google-deepmind/mujoco_menagerie (each robot keeps its own license there).
Set ``FRDV_CACHE`` to change the cache folder and ``FRDV_MENAGERIE_REF`` to pin a commit.
"""

from __future__ import annotations

import os
import hashlib
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

MENAGERIE = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{ref}/{path}"

# robot folder -> model file (the floating-base robot, not the scene)
MENAGERIE_MODELS = {
    "unitree_g1": "g1.xml",
}


def cache_dir() -> Path:
    return Path(os.environ.get("FRDV_CACHE", Path.home() / ".cache" / "frdv")).expanduser()


def _get(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.replace(dst)


def _refs(xml_file: Path):
    """(includes, asset files) referenced by an MJCF file, relative paths as written."""
    root = ET.parse(xml_file).getroot()
    comp = root.find("compiler")
    meshdir = comp.get("meshdir", comp.get("assetdir", "")) if comp is not None else ""
    texdir = comp.get("texturedir", comp.get("assetdir", "")) if comp is not None else ""
    inc = [e.get("file") for e in root.iter("include") if e.get("file")]
    files = [os.path.join(meshdir, e.get("file")) for e in root.iter("mesh") if e.get("file")]
    files += [os.path.join(meshdir, e.get("file")) for e in root.iter("hfield") if e.get("file")]
    files += [os.path.join(texdir, e.get("file")) for e in root.iter("texture") if e.get("file")]
    return inc, files


def menagerie(robot: str, model: str | None = None, quiet: bool = False) -> str:
    """Local path of a MuJoCo Menagerie model, downloaded on first use."""
    model = model or MENAGERIE_MODELS.get(robot)
    if model is None:
        raise KeyError(f"unknown Menagerie robot '{robot}'; pass model='<file>.xml' "
                       f"(known: {', '.join(MENAGERIE_MODELS)})")
    ref = os.environ.get("FRDV_MENAGERIE_REF", "main")
    # A ref can contain slashes (branch names), so use a stable, path-safe identity.
    # Unversioned caches cannot establish which ref they contain and are left untouched.
    key = hashlib.sha256(ref.encode()).hexdigest()[:16]
    base = cache_dir() / "mujoco_menagerie" / 'refs' / key / robot
    todo, seen, n = [model], set(), 0
    while todo:
        rel = os.path.normpath(todo.pop())
        if rel in seen:
            continue
        seen.add(rel)
        dst = base / rel
        if not dst.exists():
            if not quiet and n == 0:
                print(f"[frdv] downloading {robot} from MuJoCo Menagerie -> {base}", flush=True)
            _get(MENAGERIE.format(ref=ref, path=f"{robot}/{rel}"), dst)
            n += 1
        if rel.endswith(".xml"):
            inc, files = _refs(dst)
            d = os.path.dirname(rel)
            todo += [os.path.join(d, f) for f in inc] + [os.path.join(d, f) for f in files]
    if n and not quiet:
        print(f"[frdv] {robot}: {n} file(s) downloaded", flush=True)
    return str(base / model)
