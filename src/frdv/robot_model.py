"""Load any robot description (MJCF or URDF) as a floating-base MuJoCo model.

* URDF and MJCF are both parsed with ``mujoco.MjSpec``.
* A floating base is added when the model has none (URDF roots are welded to the world).
* Fixed links are kept as bodies (MuJoCo's URDF import would otherwise fuse them away, and they
  are often exactly the frames you want to track: toes, hands, sole frames).
* Mesh paths are repaired: ``package://pkg/...`` URIs and wrong relative folders are resolved by
  searching around the file; meshes that still cannot be loaded (e.g. ``.dae``, not supported by
  MuJoCo) are dropped with a warning -- kinematics doesn't need them.
"""

from __future__ import annotations

import os
import re
import warnings
from functools import lru_cache

import mujoco

MESH_EXT = {".stl", ".obj", ".msh"}


def _mesh_index(root: str):
    """basename (lowercase) -> path for mesh files under ``root`` and its parent."""
    index = {}
    for base in {root, os.path.dirname(root)}:
        for dirpath, _, files in os.walk(base):
            if dirpath.count(os.sep) - base.count(os.sep) > 4:
                continue
            for f in files:
                if os.path.splitext(f)[1].lower() in MESH_EXT | {".dae"}:
                    index.setdefault(f.lower(), os.path.join(dirpath, f))
    return index


def _fix_meshes(spec: mujoco.MjSpec, src: str):
    base = os.path.dirname(os.path.abspath(src))
    meshdir = os.path.join(base, spec.meshdir) if spec.meshdir else base
    index = None
    dropped = []
    for mesh in list(spec.meshes):
        f = mesh.file
        if not f:
            continue
        cand = [f, os.path.join(meshdir, f), os.path.join(base, f)]
        if f.startswith("package://"):
            rel = f[len("package://"):].split("/", 1)[-1]
            cand += [os.path.join(base, rel), os.path.join(os.path.dirname(base), rel)]
        path = next((c for c in cand if os.path.isfile(c)), None)
        if path is None:
            if index is None:
                index = _mesh_index(base)
            path = index.get(os.path.basename(f).lower())
        if path is None or os.path.splitext(path)[1].lower() not in MESH_EXT:
            dropped.append(os.path.basename(f))
            for g in list(spec.geoms):
                if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname == mesh.name:
                    spec.delete(g)
            spec.delete(mesh)
            continue
        mesh.file = os.path.abspath(path)
    if dropped:
        warnings.warn(f"{os.path.basename(src)}: dropped {len(dropped)} mesh(es) MuJoCo cannot load "
                      f"(e.g. {dropped[0]}); kinematics is unaffected", stacklevel=3)
    spec.meshdir = ""


def _ensure_free_joint(spec: mujoco.MjSpec):
    if any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in spec.joints):
        return
    world = spec.worldbody
    roots = [b for b in spec.bodies if b.parent is not None and b.parent.name == world.name]
    if not roots:
        raise ValueError("model has no bodies")
    root = roots[0]
    old = list(root.joints)
    if old:   # e.g. a planar / slide+hinge base: replace it by a free joint
        warnings.warn(f"replacing the {len(old)} base joint(s) of '{root.name}' by a free joint", stacklevel=3)
        for j in old:
            spec.delete(j)
    root.add_freejoint()


def load_spec(path: str) -> mujoco.MjSpec:
    path = os.path.abspath(os.path.expanduser(path))
    if path.lower().endswith(".urdf"):
        # compiler options must be in the file: the URDF parser drops visuals while parsing
        with open(path, encoding="utf-8", errors="replace") as f:
            text = re.sub(r"<!--.*?-->", "", f.read(), flags=re.S)
        opts = '<compiler discardvisual="false" fusestatic="false" strippath="false" balanceinertia="true"/>'
        if "<mujoco>" in text or "<mujoco " in text:
            text = re.sub(r"<compiler\b[^>]*/>", "", text)
            text = re.sub(r"(<mujoco\b[^>]*>)", r"\1" + opts, text, count=1)
        else:
            text = re.sub(r"(<robot\b[^>]*>)", r"\1<mujoco>" + opts + "</mujoco>", text, count=1)
        spec = mujoco.MjSpec.from_string(text)
    else:
        spec = mujoco.MjSpec.from_file(path)
    _fix_meshes(spec, path)
    _ensure_free_joint(spec)
    return spec


@lru_cache(maxsize=32)
def _compiled_xml(path: str, mtime: float) -> str:
    return load_spec(path).to_xml()


def load_model(path: str) -> mujoco.MjModel:
    """MjModel with a free joint first, for an MJCF or URDF file.

    Compiling a model with meshes takes ~50 ms and the CLI tools ask for the same one several times,
    so the compiled model is cached (per path and modification time) and a private copy returned."""
    import copy
    path = os.path.abspath(os.path.expanduser(path))
    return copy.copy(_load_model(path, os.path.getmtime(path)))


@lru_cache(maxsize=8)
def _load_model(path: str, mtime: float) -> mujoco.MjModel:
    cache = _binary_cache_path(path)
    if cache is not None and cache.exists():
        try:
            return mujoco.MjModel.from_binary_path(str(cache))
        except Exception:  # corrupt / other mujoco build: recompile
            pass
    m = _compile_model(path)
    if cache is not None:
        try:
            import tempfile
            cache.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(suffix=".mjb", dir=cache.parent)
            os.close(fd)
            mujoco.mj_saveModel(m, tmp, None)
            os.replace(tmp, cache)
        except Exception:
            pass
    return m


def _compile_model(path: str) -> mujoco.MjModel:
    # (same as `from_xml_path if _is_floating_mjcf else load_spec(...).compile()`, compiling once)
    if not path.lower().endswith(".urdf"):
        try:
            m = mujoco.MjModel.from_xml_path(path)
            if m.njnt > 0 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
                return m
        except ValueError:
            pass
    return load_spec(path).compile()


def _binary_cache_path(path: str):
    """Compiled-model cache (MJB) keyed by the model file, every file next to it (meshes, includes)
    and the MuJoCo version. Compiling a model with meshes costs ~0.1 s per process otherwise."""
    if os.environ.get("FRDV_NO_MODEL_CACHE", "0") in ("1", "true", "True"):
        return None
    import hashlib
    from pathlib import Path
    root = os.path.dirname(path)
    h = hashlib.sha256(f"{path}|{mujoco.__version__}".encode())
    try:
        for dirpath, dirs, files in os.walk(root):
            dirs.sort()
            if dirpath.count(os.sep) - root.count(os.sep) > 3:
                continue
            for f in sorted(files):
                st = os.stat(os.path.join(dirpath, f))
                h.update(f"{dirpath}/{f}|{st.st_mtime_ns}|{st.st_size}".encode())
    except OSError:
        return None
    base = os.environ.get("FRDV_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "frdv")
    return Path(base) / "models" / f"{h.hexdigest()[:24]}.mjb"


def _is_floating_mjcf(path: str) -> bool:
    if path.lower().endswith(".urdf"):
        return False
    try:
        m = mujoco.MjModel.from_xml_path(path)
    except ValueError:
        return False
    return m.njnt > 0 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE


def _geom_points(m: mujoco.MjModel, g: int):
    """Sample points of geom ``g`` in its body's frame (mesh vertices or primitive extremes)."""
    import numpy as np
    t, s = mujoco.mjtGeom(int(m.geom_type[g])), m.geom_size[g]     # enum: numpy ints fail `in (enum, ...)`
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        local = m.mesh_vert[a:a + n]
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        local = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s
    elif t in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER,
               mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        r = s[0]
        h = s[1] if t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER) else 0.0
        e = np.r_[np.eye(3), -np.eye(3)]
        sc = s if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID else np.array([r, r, r])
        local = np.concatenate([e * sc + [0, 0, z] for z in ((-h, h) if h else (0.0,))])
        if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            local = np.concatenate([local, [[0, 0, h + r], [0, 0, -h - r]]])
    else:
        return np.zeros((0, 3))
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    return local @ R.reshape(3, 3).T + m.geom_pos[g]


def surface_points(m: mujoco.MjModel, min_sep: float = 0.03, exclude_bodies=()):
    """[(body name, offset)] extreme surface points of every body's geometry along 26 directions.

    Used as "stay above the floor" points for robots without collision capsules: knees, hands,
    elbows and the torso then can't sink into the ground in crawling / falling motions.
    Collision geoms are used when a body has any, else its visual geoms.
    """
    import numpy as np
    import itertools
    dirs = np.array([v for v in itertools.product((-1, 0, 1), repeat=3) if any(v)], dtype=float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)              # 26 directions
    out = []
    for b in range(1, m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name in exclude_bodies:
            continue
        gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_rgba[g][3] > 0 or
              (m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g]))]
        col = [g for g in gs if m.geom_contype[g] or m.geom_conaffinity[g]]
        pts = [_geom_points(m, g) for g in (col or gs)]
        pts = np.concatenate(pts) if pts else np.zeros((0, 3))
        if len(pts) == 0 or np.ptp(pts, axis=0).max() < 0.02:
            continue
        chosen = []
        for dvec in dirs:
            p = pts[np.argmax(pts @ dvec)]
            if all(np.linalg.norm(p - c) > min_sep for c in chosen):
                chosen.append(p)
        out += [(name, c) for c in chosen]
    return out
