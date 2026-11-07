"""frdv-view: play back retargeted motions, all at once in one scene (viser).

    frdv-view outputs/unitree_g1/lafan1              # then open http://localhost:8080
    frdv-view outputs/unitree_g1/lafan1 --limit 20 --spacing 2 --port 8081
    frdv-view outputs/unitree_g1/lafan1/walk1_subject1.npz
    frdv-view outputs --clip dance2_subject4 --labels   # one motion on every robot under outputs/

All clips play simultaneously. Each starts on its own cell of a compact grid and follows its real
trajectory, so robots may walk through each other. Optional: "packed" layout (non-overlapping
patches sized to each clip's path) and "in place" (root's horizontal travel removed). Click a robot to see which motion it
is: it is highlighted and labelled, its mocap targets (red) and tracking-error segments (yellow)
are shown, and the side panel shows the clip name and current frame.

Rendering: one instanced mesh per robot body (instances = clips), so a frame update is one
message per body regardless of how many clips are shown.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time

import mujoco
import numpy as np

try:
    import viser
except ImportError:  # pragma: no cover
    sys.exit("frdv-view needs viser: pip install 'frdv[vis]'  (or uv sync --extra vis)")

from ..kinematics import Kinematics

HIGHLIGHT = np.array([255, 140, 0], dtype=np.uint8)


def load_clips(paths, limit=None, clip=None, lazy=False):
    """Clips from .npz files / folders, including nested AMASS collection/subject outputs. With ``clip``,
    ``frdv-view out --clip walk1_subject1`` shows that motion on every robot under ``out/``.
    ``lazy``: keep qpos as float32 and read the mocap targets only when a clip is selected (for
    scenes with hundreds of clips)."""
    paths = [paths] if isinstance(paths, str) else list(paths)
    files = []
    for p in paths:
        if os.path.isfile(p):
            files.append(p)
        elif clip:
            files += sorted(glob.glob(os.path.join(p, "**", f"{clip}.npz"), recursive=True))
        else:
            files += sorted(glob.glob(os.path.join(p, "**", "*.npz"), recursive=True))
    clips = []
    for f in files[:limit]:
        d = np.load(f)
        if "robot_xml" not in d:
            raise SystemExit(f"{f} has no robot metadata (robot_xml / robot_links); re-run frdv-retarget")
        clips.append(dict(name=os.path.splitext(os.path.basename(f))[0], file=f,
                          q=d["qpos"].astype(np.float32) if lazy else d["qpos"],
                          X=None if lazy else d["targets"], fps=float(d["fps"]), xml=str(d["robot_xml"]),
                          robot=str(d["robot"]) if "robot" in d else os.path.basename(os.path.dirname(f)),
                          links=[str(x) for x in d["robot_links"]],
                          offsets=d["robot_link_offsets"] if "robot_link_offsets" in d else None))
    by_identity = {}
    for c in clips:
        by_identity.setdefault((c["xml"], c["name"]), []).append(c)
    for group in by_identity.values():
        if len(group) > 1:
            common = os.path.commonpath([os.path.dirname(c["file"]) for c in group])
            for c in group:
                c["name"] = os.path.splitext(os.path.relpath(c["file"], common))[0]
    for c in clips:
        c["clip"] = c["name"]
    if len({c["xml"] for c in clips}) > 1:      # several robots: label clips by robot
        for c in clips:
            c["name"] = f"{c['robot']} / {c['name']}"
    return clips


def _primitive_mesh(t, size):
    """(verts, faces) for box / sphere / capsule / cylinder / ellipsoid geoms, in the geom frame."""
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * size
        f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        return v, f
    n_u, n_v = 16, 9
    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    if t in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_CAPSULE):
        r = size[0]
        sc = size[:3] if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID else np.array([r, r, r])
        h = size[1] if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
        th = np.linspace(0, np.pi, n_v)
        rings = []
        for k, a in enumerate(th):
            z = np.cos(a) * sc[2] + (h if k < n_v // 2 else -h if k > n_v // 2 else 0.0)
            ring = np.stack([np.cos(u) * np.sin(a) * sc[0], np.sin(u) * np.sin(a) * sc[1], np.full(n_u, z)], 1)
            rings.append(ring)
            if h and k == n_v // 2:      # duplicate the equator for the cylindrical part
                rings[-1] = ring + [0, 0, h]
                rings.append(ring - [0, 0, h])
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r, h = size[0], size[1]
        rings = [np.stack([np.cos(u) * r * 1e-6, np.sin(u) * r * 1e-6, np.full(n_u, h)], 1),
                 np.stack([np.cos(u) * r, np.sin(u) * r, np.full(n_u, h)], 1),
                 np.stack([np.cos(u) * r, np.sin(u) * r, np.full(n_u, -h)], 1),
                 np.stack([np.cos(u) * r * 1e-6, np.sin(u) * r * 1e-6, np.full(n_u, -h)], 1)]
    else:
        return None
    v = np.concatenate(rings)
    f = []
    for k in range(len(rings) - 1):
        for j in range(n_u):
            a, b = k * n_u + j, k * n_u + (j + 1) % n_u
            f += [[a, a + n_u, b], [b, a + n_u, b + n_u]]
    return v, np.array(f)


def grid(n, spacing):
    """Square grid of n cells, centred on the origin. Returns (n,3) offsets."""
    cols = int(np.ceil(np.sqrt(n)))
    idx = np.arange(n)
    off = np.zeros((n, 3))
    off[:, 0] = (idx % cols) * spacing
    off[:, 1] = -(idx // cols) * spacing
    off[:, :2] -= (off[:, :2].min(0) + off[:, :2].max(0)) / 2
    return off


def matrix(clips, spacing, row_gap=1.0):
    """Robot x motion layout: one row per robot (in load order), one column per motion name. Clips
    start in their cell and follow their real root trajectory (the "in place" toggle pins them).
    Returns (offsets (N,3), rows [(robot, y)], cols [(motion, x)]), centred on the origin."""
    robots = list(dict.fromkeys(c["robot"] for c in clips))
    motions = sorted({c["clip"] for c in clips})
    ri, mi = {r: k for k, r in enumerate(robots)}, {m: k for k, m in enumerate(motions)}
    dy = spacing + row_gap
    off = np.zeros((len(clips), 3))
    for i, c in enumerate(clips):
        off[i, 0] = mi[c["clip"]] * spacing
        off[i, 1] = -ri[c["robot"]] * dy
    centre = np.array([(len(motions) - 1) * spacing / 2, -(len(robots) - 1) * dy / 2, 0.0])
    off -= centre
    rows = [(r, -k * dy - centre[1]) for k, r in enumerate(robots)]
    cols = [(m, k * spacing - centre[0]) for k, m in enumerate(motions)]
    return off, rows, cols


def pack(clips, margin=1.0):
    """Shelf-pack each clip's root XY bounding box into a near-square layout. Returns (N,3) offsets."""
    boxes = []
    for c in clips:
        xy = c["q"][:, :2]
        lo, hi = xy.min(0) - margin, xy.max(0) + margin
        boxes.append((lo, hi - lo))
    total = sum(w * h for _, (w, h) in boxes)
    row_width = max(np.sqrt(total) * 1.2, max(w for _, (w, _) in boxes))
    order = sorted(range(len(clips)), key=lambda i: -boxes[i][1][1])  # tallest first
    offsets = np.zeros((len(clips), 3))
    x = y = row_h = 0.0
    for i in order:
        lo, (w, h) = boxes[i]
        if x + w > row_width and x > 0:
            x, y, row_h = 0.0, y + row_h, 0.0
        offsets[i, :2] = np.array([x, y]) - lo
        x += w
        row_h = max(row_h, h)
    # centre the whole layout on the origin
    offsets[:, :2] -= (offsets[:, :2].min(0) + offsets[:, :2].max(0)) / 2
    return offsets


class Crowd:
    """All clips' robots: one batched (instanced) mesh per robot body."""

    def __init__(self, server: viser.ViserServer, clips, offsets, prefix="/robots"):
        self.server, self.clips, self.offsets = server, clips, offsets
        self.kin = Kinematics(clips[0]["xml"])
        m = self.kin.model
        offs = clips[0].get("offsets")
        self.points = self.kin.point_set(clips[0]["links"] if offs is None else list(zip(clips[0]["links"], offs)))
        N = len(clips)
        per_body: dict[int, list] = {}
        visible = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and m.geom_rgba[g][3] > 0]
        meshes = [g for g in visible if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] <= 2]
        has_mesh = {int(m.geom_bodyid[g]) for g in meshes}
        for g in visible:   # meshes; primitives only for bodies without any visual mesh
            b = int(m.geom_bodyid[g])
            if g in meshes or (b not in has_mesh and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH):
                per_body.setdefault(b, []).append(g)
        self.handles, self.base_colors, kept = [], [], []
        for b in sorted(per_body):
            verts, faces, cols, off = [], [], [], 0
            for g in per_body[b]:
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = m.geom_dataid[g]
                    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                    f = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
                else:
                    vf = _primitive_mesh(mujoco.mjtGeom(int(m.geom_type[g])), m.geom_size[g])
                    if vf is None:
                        continue
                    v, f = vf
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, m.geom_quat[g])
                verts.append(v @ R.reshape(3, 3).T + m.geom_pos[g])  # geom frame -> body frame
                faces.append(f + off)
                off += len(v)
                cols.append((m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g])[:3])
            if not verts:
                continue
            color = (np.clip(np.mean(cols, axis=0), 0.15, 1.0) * 255).astype(np.uint8)
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) or f"body{b}"
            h = server.scene.add_batched_meshes_simple(
                f"{prefix}/{name}", np.concatenate(verts).astype(np.float32),
                np.concatenate(faces).astype(np.uint32),
                batched_wxyzs=np.tile([1.0, 0, 0, 0], (N, 1)), batched_positions=np.zeros((N, 3)),
                batched_colors=np.tile(color, (N, 1)))
            self.handles.append(h)
            self.base_colors.append(color)
            kept.append(b)
        self.bodies = np.array(kept, dtype=int)
        self.root = int(m.jnt_bodyid[0])
        self.in_place = False
        self.shift = np.zeros((N, 3))
        self.pos = np.zeros((len(self.bodies), N, 3))
        self.quat = np.zeros((len(self.bodies), N, 4))
        self.frames = np.zeros(N, dtype=int)
        self.root_pos = np.zeros((N, 3))

    def update(self, t, loop=True):
        """Pose every clip at time t (seconds). Returns per-clip frame indices."""
        d = self.kin.data
        for i, c in enumerate(self.clips):
            T = len(c["q"])
            f = int(t * c["fps"])
            f = f % T if loop else min(f, T - 1)
            self.frames[i] = f
            self.shift[i] = self.offsets[i]
            if self.in_place:  # remove the root's horizontal travel: the robot moves on the spot
                self.shift[i, :2] -= c["q"][f, :2]
            self.kin.forward(c["q"][f], with_jac=False)
            self.pos[:, i] = d.xpos[self.bodies] + self.shift[i]
            self.root_pos[i] = d.xpos[self.root] + self.shift[i]
            self.quat[:, i] = d.xquat[self.bodies]
        for h, p, w in zip(self.handles, self.pos, self.quat):
            h.batched_positions = p
            h.batched_wxyzs = w
        return self.frames

    def set_highlight(self, sel):
        for h, base in zip(self.handles, self.base_colors):
            cols = np.tile(base, (len(self.clips), 1))
            if sel is not None:
                cols[sel] = (0.35 * base + 0.65 * HIGHLIGHT).astype(np.uint8)
            h.batched_colors = cols

    def pick(self, origin, direction, radius=0.18):
        """Clip whose robot is hit by the click ray (nearest body to the ray, closest to camera)."""
        o, dvec = np.asarray(origin), np.asarray(direction)
        dvec = dvec / np.linalg.norm(dvec)
        v = self.pos - o                       # (B, N, 3)
        along = v @ dvec
        dist = np.linalg.norm(v - along[..., None] * dvec, axis=-1)
        dist = np.where(along > 0, dist, np.inf)
        hit = dist < radius
        if not hit.any():
            j = np.unravel_index(np.argmin(dist), dist.shape)
            return int(j[1]) if dist[j] < 2 * radius else None
        along = np.where(hit, along, np.inf)
        return int(np.unravel_index(np.argmin(along), along.shape)[1])

    def target_points(self, i):
        c = self.clips[i]
        f = self.frames[i]
        self.kin.forward(c["q"][f], with_jac=False)
        if c["X"] is None:                      # lazy: read this clip's targets on first selection
            c["X"] = np.load(c["file"])["targets"]
        return c["X"][f] + self.shift[i], self.kin.positions(self.points) + self.shift[i]


class MultiCrowd:
    """Several robots in one scene: one Crowd per robot model, global clip indices."""

    def __init__(self, server, clips, offsets):
        self.clips, self.offsets = clips, offsets
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(clips):
            groups.setdefault(c["xml"], []).append(i)
        self.parts = []
        for k, (xml, idx) in enumerate(groups.items()):
            cr = Crowd(server, [clips[i] for i in idx], offsets[idx], prefix=f"/robots/{k}_{clips[idx[0]]['robot']}")
            self.parts.append((cr, np.array(idx)))
        self.handles = [h for cr, _ in self.parts for h in cr.handles]
        self.frames = np.zeros(len(clips), dtype=int)
        self.root_pos = np.zeros((len(clips), 3))
        self._in_place = False

    @property
    def in_place(self):
        return self._in_place

    @in_place.setter
    def in_place(self, v):
        self._in_place = v
        for cr, _ in self.parts:
            cr.in_place = v

    def update(self, t, loop=True):
        for cr, idx in self.parts:
            cr.offsets = self.offsets[idx]
            self.frames[idx] = cr.update(t, loop)
            self.root_pos[idx] = cr.root_pos
        return self.frames

    def set_highlight(self, sel):
        for cr, idx in self.parts:
            hit = np.flatnonzero(idx == sel) if sel is not None else []
            cr.set_highlight(int(hit[0]) if len(hit) else None)

    def pick(self, origin, direction, radius=0.18):
        best, best_d = None, np.inf
        o, dvec = np.asarray(origin), np.asarray(direction) / np.linalg.norm(direction)
        for cr, idx in self.parts:
            j = cr.pick(origin, direction, radius)
            if j is not None:
                d = (cr.root_pos[j] - o) @ dvec
                if d < best_d:
                    best, best_d = int(idx[j]), d
        return best

    def target_points(self, i):
        for cr, idx in self.parts:
            hit = np.flatnonzero(idx == i)
            if len(hit):
                return cr.target_points(int(hit[0]))


MATRIX = "robot × motion"


def serve(path, port: int = 8080, limit: int | None = None, block: bool = True, spacing: float = 1.5,
          labels: bool = False, clip: str | None = None, layout_mode: str = "auto"):
    """``layout_mode``: "grid", "packed" or "matrix" (rows = robots, columns = motions, each robot in
    place in its cell). "auto" picks the matrix when several robots each show several motions."""
    lazy = clip is None
    clips = load_clips(path, limit, clip, lazy=lazy)
    if not clips:
        raise SystemExit(f"no .npz motions found in {path}")
    robots = {c["robot"] for c in clips}
    if layout_mode == "auto":
        layout_mode = "matrix" if len(robots) > 1 and len(clips) > len(robots) else "grid"
    big = layout_mode == "matrix"
    if big:
        spacing = max(spacing, 1.8)
        labels = False                           # row/column labels instead of one per robot
    server = viser.ViserServer(port=port)
    crowd = MultiCrowd(server, clips, grid(len(clips), spacing))
    ground = {"h": None}
    axes = []                                    # matrix row / column labels

    def set_layout(mode, spacing_m):
        # Grid: every clip starts on its own grid cell and follows its real trajectory, so robots may
        # walk through each other. Packed: clips get non-overlapping patches sized to their paths.
        # Matrix: one row per robot, one column per motion.
        for h in axes:
            h.remove()
        axes.clear()
        if mode == MATRIX:
            crowd.offsets, rows, cols = matrix(clips, spacing_m)
            x0 = min(x for _, x in cols) - spacing_m
            y0 = max(y for _, y in rows) + spacing_m
            for k, (r, y) in enumerate(rows):
                axes.append(server.scene.add_label(f"/axes/row{k}", r, position=(x0, y, 0.3),
                                                   anchor="center-right", font_size_mode="scene",
                                                   font_scene_height=0.35))
            for k, (m, x) in enumerate(cols):
                yk = y0 + (0.7 if k % 2 else 0.0)   # staggered: long names don't overlap
                axes.append(server.scene.add_label(f"/axes/col{k}", m, position=(x, yk, 0.05),
                                                   anchor="bottom-center", font_size_mode="scene",
                                                   font_scene_height=0.2))
        else:
            crowd.offsets = grid(len(clips), spacing_m) if mode == "grid" else pack(clips)
        span = np.ptp(crowd.offsets[:, :2], axis=0).max() + 20
        if ground["h"] is not None:
            ground["h"].remove()
        ground["h"] = server.scene.add_grid("/ground", width=span, height=span, cell_size=1.0)

    set_layout(MATRIX if big else layout_mode if layout_mode == "grid" else "packed (no overlap)", spacing)
    if big:
        w, h = np.ptp(crowd.offsets[:, :2], axis=0)
        server.initial_camera.look_at = (0.0, 0.0, 0.0)
        server.initial_camera.position = (0.0, -0.75 * h - 0.45 * w, 0.45 * w + 0.5 * h)
    names = [c["name"] for c in clips]
    duration = max(len(c["q"]) / c["fps"] for c in clips)

    st = dict(t=0.0, playing=True, sel=None, syncing=False)
    lock = threading.Lock()

    with server.gui.add_folder("Selection"):
        info = server.gui.add_markdown("Click a robot to identify its motion.")
        dd = server.gui.add_dropdown("motion", ("(none)", *names), initial_value="(none)")
        goto = server.gui.add_button("Go to selected")
    with server.gui.add_folder("Playback"):
        play = server.gui.add_button("Pause")
        speed = server.gui.add_slider("speed", min=0.1, max=4.0, step=0.1, initial_value=1.0)
        scrub = server.gui.add_slider("time [s]", min=0.0, max=round(duration, 1), step=0.1, initial_value=0.0)
        loop = server.gui.add_checkbox("loop each clip", True)
    with server.gui.add_folder("Layout"):
        modes = ("grid", "packed (no overlap)") + ((MATRIX,) if len(robots) > 1 else ())
        layout = server.gui.add_dropdown("layout", modes, initial_value=MATRIX if big else modes[
            0 if layout_mode == "grid" else 1])
        spacing_s = server.gui.add_slider("spacing [m]", min=0.5, max=5.0, step=0.1, initial_value=spacing)
        in_place = server.gui.add_checkbox("in place (remove root travel)", False)
    with server.gui.add_folder("Display"):
        show_labels = server.gui.add_checkbox("all motion names", labels)
        server.gui.add_markdown(f"{len(clips)} motions on {len(robots)} robot(s), longest {duration:.0f} s")

    tgt = server.scene.add_point_cloud("/selected/targets", np.zeros((max(len(c['links']) for c in clips), 3), np.float32),
                                       colors=(230, 60, 60), point_size=0.03, point_shape="circle", visible=False)
    err = server.scene.add_line_segments("/selected/error", np.zeros((max(len(c['links']) for c in clips), 2, 3), np.float32),
                                         colors=(255, 200, 0), thickness=0.004, visible=False)
    sel_label = server.scene.add_label("/selected/label", "", visible=False, anchor="bottom-center")
    labels = [server.scene.add_label(f"/labels/{n}", n, visible=False, anchor="bottom-center",
                                     font_screen_scale=0.7) for n in names]

    @layout.on_update
    def _(_):
        set_layout(layout.value, spacing_s.value)

    @spacing_s.on_update
    def _(_):
        if layout.value in ("grid", MATRIX):
            set_layout(layout.value, spacing_s.value)

    @in_place.on_update
    def _(_):
        crowd.in_place = in_place.value

    def select(i):
        with lock:
            st["sel"] = i
        crowd.set_highlight(i)
        for h in (tgt, err, sel_label):
            h.visible = i is not None
        if i is None:
            info.content = "Click a robot to identify its motion."
        st["syncing"] = True
        dd.value = "(none)" if i is None else names[i]
        st["syncing"] = False

    def on_click(event):
        i = crowd.pick(event.ray_origin, event.ray_direction)
        select(i if i != st["sel"] else None)  # click the selected robot again to clear

    for h in crowd.handles:
        h.on_click(on_click)

    @dd.on_update
    def _(_):
        if not st["syncing"]:
            select(None if dd.value == "(none)" else names.index(dd.value))

    @goto.on_click
    def _(event):
        i = st["sel"]
        if i is None:
            return
        p = crowd.root_pos[i]
        event.client.camera.look_at = tuple(p)
        event.client.camera.position = tuple(p + np.array([2.5, -2.5, 1.5]))

    @play.on_click
    def _(_):
        st["playing"] = not st["playing"]
        play.label = "Pause" if st["playing"] else "Play"

    @scrub.on_update
    def _(_):
        if not st["syncing"]:
            with lock:
                st["t"] = float(scrub.value)

    print(f"viewer: http://localhost:{port}  ({len(clips)} motions in one scene)")

    def tick():
        last = time.perf_counter()
        n_tick = 0
        while True:
            now = time.perf_counter()
            dt, last = now - last, now
            with lock:
                if st["playing"]:
                    st["t"] += dt * speed.value
                    if st["t"] > duration:
                        st["t"] = 0.0 if loop.value else duration
                t, sel = st["t"], st["sel"]
            with server.atomic():
                frames = crowd.update(t, loop.value)
                head = crowd.root_pos + np.array([0.0, 0.0, 0.9])  # above the pelvis
                if sel is not None:
                    X, P = crowd.target_points(sel)
                    tgt.points = X.astype(np.float32)
                    err.points = np.stack([X, P], axis=1).astype(np.float32)
                    sel_label.text = names[sel]
                    sel_label.position = tuple(head[sel])
                for i, lab in enumerate(labels):
                    lab.visible = show_labels.value
                    if show_labels.value:
                        lab.position = tuple(head[i])
            if sel is not None and n_tick % 6 == 0:
                c = clips[sel]
                info.content = (f"**{names[sel]}**  \nframe {frames[sel]} / {len(c['q'])} "
                                f"({frames[sel] / c['fps']:.1f} / {len(c['q']) / c['fps']:.1f} s)")
            if n_tick % 6 == 0:
                st["syncing"] = True
                scrub.value = round(t, 1)
                st["syncing"] = False
            n_tick += 1
            time.sleep(max(0.0, 1.0 / 30.0 - (time.perf_counter() - now)))

    if block:
        tick()
    threading.Thread(target=tick, daemon=True).start()
    return server


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="+", help="folders of retargeted .npz files and/or .npz files")
    ap.add_argument("--clip", default=None, help="only this motion (searched in each folder and its sub-folders: "
                                                 "one robot per sub-folder shows every robot doing it)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--limit", type=int, default=None, help="only the first N motions")
    ap.add_argument("--spacing", type=float, default=1.5, help="grid spacing in metres")
    ap.add_argument("--labels", action="store_true", help="show every motion's name above its robot")
    args = ap.parse_args()
    serve(args.path, args.port, args.limit, spacing=args.spacing, labels=args.labels, clip=args.clip)


if __name__ == "__main__":
    main()
