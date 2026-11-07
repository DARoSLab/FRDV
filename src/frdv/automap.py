"""Automatic robot setup: find limbs and joints from the kinematic tree (names are not needed).

Given any floating-base humanoid model, ``analyze`` returns the canonical keypoints frdv tracks
(pelvis, spine, hips, knees, ankles, toes, shoulders, elbows, wrists), orientation-marker links,
sole contact points and the standing height, using only the tree structure and the geometry at a
nominal pose:

* legs  = the root->leaf paths to the two lowest leaves on opposite sides (y > 0 is left);
* arms  = the remaining branches with the most joints and the largest lateral reach, one per side;
          a hand that branches into fingers ends at the branching body;
* joint clusters: consecutive joint bodies whose origins coincide (within ``cluster_tol``) form
  a hip / ankle / shoulder / wrist; the knee / elbow is the first joint body clearly between them;
* toe and sole points come from the foot geometry (front of the foot, four sole corners).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass
class Limb:
    side: str                      # "left" / "right"
    path: list                     # body ids from the limb root (first joint body) to its end
    clusters: list                 # [[body ids], ...] co-located joint bodies along the path


@dataclass
class RobotLayout:
    keypoints: dict = field(default_factory=dict)      # canonical -> (body name, offset (3,))
    markers: dict = field(default_factory=dict)        # canonical marker -> body name (offset calibrated)
    foot_points: list = field(default_factory=list)    # [(body name, offset (3,)), ...]
    height: float = 0.0
    nominal_key: str | None = None
    notes: list = field(default_factory=list)


def _name(m, b):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"


def _name_side(name: str):
    """'left' / 'right' / None from a link name (left_hip, l_knee, hip_roll_l_link, LeftArm, L_foot...)."""
    import re
    toks = [t for t in re.split(r"[^a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower()) if t]
    lt = any(t in ("l", "left", "lf", "lh") or t.startswith("left") for t in toks)
    rt = any(t in ("r", "right", "rf", "rh") or t.startswith("right") for t in toks)
    return "left" if lt and not rt else "right" if rt and not lt else None


def _nominal(m):
    keys = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, k) or "" for k in range(m.nkey)]
    for pref in ("t_pose", "tpose", "stand", "home", "default", "init"):
        for k, n in enumerate(keys):
            if pref in n.lower().replace("-", "_"):
                return n, m.key_qpos[k].copy()
    return None, m.qpos0.copy()


def _geom_box(m, d, bodies):
    """World AABB (lo, hi) of all geoms of ``bodies`` (meshes via their local AABB)."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) not in bodies:
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        c, e = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * e + c
        w = d.geom_xpos[g] + corners @ R.T
        lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
    return lo, hi


def analyze(m: mujoco.MjModel, cluster_tol: float = 0.06) -> RobotLayout:
    out = RobotLayout()
    out.nominal_key, q = _nominal(m)
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    pos = d.xpos
    root = int(m.jnt_bodyid[0])
    children = {b: [] for b in range(m.nbody)}
    for b in range(1, m.nbody):
        children[int(m.body_parentid[b])].append(b)

    def path_to(b):
        p = []
        while b != root and b != 0:
            p.append(b)
            b = int(m.body_parentid[b])
        return [root] + p[::-1]

    def subtree(b):
        st, res = [b], []
        while st:
            x = st.pop()
            res.append(x)
            st.extend(children[x])
        return res

    def common_prefix(a, b):
        c = []
        for x, y in zip(a, b):
            if x != y:
                break
            c.append(x)
        return c

    has_joint = lambda b: m.body_jntnum[b] > 0                       # noqa: E731
    njoints = lambda p: sum(int(m.body_jntnum[b]) for b in p)         # noqa: E731
    desc = subtree(root)
    leaves = [b for b in desc if not children[b] and b != root]
    # drop jointless side branches (linkage rods, motor housings, sensor frames hanging off a limb):
    # a leaf whose joint sequence is a strict prefix of another leaf's is not a limb end
    jseq = {b: tuple(x for x in path_to(b) if has_joint(x)) for b in leaves}
    longer = {s[:k] for s in jseq.values() for k in range(len(s))}
    leaves = [b for b in leaves if jseq[b] not in longer]
    up =np.array([0.0, 0.0, 1.0])

    # ---- legs: the two lowest feet, as far apart as possible ----------------------------------
    def foot_low(b):
        lo, _ = _geom_box(m, d, set(subtree(_last_joint_body(m, path_to(b)))))
        return lo[2] if np.isfinite(lo[2]) else pos[b, 2]
    legc = [b for b in leaves if njoints(path_to(b)) >= 3]
    if len(legc) < 2:
        out.notes.append("could not find two legs")
        return out
    zf = {b: foot_low(b) for b in legc}
    zmin = min(zf.values())
    low = [b for b in legc if zf[b] < zmin + 0.05]
    pair = max(((a, b) for i, a in enumerate(low) for b in low[i + 1:]),
               key=lambda ab: np.linalg.norm((pos[ab[0]] - pos[ab[1]])[:2]), default=None)
    if pair is None:
        out.notes.append("could not find two legs")
        return out
    # left/right from link names when they are unambiguous, else from the feet geometry
    def vote(leaf):
        v = [_name_side(_name(m, b)) for b in path_to(leaf)]
        nl, nr = v.count("left"), v.count("right")
        return "left" if nl > nr else "right" if nr > nl else None
    named = {vote(b): b for b in pair}
    fwd = np.zeros(3)
    if set(named) == {"left", "right"}:
        ld = pos[named["left"]] - pos[named["right"]]
        ld[2] = 0.0
        if np.linalg.norm(ld) > 1e-3:
            ld /= np.linalg.norm(ld)
            fwd = np.cross(ld, up)
    for leaf in pair if not fwd.any() else ():
        foot = _last_joint_body(m, path_to(leaf))
        lo, hi = _geom_box(m, d, set(subtree(foot)))
        if np.all(np.isfinite(lo)):
            c = 0.5 * (lo + hi)
            fwd += np.r_[(c - pos[foot])[:2], 0.0]
    if np.linalg.norm(fwd) < 1e-3:
        fwd = np.array([1.0, 0.0, 0.0])
        out.notes.append("forward direction from the feet is ambiguous; assuming +x")
    fwd /= np.linalg.norm(fwd)
    left_dir = np.cross(up, fwd)
    side_of = lambda b: "left" if (pos[b] - pos[root]) @ left_dir > 0 else "right"   # noqa: E731
    leg = {side_of(b): b for b in pair}
    if len(leg) < 2:
        out.notes.append("both legs on the same side?")
        return out
    leg_div = common_prefix(path_to(leg["left"]), path_to(leg["right"]))
    leg_used = set(path_to(leg["left"])[len(leg_div):]) | set(path_to(leg["right"])[len(leg_div):])

    # ---- arms: outermost multi-joint branches that are not legs ---------------------------------
    arm = {}
    for side in ("left", "right"):
        sgn = 1 if side == "left" else -1
        cand = [b for b in leaves if not set(path_to(b)) & leg_used
                and sgn * ((pos[b] - pos[root]) @ left_dir) > 0.05 and njoints(path_to(b)) >= 3]
        if cand:
            arm[side] = max(cand, key=lambda b: (njoints(path_to(b)), len(path_to(b)), sgn *((pos[b] - pos[root]) @ left_dir)))
        else:
            out.notes.append(f"no {side} arm found")

    def clusters(p):
        cl = []
        for b in p:
            if not has_joint(b):
                continue
            if cl and np.linalg.norm(pos[b] - pos[cl[-1][0]]) < cluster_tol:
                cl[-1].append(b)
            else:
                cl.append([b])
        return cl

    def axes(c):
        out_ = []
        for b in c:
            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                out_.append(d.xmat[b].reshape(3, 3) @ m.jnt_axis[j])
        return out_

    def mid_joint(cl, end, lo_i, hi_i, limb_dir):
        """index of the bending cluster closest to half the chain length (knee / elbow)."""
        pts = [pos[c[0]] for c in cl] + [pos[end]]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        cand = [i for i in range(lo_i, hi_i) if bends(cl[i], bone_dir(cl, i, end, limb_dir))]
        return min(cand, key=lambda i: abs(arc[i] - 0.5 * arc[-1])) if cand else None

    def bone_dir(cl, i, end, fallback):
        """direction of the bone leaving cluster i (to the next cluster, or to the chain end).
        Roll joints (shoulder yaw, wrist roll) turn about this bone; knees and elbows don't."""
        nxt = pos[cl[i + 1][0]] if i + 1 < len(cl) else pos[end]
        v = nxt - pos[cl[i][0]]
        n = np.linalg.norm(v)
        return v / n if n > 0.02 else fallback

    def bends(c, limb_dir):          # cluster contains a joint that bends the limb (axis not along it)
        return any(abs(a @ limb_dir) < 0.5 for a in axes(c))

    # ---- pelvis / spine ------------------------------------------------------------------------
    hip_anchor = leg_div[-1]

    # ---- legs ----------------------------------------------------------------------------------
    hips = {}
    for side, leaf in leg.items():
        p = path_to(leaf)[len(leg_div):]
        cl = clusters(p)
        if len(cl) < 3:
            out.notes.append(f"{side} leg: only {len(cl)} joint cluster(s)")
            continue
        foot = _last_joint_body(m, path_to(leaf))
        limb = pos[foot] - pos[cl[0][0]]
        limb /= max(np.linalg.norm(limb), 1e-9)
        knee_i = mid_joint(cl, foot, 1, len(cl) - 1, limb)
        if knee_i is None:
            out.notes.append(f"{side} leg: no knee found")
            continue
        hip_group = cl[:knee_i]
        hip = next((c for c in reversed(hip_group) if bends(c, limb)), hip_group[-1])
        ankle_group = cl[knee_i + 1:] or [cl[-1]]
        ankle = next((c for c in ankle_group if bends(c, limb)), ankle_group[0])
        hips[side] = hip
        out.keypoints[f"{side}_hip"] = (_name(m, hip[-1]), np.zeros(3))
        out.keypoints[f"{side}_knee"] = (_name(m, cl[knee_i][0]), np.zeros(3))
        out.keypoints[f"{side}_ankle"] = (_name(m, ankle[0]), np.zeros(3))
        lo, hi = _geom_box(m, d, set(subtree(foot)))
        R = d.xmat[foot].reshape(3, 3)
        if np.all(np.isfinite(lo)):
            a = pos[ankle[0]]
            # foot frame in the horizontal plane: forward = fwd, lateral = left_dir
            ext_f = max((np.array([[x, y, lo[2]] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]) - a) @ fwd)
            ext_b = min((np.array([[x, y, lo[2]] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]) - a) @ fwd)
            ext_l = max((np.array([[x, y, lo[2]] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]) - a) @ left_dir)
            ext_r = min((np.array([[x, y, lo[2]] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]) - a) @ left_dir)
            sole = np.array([a[0], a[1], lo[2]])
            toe = sole + 0.75 * ext_f * fwd + 0.02 * up
            out.keypoints[f"{side}_toe"] = (_name(m, foot), R.T @ (toe - pos[foot]))
            for fx in (ext_b * 0.8, ext_f * 0.85):
                for fy in (ext_r * 0.7, ext_l * 0.7):
                    w = sole + fx * fwd + fy * left_dir
                    out.foot_points.append((_name(m, foot), R.T @ (w - pos[foot])))
        else:
            out.notes.append(f"{side} foot has no geometry: no toe / sole points")

    if len(hips) == 2:
        mid = 0.5 * (pos[hips["left"][-1]] + pos[hips["right"][-1]])
        Ra = d.xmat[hip_anchor].reshape(3, 3)
        out.keypoints["pelvis"] = (_name(m, hip_anchor), Ra.T @ (mid - pos[hip_anchor]))
    else:
        out.keypoints["pelvis"] = (_name(m, root), np.zeros(3))

    # ---- arms ----------------------------------------------------------------------------------
    arm_div = common_prefix(path_to(arm["left"]), path_to(arm["right"])) if len(arm) == 2 else []
    shoulders = {}
    for side, leaf in arm.items():
        full = path_to(leaf)
        own = full[len(arm_div):] if arm_div else [b for b in full if b not in leg_div]
        # hand end: first body (on this arm's own part) where it branches into >1 jointed sub-chains
        hand_end = own[-1]
        for b in own:
            if len([c for c in children[b] if njoints(subtree(c)) > 0]) > 1:
                hand_end = b
                break
        own = own[: own.index(hand_end) + 1]
        # drop a jointless, geometry-less tail (marker / site bodies past the hand)
        while (len(own) > 2 and not has_joint(own[-1]) and m.body_geomnum[own[-1]] == 0
               and not has_joint(own[-2])):     # ... but keep the first body past the last joint
            own = own[:-1]
        hand_end = own[-1]
        cl = clusters(own)
        if len(cl) < 2:
            out.notes.append(f"{side} arm: only {len(cl)} joint cluster(s)")
            continue
        limb = pos[hand_end] - pos[cl[0][0]]
        if np.linalg.norm(limb) < 0.1:
            limb = pos[own[-1]] - pos[cl[0][0]]
        limb /= max(np.linalg.norm(limb), 1e-9)
        elbow_i = mid_joint(cl, hand_end, 1, len(cl), limb)
        if elbow_i is None:
            out.notes.append(f"{side} arm: no elbow found")
            continue
        sh_group = cl[:elbow_i]
        shoulder = next((c for c in sh_group if bends(c, limb)), sh_group[0])
        shoulders[side] = shoulder
        wrist_group = cl[elbow_i + 1:]
        wrist = wrist_group[-1] if wrist_group else None
        out.keypoints[f"{side}_shoulder"] = (_name(m, shoulder[0]), np.zeros(3))
        out.keypoints[f"{side}_elbow"] = (_name(m, cl[elbow_i][0]), np.zeros(3))
        out.keypoints[f"{side}_wrist"] = (_name(m, wrist[0] if wrist else hand_end), np.zeros(3))
        hand_link = (wrist[-1] if wrist else hand_end)
        for mk in ("thumb", "middle", "pinky"):
            out.markers[f"{side}_{mk}"] = _name(m, hand_link)
        for mk in ("upper_arm_up", "upper_arm_down"):
            out.markers[f"{side}_{mk}"] = _name(m, sh_group[-1][-1])
        roll = next((c for c in wrist_group if not bends(c, limb)), None) if wrist_group else None
        fl = roll[-1] if roll else (wrist_group[0][0] if wrist_group else cl[elbow_i][-1])
        for mk in ("forearm_up", "forearm_down"):
            out.markers[f"{side}_{mk}"] = _name(m, fl)

    # spine: a quarter of the way from the hip midpoint to the shoulder midpoint, on the upper torso
    if len(hips) == 2 and len(shoulders) == 2 and arm_div:
        hm = 0.5 * (pos[hips["left"][-1]] + pos[hips["right"][-1]])
        sm = 0.5 * (pos[shoulders["left"][0]] + pos[shoulders["right"][0]])
        torso = arm_div[-1]
        waist = [b for b in arm_div if b not in leg_div and has_joint(b)]
        anchor = waist[0] if waist else torso
        Rt = d.xmat[anchor].reshape(3, 3)
        out.keypoints["spine"] = (_name(m, anchor), Rt.T @ (hm + 0.25 * (sm - hm) - pos[anchor]))

    lo, hi = _geom_box(m, d, set(desc))
    out.height = float(hi[2] - lo[2]) if np.all(np.isfinite(lo)) else 0.0
    out.forward = fwd
    return out


def _last_joint_body(m, path):
    jb = [b for b in path if m.body_jntnum[b] > 0]
    return jb[-1] if jb else path[-1]
