"""Per-frame kinematic retargeting (IK) problem and sequential solver.

Cost:

    0.5 * sum_s w_s ||p_s(q) - target_s||^2          keypoint tracking
  + 0.5 * sum_j r_j (q_j - q_nom_j)^2                joint regularisation
  + 0.5 * w_vel ||q (-) q_prev||^2                    velocity smoothing
  + 0.5 * w_acc ||q (-) (q_prev (+) (q_prev (-) q_prev2))||^2   acceleration smoothing

The two smoothing terms are both identity-Jacobian quadratics, so they are folded into one:
weight w_vel + w_acc, centred at q_prev advanced by w_acc/(w_vel+w_acc) of the last velocity.

Constraints:  joint limits, foot points above ground, capsule endpoints above
ground, optional capsule-capsule self-collision distance >= margin.

The regularisation and smoothing terms have identity Jacobians, so they are
added to H and g analytically instead of being stacked into the residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import _jit
from .kinematics import Kinematics, PointSet, segment_closest_points, shift_jacobian
from .sqp import Linearization, SQPInfo, SQPOptions, sqp_solve
from .validation import ValidationError, pose_array, qp_succeeded, targets_array, trajectory_array


@dataclass
class IKConfig:
    xml_path: str
    targets: list[str]                       # site/body names tracked, in target-array order
    target_weights: list[float] | None = None
    foot_points: list[str] = field(default_factory=list)
    q_nominal: np.ndarray | None = None      # MuJoCo qpos; default: keyframe 't_pose' or qpos0
    joint_reg_weights: np.ndarray | float = 1.0
    smoothing_weight: float = 10.0
    # Penalises deviation from constant-velocity motion (q_{t-1} + (q_{t-1} - q_{t-2})). Damps
    # one-frame flicks (contacts, joint limits) without slowing down sustained fast motion.
    accel_weight: float = 0.0
    # Scale of the rotational smoothing term. 0.25 makes it equal to penalising raw quaternion
    # component differences (||dquat||^2 ~= ||dtheta||^2 / 4).
    rot_smoothing_scale: float = 0.25
    ground_height: float = 0.0
    capsule_ground: bool = True
    self_collision_pairs: list[tuple[str, str]] = field(default_factory=list)
    self_collision_margin: float = 0.02
    hessian_reg: float = 1e-8
    # Only constraints whose value is below this margin enter the QP (the merit function
    # still checks all of them). None = always include every constraint.
    constraint_activation: float | None = 0.10
    use_jit: bool | None = None  # None: use numba kernels if installed
    # Max change of any joint between consecutive frames (rad/frame; None = unlimited). Set it
    # from a joint velocity limit: max_joint_velocity / fps. Prevents one-frame "snaps" when the
    # solution switches branch (e.g. a joint leaving a limit it was pinned against).
    max_joint_step: float | None = None
    joint_limits: dict | None = None  # {joint name: (lo, hi)} overriding the model's limits

    # Solver schedule
    sqp: SQPOptions = field(default_factory=lambda: SQPOptions(max_iter=1))
    warmup_iters: int = 200          # proximal GN iterations on the first frame of a cold start
    warmup_tol: float = 1e-5
    boost_frames: int = 20           # first frames after a cold start get more iterations
    boost_iters: int = 10


_EMPTY_I = np.zeros(0, dtype=np.int64)


class IKProblem:
    """Holds per-frame data (targets, q_prev) and provides the SQP oracle."""

    def __init__(self, cfg: IKConfig, kin: Kinematics | None = None):
        self.cfg = cfg
        self.kin = kin if kin is not None else Kinematics(cfg.xml_path)
        k = self.kin
        m = k.model
        if cfg.joint_limits:
            k.q_min, k.q_max = k.q_min.copy(), k.q_max.copy()
            for name, (lo, hi) in cfg.joint_limits.items():
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                if j < 0:
                    raise KeyError(f"joint_limits: no joint named '{name}'")
                a = m.jnt_qposadr[j] - 7
                k.q_min[a], k.q_max[a] = lo, hi
        nv = k.nv
        self.nv = nv

        if cfg.q_nominal is not None:
            self.q_nominal = pose_array(cfg.q_nominal, k.nq).copy()
        else:
            kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "t_pose")
            self.q_nominal = (m.key_qpos[kid] if kid >= 0 else m.qpos0).copy()

        # ---- points: targets | feet | capsule endpoints ------------------------
        tgt = k.point_set(cfg.targets)
        feet = k.point_set(cfg.foot_points) if cfg.foot_points else PointSet([], np.zeros((0, 3)))
        caps = k.capsules() if cfg.capsule_ground else []
        cap_b = [c.body for c in caps for _ in (0, 1)]
        cap_p = [p for c in caps for p in (c.p1, c.p2)]
        self.cap_radius = np.array([c.radius for c in caps for _ in (0, 1)])

        self.S, self.F, self.Kc = len(tgt.body_ids), len(feet.body_ids), len(cap_b)
        all_b = np.concatenate([tgt.body_ids, feet.body_ids, np.asarray(cap_b, dtype=np.int64)])
        all_p = np.concatenate([tgt.local_pos, feet.local_pos, np.asarray(cap_p).reshape(-1, 3)])

        # ---- self collision -----------------------------------------------------
        self.pairs = []
        if cfg.self_collision_pairs:
            by_body: dict[str, list] = {}
            for c in k.capsules():
                by_body.setdefault(c.name, []).append(c)
            seen = set()
            for a, b in cfg.self_collision_pairs:
                key = frozenset((a, b))
                if a == b or key in seen:  # duplicates would only add dependent QP rows
                    continue
                seen.add(key)
                for ca in by_body.get(a, []):
                    for cb in by_body.get(b, []):
                        self.pairs.append((ca, cb))
        P = len(self.pairs)
        if P:
            self.pa_body = np.array([a.body for a, _ in self.pairs])
            self.pb_body = np.array([b.body for _, b in self.pairs])
            self.pa_loc = np.array([[a.p1, a.p2] for a, _ in self.pairs])  # (P,2,3)
            self.pb_loc = np.array([[b.p1, b.p2] for _, b in self.pairs])
            self.pair_r = np.array([a.radius + b.radius for a, b in self.pairs])
            all_b_sc = np.concatenate([self.pa_body, self.pb_body])
        else:
            all_b_sc = np.zeros(0, dtype=np.int64)

        self.uniq, inv = np.unique(np.concatenate([all_b, all_b_sc]), return_inverse=True)
        self.pt_map = inv[: len(all_b)]
        self.sc_map = inv[len(all_b):]
        self.pts = PointSet(all_b, all_p)

        # ---- weights -------------------------------------------------------------
        w = np.ones(self.S) if cfg.target_weights is None else np.asarray(cfg.target_weights, float)
        if w.shape != (self.S,) or not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0):
            raise ValidationError(f"Target weights must contain {self.S} finite nonnegative values, "
                                  "with at least one positive weight")
        self.w_tgt = w
        self.w_tgt3 = np.repeat(w, 3)

        reg = np.broadcast_to(np.asarray(cfg.joint_reg_weights, float), (nv - 6,)).copy()
        self.w_reg = reg
        ws = np.full(nv, float(cfg.smoothing_weight) + float(cfg.accel_weight))
        self._alpha = cfg.accel_weight / max(cfg.smoothing_weight + cfg.accel_weight, 1e-12)
        ws[3:6] *= cfg.rot_smoothing_scale
        self.w_smooth = ws
        self.D = ws.copy()
        self.D[6:] += reg
        self.D += cfg.hessian_reg
        self.bidx = np.arange(6, nv)

        self.n_ground = self.F + self.Kc
        self.n_cons = self.n_ground + len(self.pairs)   # global constraint ids, used for warm starts
        self._Jt = np.zeros((self.S, 3, nv))
        self._tb = [int(b) for b in tgt.body_ids]
        self._Jp_u = np.zeros((len(self.uniq), 3, nv))
        self._Jr_u = np.zeros((len(self.uniq), 3, nv))
        self._cb = all_b[self.S:].astype(np.int64)
        self._jit = _jit.AVAILABLE if cfg.use_jit is None else (cfg.use_jit and _jit.AVAILABLE)
        # Dof ancestor chain of every point's body (same walk as mj_jac), for the fused JIT kernel.
        chains = [_dof_chain(m, int(b)) for b in all_b]
        L = max([len(ch) for ch in chains] + [1])
        self._chain = np.zeros((len(all_b), L), dtype=np.int64)
        self._chain_len = np.array([len(ch) for ch in chains], dtype=np.int64)
        for i, ch in enumerate(chains):
            self._chain[i, : len(ch)] = ch
        self._root = m.body_rootid[all_b].astype(np.int64)
        self._Cg = np.zeros((self.n_ground, nv))
        self._pb = np.ascontiguousarray(all_b, dtype=np.int64)
        self._pl = np.ascontiguousarray(all_p, dtype=float)
        self._pos = np.zeros((len(all_b), 3))
        self._c = np.zeros(self.n_ground)

        # Fused per-frame path (numba, no self-collision pairs): preallocated buffers, see solve_fast
        self.fast = bool(self._jit and not self.pairs)
        nj = nv - 6
        self._H, self._g = np.empty((nv, nv)), np.empty(nv)
        self._s, self._e = np.empty(nv), np.empty(nj)
        self._lb, self._ub = np.empty(nj), np.empty(nj)
        self._dneg = np.empty(self.n_ground)
        ng = self.n_ground
        self._Csub, self._dsub = np.empty((ng, nv)), np.empty(ng)
        self._idx, self._local_of = np.empty(ng, np.int64), np.empty(ng, np.int64)
        self._a0 = np.empty(ng + 2 * (nv - 6) + 1, np.int64)
        self._actg = np.empty(nv + 1, np.int64)
        self._qt = np.empty(k.nq)
        self._bidx64 = self.bidx.astype(np.int64)
        self._Jt2 = self._Jt.reshape(-1, nv)

        # Per-frame data
        self.targets = np.zeros((self.S, 3))
        self.q_prev = self.q_nominal.copy()
        self.q_ref = self.q_prev    # smoothing centre (q_prev advanced along the last velocity)

    # --------------------------------------------------------------- helpers
    def set_frame(self, targets, q_prev, q_prev2=None):
        self.targets = np.asarray(targets, dtype=float).reshape(self.S, 3)
        self.q_prev = q_prev
        if q_prev2 is None or self._alpha == 0.0:
            self.q_ref = q_prev
        else:
            self.q_ref = self.integrate(q_prev, self._alpha * self.difference(q_prev2, q_prev))

    def integrate(self, q, dv):
        out = np.empty_like(q) if self._jit else q.copy()
        if self._jit:
            _jit.qint(q, dv, 1.0, out)
        else:
            mujoco.mj_integratePos(self.kin.model, out, dv, 1.0)
        return out

    def difference(self, q0, q1):
        """Tangent vector v with q1 = integrate(q0, v)."""
        v = np.empty(self.nv)
        if self._jit:
            _jit.qdiff(q0, q1, v)
        else:
            mujoco.mj_differentiatePos(self.kin.model, v, 1.0, q0, q1)
        return v

    def solve_fast(self, q, opts, info):
        """``sqp_solve`` for the fused case: the same Gauss-Newton step, QP and L1-merit line
        search, with every array preallocated and one compiled call per stage."""
        m, d = self.kin.model, self.kin.data
        cfg = self.cfg
        cap = -1.0 if cfg.max_joint_step is None else float(cfg.max_joint_step)
        H, g, c, C, s, e = self._H, self._g, self._c, self._Cg, self._s, self._e
        h = cfg.ground_height
        for _ in range(opts.max_iter):
            info.iters += 1
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            _jit.step_terms(q, self.q_ref, self.q_nominal, self.q_prev, self.kin.q_min, self.kin.q_max, cap,
                            s, e, self._lb, self._ub)
            cost = _jit.linearize_all(
                d.xpos, d.xmat, d.cdof, d.subtree_com, self._pb, self._pl, self._root, self._chain,
                self._chain_len, self.S, self.F, self.cap_radius, h, self.targets, self.w_tgt3, self.D,
                self.w_smooth, s, self.w_reg, e, self._pos, self._Jt2, H, g, c, C)
            viol = _jit.neg_viol(c, self._dneg)
            ga = getattr(self, "qp_active", None)
            ga = _EMPTY_I if ga is None or len(ga) == 0 else np.asarray(ga, dtype=np.int64)
            act_lim = cfg.constraint_activation
            dv = None
            if act_lim is not None and c.size:
                # QP on the rows near the floor only; exact: accepted only if the pruned rows hold too
                k, n0 = _jit.select_rows(c, C, float(act_lim), ga, self._Csub, self._dsub, self._idx,
                                         self._local_of, self._a0)
                dv, status, iters, act, na = _jit.gi_qp(H, g, self._Csub[:k], self._dsub[:k], self._bidx64,
                                                        self._lb, self._ub, self._a0[:n0], opts.qp_tol, 200)
                if status == 0 and _jit.pruned_ok(c, C, float(act_lim), dv, 1e-9):
                    self.qp_active = _jit.to_global(act, na, k, self._idx, c.size, self._actg)
                else:
                    info.qp_iters += int(iters)
                    dv = None
            if dv is None:
                dv, status, iters, act, na = _jit.gi_qp(H, g, C, self._dneg, self._bidx64, self._lb, self._ub,
                                                        ga, opts.qp_tol, 200)
                self.qp_active = act[:na].copy()
            info.qp_iters += int(iters)
            if status != 0:
                info.qp_failures += 1
            step = float(np.max(np.abs(dv)))
            info.step_norm = step

            alpha = 1.0
            if opts.line_search:
                mu = opts.mu_init
                phi0 = cost + mu * viol
                dphi = float(g @ dv) - mu * viol
                if dphi < 0.0:
                    qt = self._qt
                    for _ in range(opts.max_ls_iter):
                        _jit.qint(q, dv, alpha, qt)
                        d.qpos[:] = qt
                        mujoco.mj_kinematics(m, d)
                        cm, vm = _jit.merit_q(d.xpos, d.xmat, self._pb, self._pl, self.S, self.F, self.cap_radius,
                                              h, self.targets, self.w_tgt, self.w_reg, self.w_smooth, qt,
                                              self.q_ref, self.q_nominal, s, e, self._pos)
                        if cm + mu * vm <= phi0 + opts.c1 * alpha * dphi:
                            break
                        alpha *= opts.tau
                        info.ls_iters += 1
                    else:
                        alpha = 1.0
            q_new = np.empty_like(q)
            _jit.qint(q, dv, alpha, q_new)
            q = q_new
            if step < opts.tol:
                info.converged = True
                break
        return q, info

    def _joint_bounds(self, qj):
        """Step bounds for the joints: limits, intersected with the per-frame step cap around q_prev."""
        lo, hi = self.kin.q_min, self.kin.q_max
        cap = self.cfg.max_joint_step
        if cap is not None:
            qp = np.clip(self.q_prev[7:], lo, hi)
            lo, hi = np.maximum(lo, qp - cap), np.minimum(hi, qp + cap)
        return {"lb": lo - qj, "ub": hi - qj}

    def _positions(self):
        """World positions of all points [targets | feet | capsule endpoints]."""
        d = self.kin.data
        if self._jit:
            _jit.points(d.xpos, d.xmat, self._pb, self._pl, self._pos)
            return self._pos
        return self.kin.positions(self.pts)

    def _ground(self, pos):
        if self._jit:
            _jit.ground_values(pos, self.S, self.F, self.cap_radius, self.cfg.ground_height, self._c)
            return self._c
        S, F, h = self.S, self.F, self.cfg.ground_height
        return np.concatenate([pos[S:S + F, 2] - h, pos[S + F:, 2] - self.cap_radius - h])

    # kept for backwards compatibility with earlier scripts
    _constraints = _ground

    def _self_collision(self, need_jac):
        d = self.kin.data
        Ra = d.xmat[self.pa_body].reshape(-1, 3, 3)
        Rb = d.xmat[self.pb_body].reshape(-1, 3, 3)
        a = d.xpos[self.pa_body][:, None] + np.einsum("pij,pkj->pki", Ra, self.pa_loc)
        b = d.xpos[self.pb_body][:, None] + np.einsum("pij,pkj->pki", Rb, self.pb_loc)
        pa, pb = segment_closest_points(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
        diff = pa - pb
        dist = np.linalg.norm(diff, axis=1)
        val = dist - self.pair_r - self.cfg.self_collision_margin
        if not need_jac:
            return val, None
        n = diff / np.maximum(dist, 1e-9)[:, None]
        return val, n, pa, pb

    # ------------------------------------------------------------ SQP oracle
    def merit_terms(self, q):
        """(cost, L1 constraint violation) at q -- forward kinematics only, no Jacobians."""
        self.kin.forward(q, with_jac=False)
        if self._jit and not self.pairs:
            d = self.kin.data
            return _jit.merit_all(d.xpos, d.xmat, self._pb, self._pl, self.S, self.F, self.cap_radius,
                                  self.cfg.ground_height, self.targets, self.w_tgt, q[7:] - self.q_nominal[7:],
                                  self.w_reg, self.difference(self.q_ref, q), self.w_smooth, self._pos)
        pos = self._positions()
        e = q[7:] - self.q_nominal[7:]
        s = self.difference(self.q_ref, q)
        if self._jit:
            cost = _jit.merit_cost(pos, self.targets, self.w_tgt, e, self.w_reg, s, self.w_smooth)
        else:
            r = pos[: self.S] - self.targets
            cost = 0.5 * (float(self.w_tgt @ np.einsum("ij,ij->i", r, r)) + float(self.w_reg @ (e * e))
                          + float(self.w_smooth @ (s * s)))
        c = self._ground(pos)
        viol = float(np.maximum(0.0, -c).sum()) if c.size else 0.0
        if self.pairs:
            v, _ = self._self_collision(False)
            viol += float(np.maximum(0.0, -v).sum())
        return cost, viol

    def linearize(self, q) -> Linearization:
        """Gauss-Newton model and linearised constraints at q (steps in the tangent space)."""
        k = self.kin
        m, d = k.model, k.data
        k.forward(q, with_jac=True)
        S, nv = self.S, self.nv
        if self._jit and not self.pairs:
            # Fully fused path: one compiled call computes points, Jacobians, H, g and every
            # ground-constraint row (all rows are kept, so the QP warm start needs no remapping).
            H = np.empty((nv, nv))
            g = np.empty(nv)
            c, C = self._c, self._Cg
            qj = q[7:]
            cost = _jit.linearize_all(
                d.xpos, d.xmat, d.cdof, d.subtree_com, self._pb, self._pl, self._root, self._chain,
                self._chain_len, S, self.F, self.cap_radius, self.cfg.ground_height, self.targets,
                self.w_tgt3, self.D, self.w_smooth, self.difference(self.q_ref, q), self.w_reg,
                qj - self.q_nominal[7:], self._pos, self._Jt.reshape(-1, nv), H, g, c, C)
            viol = float(np.maximum(0.0, -c).sum()) if c.size else 0.0
            return Linearization(H=H, g=g, cost=cost, C=C if c.size else None, d=-c if c.size else None,
                                 bidx=self.bidx, viol=viol, **self._joint_bounds(qj))
        pos = self._positions()

        # Tracking Jacobians straight from MuJoCo (C), one call per tracked point.
        Jt3 = self._Jt
        for i, b in enumerate(self._tb):
            mujoco.mj_jac(m, d, Jt3[i], None, pos[i], b)
        Jt = Jt3.reshape(-1, nv)
        e = q[7:] - self.q_nominal[7:]
        s = self.difference(self.q_ref, q)
        if self._jit:
            H = np.empty((nv, nv))
            g = np.empty(nv)
            cost = _jit.assemble(Jt, pos, self.targets, self.w_tgt3, self.D, self.w_smooth, s, self.w_reg, e, H, g)
        else:
            r = (pos[:S] - self.targets).reshape(-1)
            WJ = Jt * self.w_tgt3[:, None]
            H = Jt.T @ WJ
            H.flat[:: nv + 1] += self.D
            g = WJ.T @ r
            g += self.w_smooth * s
            g[6:] += self.w_reg * e
            cost = 0.5 * (float(r @ (self.w_tgt3 * r)) + float(self.w_reg @ (e * e))
                          + float(self.w_smooth @ (s * s)))

        c = self._ground(pos)
        if self.pairs:
            val, n, pa, pb = self._self_collision(True)
            c = np.concatenate([c, val])
        viol = float(np.maximum(0.0, -c).sum()) if c.size else 0.0

        act = self.cfg.constraint_activation
        rows = np.arange(c.size) if act is None else np.flatnonzero(c < act)
        C = None
        if rows.size:
            # One mj_jacBody per body that owns a screened row, then shift to the points
            # (only the z row is needed for ground points).
            ng = self.n_ground
            P = len(self.pairs)
            gr = rows[rows < ng] if P else rows
            Jp_u, Jr_u = self._Jp_u, self._Jr_u
            need = np.zeros(len(self.uniq), bool)
            need[self.pt_map[S + gr]] = True
            if gr.size < rows.size:
                sel = rows[rows >= ng] - ng
                need[self.sc_map[:P][sel]] = True
                need[self.sc_map[P:][sel]] = True
            uniq = self.uniq
            for u in np.flatnonzero(need).tolist():
                mujoco.mj_jacBody(m, d, Jp_u[u], Jr_u[u], int(uniq[u]))
            parts = []
            if gr.size:
                if self._jit:
                    Cg = np.empty((gr.size, nv))
                    _jit.ground_rows(gr, pos, S, d.xpos, self._cb, self.pt_map, Jp_u, Jr_u, Cg)
                else:
                    mp = self.pt_map[S + gr]
                    o = pos[S + gr] - d.xpos[self._cb[gr]]
                    Jr_c = Jr_u[mp]
                    Cg = Jp_u[mp, 2] + Jr_c[:, 0] * o[:, 1, None] - Jr_c[:, 1] * o[:, 0, None]
                parts.append(Cg)
            if gr.size < rows.size:
                ia, ib = self.sc_map[:P][sel], self.sc_map[P:][sel]
                JA = shift_jacobian(Jp_u[ia], Jr_u[ia], pa[sel] - d.xpos[self.pa_body[sel]])
                JB = shift_jacobian(Jp_u[ib], Jr_u[ib], pb[sel] - d.xpos[self.pb_body[sel]])
                parts.append(np.einsum("pi,piv->pv", n[sel], JA - JB))
            C = parts[0] if len(parts) == 1 else np.vstack(parts)

        qj = q[7:]
        return Linearization(
            H=H, g=g, cost=cost, C=C, d=-c[rows] if rows.size else None, row_ids=rows,
            n_rows=self.n_cons, bidx=self.bidx, viol=viol, **self._joint_bounds(qj),
        )


class Retargeter:
    """Sequential (causal) retargeting: q_t = SQP(q_{t-1}, targets_t)."""

    def __init__(self, cfg: IKConfig, problem: IKProblem | None = None):
        self.cfg = cfg
        self.problem = problem if problem is not None else IKProblem(cfg)
        self._boost = SQPOptions(**{**cfg.sqp.__dict__, "max_iter": cfg.boost_iters})
        self._warm = SQPOptions(**{**cfg.sqp.__dict__, "max_iter": 1})

    @property
    def nq(self):
        return self.problem.kin.nq

    def step(self, q_prev, targets, opts: SQPOptions | None = None, info: SQPInfo | None = None,
             q_prev2=None):
        """One frame: solve from ``q_prev`` (state = the two previous frames when accel_weight > 0)."""
        self.problem.set_frame(targets, q_prev, q_prev2)
        return sqp_solve(self.problem, q_prev, opts or self.cfg.sqp, info)[0]

    def warm_start(self, targets, q_init=None, info=None):
        """Converge the first frame from the nominal pose (proximal GN iterations).

        Re-centring the smoothing term on the current iterate every step removes its
        bias, so this converges to the constrained IK solution of frame 0.
        """
        q = (self.problem.q_nominal if q_init is None else q_init).copy()
        q = _init_yaw(self.problem, q, targets)
        for _ in range(self.cfg.warmup_iters):
            q_new = self.step(q, targets, self._warm, info)
            if np.max(np.abs(self.problem.difference(q, q_new))) < self.cfg.warmup_tol:
                q = q_new
                break
            q = q_new
        return q

    def run(self, targets, q_prev=None, cold_start=True, info: SQPInfo | None = None):
        """Retarget a (T, S, 3) target sequence. Returns (T, nq) MuJoCo qpos."""
        targets = targets_array(targets, self.problem.S)
        if q_prev is not None:
            q_prev = pose_array(q_prev, self.nq)
        info = info if info is not None else SQPInfo()
        failures_before = info.qp_failures
        T = targets.shape[0]
        out = np.empty((T, self.nq))
        if q_prev is None:
            q_prev = self.warm_start(targets[0], info=info)
        q_prev2 = None
        for t in range(T):
            opts = self._boost if (cold_start and t < self.cfg.boost_frames) else self.cfg.sqp
            q_new = self.step(q_prev, targets[t], opts, info, q_prev2)
            q_prev2, q_prev = (q_prev if t > 0 else None), q_new
            out[t] = q_new
        qp_succeeded(info.qp_failures - failures_before)
        return trajectory_array(out, T, self.nq)


def _init_yaw(problem: IKProblem, q, targets):
    """Rotate the nominal root yaw so the robot initially faces the mocap heading.

    Uses the best-fit yaw between the robot's and the mocap's horizontal keypoint
    layouts (Kabsch in 2D).
    """
    k = problem.kin
    k.forward(q, with_jac=False)
    p = k.positions(problem.pts)[: problem.S]
    w = problem.w_tgt
    a = p[:, :2] - np.average(p[:, :2], axis=0, weights=w)
    b = targets[:, :2] - np.average(targets[:, :2], axis=0, weights=w)
    sxx = np.sum(w * (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    sxy = np.sum(w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
    yaw = np.arctan2(sxy, sxx)
    qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    out = q.copy()
    mujoco.mju_mulQuat(out[3:7], qz, q[3:7])
    out[0:2] = np.average(targets[:, :2], axis=0, weights=w) - (
        _rotz(yaw) @ (np.average(p, axis=0, weights=w) - q[0:3]))[:2]
    return out


def _rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _dof_chain(m, body):
    """Dofs affecting ``body`` (as walked by mj_jac), from the body's last dof to the root."""
    while body and not m.body_dofnum[body]:
        body = m.body_parentid[body]
    if not body:
        return []
    out = []
    i = m.body_dofadr[body] + m.body_dofnum[body] - 1
    while i >= 0:
        out.append(int(i))
        i = m.dof_parentid[i]
    return out
