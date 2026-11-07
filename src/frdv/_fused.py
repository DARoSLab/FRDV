"""Fully fused, GIL-free clip solver (numba).

``IKProblem.solve_fast`` runs one compiled call per stage and MuJoCo for the kinematics, so every
frame still pays ~20 us of Python glue and MuJoCo call overhead. This module runs a whole clip --
warm start, boosted cold-start frames, the per-frame SQP and the despike pass -- inside one
``nogil`` numba call, so clips can be solved on plain threads with no per-frame Python at all.

The math is the same as ``solve_fast`` / ``smooth.despike`` (same cost, constraints, QP, warm
starts, line search and despike decisions); only the floating-point evaluation order differs:

* forward kinematics is re-implemented for free + hinge/slide trees (``mj_kinematics``);
* point Jacobians come from per-dof spatial motion vectors (axis, anchor) instead of ``cdof``;
* the Gauss-Newton Hessian J'WJ is accumulated with composite (subtree) 6x6 point "inertias", like
  CRBA: H_ij = s_i' K_subtree(j) s_j for i an ancestor of j, O(nv * depth) instead of O(S * chain^2);
* H is factored as L'DL with the tree's branch-induced sparsity (no fill-in, like mj_factorM)
  instead of a dense Cholesky;
* only ground rows near the floor are built for the QP; the pruned rows are checked with the
  body velocities propagated down the tree instead of explicit Jacobian rows.
"""

from __future__ import annotations

import numpy as np

from . import _jit
from .validation import pose_array, qp_succeeded, targets_array, trajectory_array

AVAILABLE = _jit.AVAILABLE

if AVAILABLE:
    import numba as nb

    _FM = {"reassoc", "contract", "arcp", "nsz"}
    _nj = nb.njit(cache=True, nogil=True, fastmath=_FM)
    # Kernels that never allocate are compiled without numba's reference counting: with it, every
    # call that passes the workspace tuples increfs/decrefs ~80 arrays (measured: most of the
    # per-frame time). Only run_clip/despike (and helpers that allocate) keep NRT.
    _nk = nb.njit(cache=True, nogil=True, fastmath=_FM, _nrt=False)

    @_nk
    def qdiff(q0, q1, out):
        """_jit.qdiff (mj_differentiatePos for [free | hinge/slide]) without NRT."""
        for i in range(3):
            out[i] = q1[i] - q0[i]
        aw, ax, ay, az = q0[3], -q0[4], -q0[5], -q0[6]
        bw, bx, by, bz = q1[3], q1[4], q1[5], q1[6]
        w = aw * bw - ax * bx - ay * by - az * bz
        x = aw * bx + bw * ax + ay * bz - az * by
        y = aw * by + bw * ay + az * bx - ax * bz
        z = aw * bz + bw * az + ax * by - ay * bx
        sn = np.sqrt(x * x + y * y + z * z)
        if sn < 1e-15:
            out[3] = out[4] = out[5] = 0.0
        else:
            ang = 2.0 * np.arctan2(sn, w)
            if ang > np.pi:
                ang -= 2.0 * np.pi
            f = ang / sn
            out[3], out[4], out[5] = x * f, y * f, z * f
        for i in range(7, q0.shape[0]):
            out[i - 1] = q1[i] - q0[i]

    @_nk
    def qint(q, dv, alpha, out):
        """_jit.qint (mj_integratePos(q, alpha * dv)) without NRT."""
        for i in range(3):
            out[i] = q[i] + alpha * dv[i]
        vx, vy, vz = dv[3], dv[4], dv[5]
        nrm = np.sqrt(vx * vx + vy * vy + vz * vz)
        aw, ax, ay, az = q[3], q[4], q[5], q[6]
        if nrm < 1e-15:
            w, x, y, z = aw, ax, ay, az
        else:
            half = 0.5 * alpha * nrm
            sh = np.sin(half) / nrm
            bw, bx, by, bz = np.cos(half), vx * sh, vy * sh, vz * sh
            w = aw * bw - ax * bx - ay * by - az * bz
            x = aw * bx + bw * ax + ay * bz - az * by
            y = aw * by + bw * ay + az * bx - ax * bz
            z = aw * bz + bw * az + ax * by - ay * bx
        n4 = np.sqrt(w * w + x * x + y * y + z * z)
        out[3], out[4], out[5], out[6] = w / n4, x / n4, y / n4, z / n4
        for i in range(7, q.shape[0]):
            out[i] = q[i] + alpha * dv[i - 1]

    @_nk
    def step_terms(q, q_ref, q_nom, q_prev, qmin, qmax, cap, s, e, lb, ub):
        """_jit.step_terms without NRT."""
        qdiff(q_ref, q, s)
        for i in range(e.shape[0]):
            qj = q[7 + i]
            e[i] = qj - q_nom[7 + i]
            lo, hi = qmin[i], qmax[i]
            if cap >= 0.0:
                qp = min(max(q_prev[7 + i], lo), hi)
                lo, hi = max(lo, qp - cap), min(hi, qp + cap)
            lb[i] = lo - qj
            ub[i] = hi - qj

    # ------------------------------------------------------------------ kinematics
    @_nk
    def _rotvec(q, v0, v1, v2):
        """mju_rotVecQuat: rotate (v0,v1,v2) by quaternion q."""
        if v0 == 0.0 and v1 == 0.0 and v2 == 0.0:
            return 0.0, 0.0, 0.0
        w, x, y, z = q[0], q[1], q[2], q[3]
        t0 = w * v0 + y * v2 - z * v1
        t1 = w * v1 + z * v0 - x * v2
        t2 = w * v2 + x * v1 - y * v0
        return (v0 + 2.0 * (y * t2 - z * t1),
                v1 + 2.0 * (z * t0 - x * t2),
                v2 + 2.0 * (x * t1 - y * t0))

    @_nk
    def _quat2mat(q, m):
        w, x, y, z = q[0], q[1], q[2], q[3]
        if w == 1.0 and x == 0.0 and y == 0.0 and z == 0.0:
            for i in range(9):
                m[i] = 0.0
            m[0] = m[4] = m[8] = 1.0
            return
        q00, q01, q02, q03 = w * w, w * x, w * y, w * z
        q11, q12, q13 = x * x, x * y, x * z
        q22, q23, q33 = y * y, y * z, z * z
        m[0] = q00 + q11 - q22 - q33
        m[4] = q00 - q11 + q22 - q33
        m[8] = q00 - q11 - q22 + q33
        m[1] = 2.0 * (q12 - q03)
        m[2] = 2.0 * (q13 + q02)
        m[3] = 2.0 * (q12 + q03)
        m[5] = 2.0 * (q23 - q01)
        m[6] = 2.0 * (q13 - q02)
        m[7] = 2.0 * (q23 + q01)

    @_nk
    def _normalize4(q):
        n = np.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if n < 1e-15:
            q[0], q[1], q[2], q[3] = 1.0, 0.0, 0.0, 0.0
        elif abs(n - 1.0) > 1e-15:
            q[0] /= n
            q[1] /= n
            q[2] /= n
            q[3] /= n

    @_nk
    def _mulquat(a, b, out):
        w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
        x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
        y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
        z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
        out[0], out[1], out[2], out[3] = w, x, y, z

    @_nk
    def fk(mdl, wk, q, dofs):
        """mj_kinematics for [free | hinge | slide] trees: xpos, xmat (and, if ``dofs``, every dof's
        world axis and anchor)."""
        (nbody, bparent, bpos, bquat, bjnt0, bjntn, jtype, jaxis, jpos, jqadr, jdadr, qpos0,
         dparent, dbody, bchain, bchain_len, pb, pl, S, G, grad, h, wpt, q_nom, qmin, qmax, bdof0, bdofn, anc, nanc, gbody, gofs, gpt, reach, lmax, gpb_) = mdl
        xpos, xquat, xmat, dax, dpt, dkind = wk[0], wk[1], wk[2], wk[3], wk[4], wk[5]
        qtmp = wk[6]
        fkq, state = wk[33], wk[34]
        for i in range(q.shape[0]):
            fkq[i] = q[i]
        state[0] = 1 if dofs else 0   # kinematics of fkq (with dof axes)
        state[1] = 0                  # target points / ground values not yet computed
        xpos[0, 0] = xpos[0, 1] = xpos[0, 2] = 0.0
        xquat[0, 0], xquat[0, 1], xquat[0, 2], xquat[0, 3] = 1.0, 0.0, 0.0, 0.0
        _quat2mat(xquat[0], xmat[0])
        for i in range(1, nbody):
            p = bparent[i]
            j0, nj = bjnt0[i], bjntn[i]
            xq = xquat[i]
            if nj > 0 and jtype[j0] == 0:
                qa, da = jqadr[j0], jdadr[j0]
                xpos[i, 0], xpos[i, 1], xpos[i, 2] = q[qa], q[qa + 1], q[qa + 2]
                xq[0], xq[1], xq[2], xq[3] = q[qa + 3], q[qa + 4], q[qa + 5], q[qa + 6]
                _normalize4(xq)
                _quat2mat(xq, xmat[i])
                if dofs:
                    m = xmat[i]
                    for k in range(3):
                        dkind[da + k] = 1
                        dax[da + k, 0] = 1.0 if k == 0 else 0.0
                        dax[da + k, 1] = 1.0 if k == 1 else 0.0
                        dax[da + k, 2] = 1.0 if k == 2 else 0.0
                        dkind[da + 3 + k] = 0
                        dax[da + 3 + k, 0] = m[k]
                        dax[da + 3 + k, 1] = m[3 + k]
                        dax[da + 3 + k, 2] = m[6 + k]
                        for a in range(3):
                            dpt[da + 3 + k, a] = xpos[i, a]
                continue
            pm = xmat[p]
            for a in range(3):
                xpos[i, a] = xpos[p, a] + pm[3 * a] * bpos[i, 0] + pm[3 * a + 1] * bpos[i, 1] + pm[3 * a + 2] * bpos[i, 2]
            _mulquat(xquat[p], bquat[i], xq)
            for jj in range(nj):
                jid = j0 + jj
                qa, da = jqadr[jid], jdadr[jid]
                ax0, ax1, ax2 = _rotvec(xq, jaxis[jid, 0], jaxis[jid, 1], jaxis[jid, 2])
                v0, v1, v2 = _rotvec(xq, jpos[jid, 0], jpos[jid, 1], jpos[jid, 2])
                an0, an1, an2 = xpos[i, 0] + v0, xpos[i, 1] + v1, xpos[i, 2] + v2
                ang = q[qa] - qpos0[qa]
                if jtype[jid] == 2:  # slide
                    xpos[i, 0] += ax0 * ang
                    xpos[i, 1] += ax1 * ang
                    xpos[i, 2] += ax2 * ang
                    if dofs:
                        dkind[da] = 1
                else:  # hinge
                    if ang != 0.0:
                        sh = np.sin(0.5 * ang)
                        qtmp[0] = np.cos(0.5 * ang)
                        qtmp[1], qtmp[2], qtmp[3] = jaxis[jid, 0] * sh, jaxis[jid, 1] * sh, jaxis[jid, 2] * sh
                        _mulquat(xq, qtmp, qtmp[4:8])
                        xq[0], xq[1], xq[2], xq[3] = qtmp[4], qtmp[5], qtmp[6], qtmp[7]
                    v0, v1, v2 = _rotvec(xq, jpos[jid, 0], jpos[jid, 1], jpos[jid, 2])
                    xpos[i, 0], xpos[i, 1], xpos[i, 2] = an0 - v0, an1 - v1, an2 - v2
                    if dofs:
                        dkind[da] = 0
                if dofs:
                    dax[da, 0], dax[da, 1], dax[da, 2] = ax0, ax1, ax2
                    dpt[da, 0], dpt[da, 1], dpt[da, 2] = an0, an1, an2
            _normalize4(xq)
            _quat2mat(xq, xmat[i])

    @_nk
    def _points(xpos, xmat, pb, pl, pos, n):
        for i in range(n):
            b = pb[i]
            m = xmat[b]
            for r in range(3):
                pos[i, r] = xpos[b, r] + m[3 * r] * pl[i, 0] + m[3 * r + 1] * pl[i, 1] + m[3 * r + 2] * pl[i, 2]

    @_nk
    def _ground_vals(xpos, xmat, pb, pl, S, G, h, grad, c):
        """c[k] = height of ground point k above the floor (minus its radius): z only."""
        for k in range(G):
            p = S + k
            b = pb[p]
            c[k] = xpos[b, 2] + xmat[b, 6] * pl[p, 0] + xmat[b, 7] * pl[p, 1] + xmat[b, 8] * pl[p, 2] - h - grad[k]

    @_nk
    def _ground_near(mdl, wk, act, allb):
        """Ground values c[k] for the points of every body that can come within ``act`` of the floor
        (all bodies if ``allb``); near[b] says which bodies were evaluated. Points of the other bodies
        are provably >= act above the floor. Returns the L1 violation sum(max(0, -c))."""
        S, h, grad, pl = mdl[18], mdl[21], mdl[20], mdl[17]
        gbody, gofs, gpt, reach = mdl[30], mdl[31], mdl[32], mdl[33]
        xpos, xmat, c, near = wk[0], wk[2], wk[12], wk[35]
        viol = 0.0
        for n in range(gbody.shape[0]):
            b = gbody[n]
            if allb or xpos[b, 2] - h - reach[n] < act:
                near[b] = 1
                m6, m7, m8, zb = xmat[b, 6], xmat[b, 7], xmat[b, 8], xpos[b, 2]
                for u in range(gofs[n], gofs[n + 1]):
                    k = gpt[u]
                    p = S + k
                    v = zb + m6 * pl[p, 0] + m7 * pl[p, 1] + m8 * pl[p, 2] - h - grad[k]
                    c[k] = v
                    if v < 0.0:
                        viol -= v
            else:
                near[b] = 0
        return viol

    # ------------------------------------------------------------------ linearisation
    @_nk
    def linearize(mdl, wk, targets, ws, wreg, D, s, e, have_pts):
        """Tree-sparse Gauss-Newton H (lower ancestor entries only) and g. Returns the cost.
        Needs ``fk(..., dofs=True)`` first, and the target points in pos unless ``have_pts``."""
        (nbody, bparent, bpos, bquat, bjnt0, bjntn, jtype, jaxis, jpos, jqadr, jdadr, qpos0,
         dparent, dbody, bchain, bchain_len, pb, pl, S, G, grad, h, wpt, q_nom, qmin, qmax, bdof0, bdofn, anc, nanc, gbody, gofs, gpt, reach, lmax, gpb_) = mdl
        xpos, xmat, dax, dpt, dkind = wk[0], wk[2], wk[3], wk[4], wk[5]
        sw, sv, pos, H, g, c = wk[7], wk[8], wk[9], wk[10], wk[11], wk[12]
        bK, bF, Hd = wk[13], wk[14], wk[29]
        anc, nanc = mdl[28], mdl[29]
        nv = g.shape[0]
        root = dbody[0]
        o0, o1, o2 = xpos[root, 0], xpos[root, 1], xpos[root, 2]
        if not have_pts:
            _points(xpos, xmat, pb, pl, pos, S)
        # dof motion vectors about o: point velocity = sv + sw x (p - o)
        for i in range(nv):
            a0, a1, a2 = dax[i, 0], dax[i, 1], dax[i, 2]
            if dkind[i] == 1:
                sw[i, 0] = sw[i, 1] = sw[i, 2] = 0.0
                sv[i, 0], sv[i, 1], sv[i, 2] = a0, a1, a2
            else:
                sw[i, 0], sw[i, 1], sw[i, 2] = a0, a1, a2
                r0, r1, r2 = o0 - dpt[i, 0], o1 - dpt[i, 1], o2 - dpt[i, 2]
                sv[i, 0] = a1 * r2 - a2 * r1
                sv[i, 1] = a2 * r0 - a0 * r2
                sv[i, 2] = a0 * r1 - a1 * r0
        # per-body point "inertia" K = sum w [(|d|^2 I - dd'), [d]x ; -[d]x, I] (10 numbers) and
        # gradient wrench F = sum w (d x r, r)
        for b in range(nbody):
            for k in range(10):
                bK[b, k] = 0.0
            for k in range(6):
                bF[b, k] = 0.0
        cost = 0.0
        for p in range(S):
            w = wpt[p]
            b = pb[p]
            d0, d1, d2 = pos[p, 0] - o0, pos[p, 1] - o1, pos[p, 2] - o2
            r0, r1, r2 = pos[p, 0] - targets[p, 0], pos[p, 1] - targets[p, 1], pos[p, 2] - targets[p, 2]
            cost += w * (r0 * r0 + r1 * r1 + r2 * r2)
            bK[b, 0] += w
            bK[b, 1] += w * d0
            bK[b, 2] += w * d1
            bK[b, 3] += w * d2
            bK[b, 4] += w * (d1 * d1 + d2 * d2)   # Qxx
            bK[b, 5] += w * (d0 * d0 + d2 * d2)   # Qyy
            bK[b, 6] += w * (d0 * d0 + d1 * d1)   # Qzz
            bK[b, 7] -= w * d0 * d1               # Qxy
            bK[b, 8] -= w * d0 * d2               # Qxz
            bK[b, 9] -= w * d1 * d2               # Qyz
            bF[b, 0] += w * (d1 * r2 - d2 * r1)
            bF[b, 1] += w * (d2 * r0 - d0 * r2)
            bF[b, 2] += w * (d0 * r1 - d1 * r0)
            bF[b, 3] += w * r0
            bF[b, 4] += w * r1
            bF[b, 5] += w * r2
        for b in range(nbody - 1, 0, -1):   # composite (subtree) sums
            p = bparent[b]
            for k in range(10):
                bK[p, k] += bK[b, k]
            for k in range(6):
                bF[p, k] += bF[b, k]
        for j in range(nv):
            b = dbody[j]
            W, m0, m1, m2 = bK[b, 0], bK[b, 1], bK[b, 2], bK[b, 3]
            Qxx, Qyy, Qzz, Qxy, Qxz, Qyz = bK[b, 4], bK[b, 5], bK[b, 6], bK[b, 7], bK[b, 8], bK[b, 9]
            w0, w1, w2 = sw[j, 0], sw[j, 1], sw[j, 2]
            v0, v1, v2 = sv[j, 0], sv[j, 1], sv[j, 2]
            # t = K s = [Q w + m x v ; W v - m x w]
            t0 = Qxx * w0 + Qxy * w1 + Qxz * w2 + (m1 * v2 - m2 * v1)
            t1 = Qxy * w0 + Qyy * w1 + Qyz * w2 + (m2 * v0 - m0 * v2)
            t2 = Qxz * w0 + Qyz * w1 + Qzz * w2 + (m0 * v1 - m1 * v0)
            t3 = W * v0 - (m1 * w2 - m2 * w1)
            t4 = W * v1 - (m2 * w0 - m0 * w2)
            t5 = W * v2 - (m0 * w1 - m1 * w0)
            Hd[j] = w0 * t0 + w1 * t1 + w2 * t2 + v0 * t3 + v1 * t4 + v2 * t5 + D[j]
            for tt in range(nanc[j]):
                i = anc[j, tt]
                H[j, tt] = sw[i, 0] * t0 + sw[i, 1] * t1 + sw[i, 2] * t2 + sv[i, 0] * t3 + sv[i, 1] * t4 + sv[i, 2] * t5
            g[j] = (w0 * bF[b, 0] + w1 * bF[b, 1] + w2 * bF[b, 2] + v0 * bF[b, 3] + v1 * bF[b, 4] + v2 * bF[b, 5]
                    + ws[j] * s[j])
        creg = 0.0
        for i in range(nv):
            creg += ws[i] * s[i] * s[i]
        for i in range(e.shape[0]):
            g[6 + i] += wreg[i] * e[i]
            creg += wreg[i] * e[i] * e[i]
        return 0.5 * (cost + creg)

    @_nk
    def _ground_row(mdl, wk, k, row):
        """z row of the Jacobian of ground point k (dense, nv)."""
        dbody, bchain, bchain_len, pb, pl, S = mdl[13], mdl[14], mdl[15], mdl[16], mdl[17], mdl[18]
        xpos, xmat, sw, sv = wk[0], wk[2], wk[7], wk[8]
        root = dbody[0]
        for i in range(row.shape[0]):
            row[i] = 0.0
        p = S + k
        b = pb[p]
        m = xmat[b]
        d0 = xpos[b, 0] + m[0] * pl[p, 0] + m[1] * pl[p, 1] + m[2] * pl[p, 2] - xpos[root, 0]
        d1 = xpos[b, 1] + m[3] * pl[p, 0] + m[4] * pl[p, 1] + m[5] * pl[p, 2] - xpos[root, 1]
        for t in range(bchain_len[b]):
            i = bchain[b, t]
            row[i] = sv[i, 2] + sw[i, 0] * d1 - sw[i, 1] * d0

    @_nk
    def _pruned_ok(mdl, wk, act, dv, tol):
        """Every ground row with c >= act still satisfies c + J_z dv >= -tol (body velocities
        propagated down the tree instead of explicit Jacobian rows)."""
        nbody, bparent, dbody, pb, S, G, bdof0, bdofn = mdl[0], mdl[1], mdl[13], mdl[16], mdl[18], mdl[19], mdl[26], mdl[27]
        xpos, sw, sv, pos, c = wk[0], wk[7], wk[8], wk[9], wk[12]
        Vb = wk[15]
        root = dbody[0]
        for a in range(6):
            Vb[0, a] = 0.0
        for b in range(1, nbody):
            p = bparent[b]
            for a in range(3):
                Vb[b, a] = Vb[p, a]
                Vb[b, 3 + a] = Vb[p, 3 + a]
            for i in range(bdof0[b], bdof0[b] + bdofn[b]):
                x = dv[i]
                for a in range(3):
                    Vb[b, a] += sw[i, a] * x
                    Vb[b, 3 + a] += sv[i, a] * x
        # vz(point) = A_b + u_b . local  (body velocity + w x (p - o), z component). A far body
        # (not evaluated by _ground_near: every point >= act above the floor) is skipped when
        # even its worst point cannot cross -tol: c >= z_b - h - reach_b, vz >= A_b - |u_b| lmax_b.
        o0, o1 = xpos[root, 0], xpos[root, 1]
        Ab, ub = wk[30], wk[31]
        gbody, gofs, gpt, reach, lmax = mdl[30], mdl[31], mdl[32], mdl[33], mdl[34]
        pl, h, grad, near = mdl[17], mdl[21], mdl[20], wk[35]
        xmat = wk[2]
        for n in range(gbody.shape[0]):
            b = gbody[n]
            m = xmat[b]
            wx, wy = Vb[b, 0], Vb[b, 1]
            A = Vb[b, 5] + wx * (xpos[b, 1] - o1) - wy * (xpos[b, 0] - o0)
            u0, u1, u2 = wx * m[3] - wy * m[0], wx * m[4] - wy * m[1], wx * m[5] - wy * m[2]
            if near[b]:
                for t in range(gofs[n], gofs[n + 1]):
                    k = gpt[t]
                    if c[k] >= act:
                        p = S + k
                        vz = A + u0 * pl[p, 0] + u1 * pl[p, 1] + u2 * pl[p, 2]
                        if c[k] + vz < -tol:
                            return False
            else:
                un = np.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
                if xpos[b, 2] - h - reach[n] + A - un * lmax[n] >= -tol:
                    continue
                for t in range(gofs[n], gofs[n + 1]):
                    k = gpt[t]
                    p = S + k
                    ck = xpos[b, 2] + m[6] * pl[p, 0] + m[7] * pl[p, 1] + m[8] * pl[p, 2] - h - grad[k]
                    vz = A + u0 * pl[p, 0] + u1 * pl[p, 1] + u2 * pl[p, 2]
                    if ck + vz < -tol:
                        return False
        return True

    @_nk
    def _merit(mdl, wk, q, q_ref, targets, ws, wreg, act):
        """(cost, L1 ground violation) at q: forward kinematics only. Leaves the kinematics, target
        points and near-floor ground values of q in the workspace (solve_frame reuses them when the
        step is accepted, as the next linearisation point)."""
        (nbody, bparent, bpos, bquat, bjnt0, bjntn, jtype, jaxis, jpos, jqadr, jdadr, qpos0,
         dparent, dbody, bchain, bchain_len, pb, pl, S, G, grad, h, wpt, q_nom, qmin, qmax, bdof0, bdofn, anc, nanc, gbody, gofs, gpt, reach, lmax, gpb_) = mdl
        fk(mdl, wk, q, True)
        xpos, xmat, pos = wk[0], wk[2], wk[9]
        s, e = wk[16], wk[17]
        _points(xpos, xmat, pb, pl, pos, S)
        viol = _ground_near(mdl, wk, act, act < 0.0)
        wk[34][1] = 1
        qdiff(q_ref, q, s)
        cost = 0.0
        for p in range(S):
            acc = 0.0
            for a in range(3):
                r = pos[p, a] - targets[p, a]
                acc += r * r
            cost += wpt[p] * acc
        for i in range(e.shape[0]):
            ei = q[7 + i] - q_nom[7 + i]
            e[i] = ei
            cost += wreg[i] * ei * ei
        for i in range(s.shape[0]):
            cost += ws[i] * s[i] * s[i]
        return 0.5 * cost, viol

    # ------------------------------------------------------------------ tree-sparse L'DL
    # H is stored compressed: Hd[k] = H[k,k], Hc[k, t] = H[k, anc[k, t]] with anc[k] = the ancestors of
    # dof k (parent first). The ancestors of anc[k, t] are exactly anc[k, t+1:], so every update of the
    # L'DL factorisation (mj_factorM's) is a contiguous vector operation.
    @_nk
    def _ltdl(Hc, Hd, anc, nanc, F, Dg, n):
        for k in range(n):
            Dg[k] = Hd[k]
            for t in range(nanc[k]):
                F[k, t] = Hc[k, t]
        for k in range(n - 1, -1, -1):
            dk = Dg[k]
            if dk <= 1e-300:
                return False
            na = nanc[k]
            for t in range(na):
                i = anc[k, t]
                fkt = F[k, t]
                tmp = fkt / dk
                Dg[i] -= tmp * fkt
                for u in range(t + 1, na):
                    F[i, u - t - 1] -= tmp * F[k, u]
                F[k, t] = tmp
        return True

    @_nk
    def _ltdl_bwd(F, anc, nanc, x, lst, nl):
        """x <- L'^-1 x for an x supported on lst (dofs in descending order, closed under ancestors)."""
        for u in range(nl):
            k = lst[u]
            xk = x[k]
            if xk != 0.0:
                for t in range(nanc[k]):
                    x[anc[k, t]] -= F[k, t] * xk

    @_nk
    def _ltdl_fwd(F, Dg, anc, nanc, n, x):
        """x <- L^-1 D^-1 x."""
        for k in range(n):
            x[k] /= Dg[k]
        for k in range(n):
            s0 = x[k]
            s1 = 0.0
            na = nanc[k]
            t = 0
            while t + 1 < na:
                s0 -= F[k, t] * x[anc[k, t]]
                s1 -= F[k, t + 1] * x[anc[k, t + 1]]
                t += 2
            if t < na:
                s0 -= F[k, t] * x[anc[k, t]]
            x[k] = s0 + s1

    @_nk
    def _ltdl_solve(F, Dg, anc, nanc, n, b, x):
        for i in range(n):
            x[i] = b[i]
        for k in range(n - 1, -1, -1):
            xk = x[k]
            for t in range(nanc[k]):
                x[anc[k, t]] -= F[k, t] * xk
        _ltdl_fwd(F, Dg, anc, nanc, n, x)

    # ------------------------------------------------------------------ dual active-set QP
    # Goldfarb-Idnani, identical to _jit.gi_qp but with the tree L'DL and preallocated buffers.
    @_nk
    def _row_dot(i, x, C, bidx, m, k, rb, bch, bl):
        """Row i of [C ; I_b ; -I_b] times x; C row i is non-zero only on the dof chain of body rb[i]."""
        if i < m:
            b = rb[i]
            s = 0.0
            for t in range(bl[b]):
                j = bch[b, t]
                s += C[i, j] * x[j]
            return s
        if i < m + k:
            return x[bidx[i - m]]
        return -x[bidx[i - m - k]]

    @_nk
    def _row_rhs(i, d, lb, ub, m, k):
        if i < m:
            return d[i]
        if i < m + k:
            return lb[i - m]
        return -ub[i - m - k]

    @_nk
    def _chol(M, n, L):
        for j in range(n):
            s = M[j, j]
            for k in range(j):
                s -= L[j, k] * L[j, k]
            if s <= 1e-300:
                return False
            ljj = np.sqrt(s)
            L[j, j] = ljj
            for i in range(j + 1, n):
                s = M[i, j]
                for k in range(j):
                    s -= L[i, k] * L[j, k]
                L[i, j] = s / ljj
        return True

    @_nk
    def _chol_solve(L, n, b, out):
        for i in range(n):
            s = b[i]
            for k in range(i):
                s -= L[i, k] * out[k]
            out[i] = s / L[i, i]
        for i in range(n - 1, -1, -1):
            s = out[i]
            for k in range(i + 1, n):
                s -= L[k, i] * out[k]
            out[i] = s / L[i, i]

    @_nk
    def _lfwd(F, anc, nanc, n, x):
        """x <- L^-1 x (two accumulators: the gathers form a dependency chain otherwise)."""
        for k in range(n):
            s0 = x[k]
            s1 = 0.0
            na = nanc[k]
            t = 0
            while t + 1 < na:
                s0 -= F[k, t] * x[anc[k, t]]
                s1 -= F[k, t + 1] * x[anc[k, t + 1]]
                t += 2
            if t < na:
                s0 -= F[k, t] * x[anc[k, t]]
            x[k] = s0 + s1

    @_nk
    def _wrow(i, F, isd, anc, nanc, C, bidx, m, k, rb, bch, bl, dbody, out):
        """out = D^-1/2 L^-T a_i (a_i = row i of [C ; I_b ; -I_b]); sparse backward pass."""
        n = out.shape[0]
        if i < m:
            for r in range(n):
                out[r] = C[i, r]
            b = rb[i]
        else:
            for r in range(n):
                out[r] = 0.0
            if i < m + k:
                j = bidx[i - m]
                out[j] = 1.0
            else:
                j = bidx[i - m - k]
                out[j] = -1.0
            b = dbody[j]
        _ltdl_bwd(F, anc, nanc, out, bch[b], bl[b])
        for r in range(n):
            out[r] *= isd[r]

    @_nk
    def _dot(a, b, n):
        s0 = 0.0
        s1 = 0.0
        t = 0
        while t + 1 < n:
            s0 += a[t] * b[t]
            s1 += a[t + 1] * b[t + 1]
            t += 2
        if t < n:
            s0 += a[t] * b[t]
        return s0 + s1

    @_nk
    def _x_of(F, isd, anc, nanc, n, xi, x):
        """x = L^-1 D^-1/2 xi."""
        for i in range(n):
            x[i] = isd[i] * xi[i]
        _lfwd(F, anc, nanc, n, x)

    @_nk
    def gi_qp(qw, Hc, Hd, anc, nanc, g, C, d, m, rb, bch, bl, dbody, bidx, lb, ub, active0, n0, tol, max_iter):
        """Goldfarb-Idnani dual active set (same iterations as _jit.gi_qp), with H = L'DL factored
        along the tree and every vector kept in the scaled space W = D^-1/2 L^-T (H^-1 = J J',
        J = L^-1 D^-1/2): a constraint costs one sparse backward pass, products like a_i' H^-1 a_j
        are dots of W rows, and the primal x = J xi needs one forward pass per outer iteration.
        Returns (status, iters, n_active); solution in qw[0], active set in qw[1]."""
        x, act, F, Dg, wg, xi, u, Wa, M, Lm, rhs, r, zeta, Wp, isd = qw
        n = g.shape[0]
        k = bidx.shape[0]
        for i in range(n):
            x[i] = 0.0
        if not _ltdl(Hc, Hd, anc, nanc, F, Dg, n):
            return 3, 0, 0
        for i in range(n):
            isd[i] = 1.0 / np.sqrt(Dg[i])
            wg[i] = g[i]
        for kk in range(n - 1, -1, -1):        # wg = D^-1/2 L^-T g
            v = wg[kk]
            for t in range(nanc[kk]):
                wg[anc[kk, t]] -= F[kk, t] * v
        for i in range(n):
            wg[i] *= isd[i]
            xi[i] = -wg[i]
        na = 0
        it = 0
        mt = m + 2 * k
        for wi in range(n0):
            w = active0[wi]
            if w < mt and na < n and np.isfinite(_row_rhs(w, d, lb, ub, m, k)):
                act[na] = w
                na += 1
        for a in range(na):
            _wrow(act[a], F, isd, anc, nanc, C, bidx, m, k, rb, bch, bl, dbody, Wa[a])
        while na > 0:
            for a in range(na):
                for b in range(a + 1):
                    M[a, b] = M[b, a] = _dot(Wa[a], Wa[b], n)
                rhs[a] = _row_rhs(act[a], d, lb, ub, m, k) + _dot(Wa[a], wg, n)
            it += 1
            if not _chol(M, na, Lm):
                na = 0
                break
            _chol_solve(Lm, na, rhs, r)
            keep = 0
            for a in range(na):
                if r[a] >= 0.0:
                    if keep != a:
                        act[keep] = act[a]
                        for i in range(n):
                            Wa[keep, i] = Wa[a, i]
                    keep += 1
            if keep == na:
                for i in range(n):
                    s = -wg[i]
                    for a in range(na):
                        s += Wa[a, i] * r[a]
                    xi[i] = s
                for a in range(na):
                    u[a] = r[a]
                break
            na = keep

        while True:
            _x_of(F, isd, anc, nanc, n, xi, x)
            p, sp = -1, 0.0
            for i in range(mt):
                v = _row_dot(i, x, C, bidx, m, k, rb, bch, bl) - _row_rhs(i, d, lb, ub, m, k)
                if v < sp:
                    sp, p = v, i
            if p < 0 or sp >= -tol:
                return 0, it, na
            for a in range(na):
                if act[a] == p:
                    return 0, it, na
            if it >= max_iter:
                return 2, it, na
            _wrow(p, F, isd, anc, nanc, C, bidx, m, k, rb, bch, bl, dbody, Wp)
            u_p = 0.0
            while True:
                it += 1
                if na > 0:
                    for a in range(na):
                        for b in range(a + 1):
                            M[a, b] = M[b, a] = _dot(Wa[a], Wa[b], n)
                        rhs[a] = _dot(Wa[a], Wp, n)
                    if not _chol(M, na, Lm):
                        _x_of(F, isd, anc, nanc, n, xi, x)
                        return 2, it, na
                    _chol_solve(Lm, na, rhs, r)
                    for i in range(n):
                        s = Wp[i]
                        for a in range(na):
                            s -= Wa[a, i] * r[a]
                        zeta[i] = s
                else:
                    for i in range(n):
                        zeta[i] = Wp[i]
                t1 = np.inf
                kdrop = -1
                for a in range(na):
                    if r[a] > 1e-14:
                        tj = u[a] / r[a]
                        if tj < t1:
                            t1 = tj
                            kdrop = a
                nz = _dot(Wp, zeta, n)
                if nz > 1e-14:
                    t2 = (_row_rhs(p, d, lb, ub, m, k) - _dot(Wp, xi, n)) / nz
                else:
                    t2 = np.inf
                t = t1 if t1 < t2 else t2
                if t == np.inf:
                    _x_of(F, isd, anc, nanc, n, xi, x)
                    return 1, it, na
                if t2 != np.inf:
                    for i in range(n):
                        xi[i] += t * zeta[i]
                for a in range(na):
                    u[a] -= t * r[a]
                u_p += t
                if t2 != np.inf and t == t2:
                    act[na] = p
                    u[na] = u_p
                    for i in range(n):
                        Wa[na, i] = Wp[i]
                    na += 1
                    break
                for a in range(kdrop, na - 1):
                    act[a] = act[a + 1]
                    u[a] = u[a + 1]
                    for i in range(n):
                        Wa[a, i] = Wa[a + 1, i]
                na -= 1

    # ------------------------------------------------------------------ one frame
    @_nk
    def solve_frame(mdl, wk, qw, prm, opt, q0, q_ref, q_prev, targets, actg, nact, qout, stats):
        """IKProblem.solve_fast for one frame, from q0. ``actg[:nact[0]]`` is the QP warm start
        (global ids), updated in place. Result in qout."""
        (nbody, bparent, bpos, bquat, bjnt0, bjntn, jtype, jaxis, jpos, jqadr, jdadr, qpos0,
         dparent, dbody, bchain, bchain_len, pb, pl, S, G, grad, h, wpt, q_nom, qmin, qmax, bdof0, bdofn, anc, nanc, gbody, gofs, gpt, reach, lmax, gpb_) = mdl
        ws, D, wreg, cap, bidx = prm
        max_iter, tol, line_search, mu_init, c1, tau, max_ls, qp_tol, act_lim = opt
        H, g, c = wk[10], wk[11], wk[12]
        s, e, lb, ub = wk[16], wk[17], wk[18], wk[19]
        Csub, dsub, idx, local_of, a0, Cfull, dneg = wk[20], wk[21], wk[22], wk[23], wk[24], wk[25], wk[26]
        q, qt = wk[27], wk[28]
        x, act = qw[0], qw[1]
        nv = g.shape[0]
        for i in range(q0.shape[0]):
            q[i] = q0[i]
        fkq, state, near, rbs = wk[33], wk[34], wk[35], wk[36]
        gpb = mdl[35]
        allb = act_lim < 0.0
        for _ in range(max_iter):
            stats[0] += 1
            # the line search of the previous step (or frame) usually evaluated exactly this q
            same = state[0] == 1 and state[1] == 1
            if same:
                for i in range(q.shape[0]):
                    if q[i] != fkq[i]:
                        same = False
                        break
            if same:
                viol = 0.0
                for nn in range(mdl[30].shape[0]):
                    bb = mdl[30][nn]
                    if near[bb]:
                        for u in range(mdl[31][nn], mdl[31][nn + 1]):
                            if c[mdl[32][u]] < 0.0:
                                viol -= c[mdl[32][u]]
            else:
                fk(mdl, wk, q, True)
                _points(wk[0], wk[2], pb, pl, wk[9], S)
                viol = _ground_near(mdl, wk, act_lim, allb)
            state[1] = 0
            step_terms(q, q_ref, q_nom, q_prev, qmin, qmax, cap, s, e, lb, ub)
            cost = linearize(mdl, wk, targets, ws, wreg, D, s, e, True)
            ok = False
            na = 0
            if act_lim >= 0.0 and G > 0:
                k = 0
                for i in range(G):
                    if near[gpb[i]] and c[i] < act_lim:
                        _ground_row(mdl, wk, i, Csub[k])
                        dsub[k] = -c[i]
                        idx[k] = i
                        rbs[k] = gpb[i]
                        local_of[i] = k
                        k += 1
                    else:
                        local_of[i] = -1
                n0 = 0
                for ai in range(nact[0]):
                    a = actg[ai]
                    if a < G:
                        if local_of[a] >= 0:
                            a0[n0] = local_of[a]
                            n0 += 1
                    else:
                        a0[n0] = k + (a - G)
                        n0 += 1
                status, iters, na = gi_qp(qw, H, wk[29], anc, nanc, g, Csub, dsub, k, rbs, bchain, bchain_len, dbody,
                                          bidx, lb, ub, a0, n0, qp_tol, 200)
                stats[4] += k
                stats[5] += na
                if status == 0 and _pruned_ok(mdl, wk, act_lim, x, 1e-9):
                    for i in range(na):
                        a = act[i]
                        actg[i] = idx[a] if a < k else G + (a - k)
                    nact[0] = na
                    ok = True
                else:
                    stats[2] += iters
            if not ok:
                stats[6] += 1
                _ground_near(mdl, wk, act_lim, True)
                for i in range(G):
                    _ground_row(mdl, wk, i, Cfull[i])
                    dneg[i] = -c[i]
                for i in range(nact[0]):
                    a0[i] = actg[i]
                status, iters, na = gi_qp(qw, H, wk[29], anc, nanc, g, Cfull, dneg, G, gpb, bchain, bchain_len, dbody,
                                          bidx, lb, ub, a0, nact[0], qp_tol, 200)
                for i in range(na):
                    actg[i] = act[i]
                nact[0] = na
            stats[2] += iters
            if status != 0:
                stats[3] += 1
            step = 0.0
            for i in range(nv):
                if abs(x[i]) > step:
                    step = abs(x[i])
            alpha = 1.0
            if line_search:
                mu = mu_init
                phi0 = cost + mu * viol
                gd = 0.0
                for i in range(nv):
                    gd += g[i] * x[i]
                dphi = gd - mu * viol
                if dphi < 0.0:
                    found = False
                    for _ in range(max_ls):
                        qint(q, x, alpha, qt)
                        cm, vm = _merit(mdl, wk, qt, q_ref, targets, ws, wreg, act_lim)
                        stats[7] += 1
                        if cm + mu * vm <= phi0 + c1 * alpha * dphi:
                            found = True
                            break
                        alpha *= tau
                        stats[1] += 1
                    if not found:
                        alpha = 1.0
            qint(q, x, alpha, qt)
            for i in range(q.shape[0]):
                q[i] = qt[i]
            if step < tol:
                break
        for i in range(q.shape[0]):
            qout[i] = q[i]

    @_nk
    def set_ref(q_prev, q_prev2, alpha, has2, tmp, q_ref):
        """IKProblem.set_frame: smoothing centre = q_prev advanced by alpha of the last velocity."""
        if not has2 or alpha == 0.0:
            for i in range(q_prev.shape[0]):
                q_ref[i] = q_prev[i]
        else:
            qdiff(q_prev2, q_prev, tmp)
            for i in range(tmp.shape[0]):
                tmp[i] = alpha * tmp[i]
            qint(q_prev, tmp, 1.0, q_ref)

    @_nk
    def init_yaw(q_nom, p0, w, targets, out):
        """retarget._init_yaw with the nominal-pose target points p0 precomputed."""
        S = p0.shape[0]
        sw = 0.0
        pa0 = pa1 = pa2 = ta0 = ta1 = 0.0
        for i in range(S):
            sw += w[i]
            pa0 += w[i] * p0[i, 0]
            pa1 += w[i] * p0[i, 1]
            pa2 += w[i] * p0[i, 2]
            ta0 += w[i] * targets[i, 0]
            ta1 += w[i] * targets[i, 1]
        pa0, pa1, pa2, ta0, ta1 = pa0 / sw, pa1 / sw, pa2 / sw, ta0 / sw, ta1 / sw
        sxx = sxy = 0.0
        for i in range(S):
            a0, a1 = p0[i, 0] - pa0, p0[i, 1] - pa1
            b0, b1 = targets[i, 0] - ta0, targets[i, 1] - ta1
            sxx += w[i] * (a0 * b0 + a1 * b1)
            sxy += w[i] * (a0 * b1 - a1 * b0)
        yaw = np.arctan2(sxy, sxx)
        for i in range(out.shape[0]):
            out[i] = q_nom[i]
        zw, zz = np.cos(yaw / 2), np.sin(yaw / 2)   # mju_mulQuat((zw, 0, 0, zz), q_nom[3:7])
        a0, a1, a2, a3 = q_nom[3], q_nom[4], q_nom[5], q_nom[6]
        out[3] = zw * a0 - zz * a3
        out[4] = zw * a1 - zz * a2
        out[5] = zw * a2 + zz * a1
        out[6] = zw * a3 + zz * a0
        cy, sy = np.cos(yaw), np.sin(yaw)
        v0, v1 = pa0 - q_nom[0], pa1 - q_nom[1]
        out[0] = ta0 - (cy * v0 - sy * v1)
        out[1] = ta1 - (sy * v0 + cy * v1)

    @_nk
    def run_clip(mdl, wk, qw, prm, opt, opt_boost, opt_warm, p0, warmup_iters, warmup_tol, boost_frames, alpha,
                 targets, q_init, do_warm, out, actg, nact, stats):
        """Retargeter.run: warm start (if ``do_warm``, else start from q_init), the first
        ``boost_frames`` frames with ``opt_boost``, then one ``opt`` solve per frame."""
        T = targets.shape[0]
        nq = out.shape[1]
        qa, qb, qref, tmp = wk[37], wk[38], wk[39], wk[40]
        if T == 0:
            return
        if do_warm:  # proximal GN iterations from the yaw-aligned nominal pose
            init_yaw(mdl[23], p0, mdl[22], targets[0], qa)
            for _ in range(warmup_iters):
                solve_frame(mdl, wk, qw, prm, opt_warm, qa, qa, qa, targets[0], actg, nact, qb, stats)
                qdiff(qa, qb, tmp)
                mx = 0.0
                for i in range(tmp.shape[0]):
                    if abs(tmp[i]) > mx:
                        mx = abs(tmp[i])
                for i in range(nq):
                    qa[i] = qb[i]
                if mx < warmup_tol:
                    break
        else:
            for i in range(nq):
                qa[i] = q_init[i]
        # state: qa = q_prev, out[t-2] = q_prev2 (none for the first two frames, as in Retargeter.run)
        for t in range(T):
            o = opt_boost if t < boost_frames else opt
            set_ref(qa, out[t - 2] if t >= 2 else qa, alpha, t >= 2, tmp, qref)
            solve_frame(mdl, wk, qw, prm, o, qa, qref, qa, targets[t], actg, nact, out[t], stats)
            for i in range(nq):
                qa[i] = out[t, i]

    # ------------------------------------------------------------------ despike (smooth.despike)
    @_nk
    def _resolve(mdl, wk, qw, prm, opt, qs, targets, q, a, n, actg, nact, qtmp, stats):
        """Re-project q[a:a+n] onto the constraints, pulled towards qs (despike's proj.step)."""
        for i in range(n):
            solve_frame(mdl, wk, qw, prm, opt, qs[i], qs[i], qs[i], targets[a + i], actg, nact, qtmp, stats)
            for j in range(qtmp.shape[0]):
                q[a + i, j] = qtmp[j]

    @_nk
    def joint_steps(q, nv, out):
        """smooth.joint_steps: (T-1, nv) tangent steps, root rotation via the shortest-arc log."""
        T = q.shape[0]
        for t in range(T - 1):
            a, b = q[t], q[t + 1]
            for i in range(3):
                out[t, i] = b[i] - a[i]
            for i in range(7, q.shape[1]):
                out[t, i - 1] = b[i] - a[i]
            na = np.sqrt(a[3] * a[3] + a[4] * a[4] + a[5] * a[5] + a[6] * a[6])
            nb_ = np.sqrt(b[3] * b[3] + b[4] * b[4] + b[5] * b[5] + b[6] * b[6])
            w0, x0, y0, z0 = a[3] / na, -a[4] / na, -a[5] / na, -a[6] / na
            w1, x1, y1, z1 = b[3] / nb_, b[4] / nb_, b[5] / nb_, b[6] / nb_
            w = w0 * w1 - (x0 * x1 + y0 * y1 + z0 * z1)
            vx = w0 * x1 + w1 * x0 + (y0 * z1 - z0 * y1)
            vy = w0 * y1 + w1 * y0 + (z0 * x1 - x0 * z1)
            vz = w0 * z1 + w1 * z0 + (x0 * y1 - y0 * x1)
            if w < 0:
                w, vx, vy, vz = -w, -vx, -vy, -vz
            sa = np.sqrt(vx * vx + vy * vy + vz * vz)
            ang = 2.0 * np.arctan2(sa, w)
            sc = ang / max(sa, 1e-300) if sa > 1e-15 else 2.0
            out[t, 3], out[t, 4], out[t, 5] = vx * sc, vy * sc, vz * sc

    @_nj
    def find_spikes(steps, n, threshold, ratio, k, flags):
        """smooth.find_spikes on steps[:n]: flags[t] = 1 for spike steps; returns (count, worst)."""
        nv = steps.shape[1]
        T = n
        for t in range(T):
            flags[t] = 0
        if T < 3:
            return 0, 0.0
        nbuf = np.empty(2 * k)
        worst = 0.0
        cnt = 0
        for t in range(T):
            for dd in range(nv):
                v = abs(steps[t, dd])
                if v > threshold:
                    m = 0
                    for o in range(-k, k + 1):
                        if o == 0:
                            continue
                        tt = min(max(t + o, 0), T - 1)
                        nbuf[m] = abs(steps[tt, dd])
                        m += 1
                    nbuf.sort()
                    med = 0.5 * (nbuf[k - 1] + nbuf[k]) if (2 * k) % 2 == 0 else nbuf[k]
                    base = max(med, 0.02)
                    if v > ratio * base:
                        if flags[t] == 0:
                            flags[t] = 1
                            cnt += 1
                        if v > worst:
                            worst = v
        return cnt, worst

    @_nk
    def _window_score(mdl, wk, q, a, b, nv, st):
        T = q.shape[0]
        lo, hi = max(0, a - 2), min(T, b + 2)
        n = hi - lo
        acc = 0.0
        if n - 1 > 1:
            joint_steps(q[lo:hi], nv, st)
            for t in range(n - 2):
                for i in range(6, nv):
                    v = abs(st[t + 1, i] - st[t, i])
                    if v > acc:
                        acc = v
        S, G, grad, h, pb, pl = mdl[18], mdl[19], mdl[20], mdl[21], mdl[16], mdl[17]
        viol = 0.0
        for t in range(a, b):
            fk(mdl, wk, q[t], False)
            cz = wk[32]
            _ground_vals(wk[0], wk[2], pb, pl, S, G, h, grad, cz)  # all points (min over the body)
            mn = np.inf
            for k in range(G):
                if cz[k] < mn:
                    mn = cz[k]
            viol = max(viol, -mn)
        return acc, max(viol, 0.0)

    @_nj
    def despike(mdl, wk, qw, prm_list, opt, targets, q, threshold, ratio, half_window, sigma, passes, irep, frep,
                wins):
        """smooth.despike in place. prm_list[k] = (ws, D, wreg, cap, bidx) for pass k.
        irep = [spikes_before, spikes_after, windows kept, reverted], frep = [worst_before, worst_after],
        wins[:irep[2]] = the kept windows."""
        T, nq = q.shape
        nv = prm_list[0][0].shape[0]
        steps = np.empty((max(T - 1, 1), nv))
        flags = np.zeros(max(T - 1, 1), np.int64)
        st = np.empty((T + 64, nv))
        if T >= 2:
            joint_steps(q, nv, steps)
        cnt, worst = find_spikes(steps, T - 1, threshold, ratio, 7, flags)
        irep[0], frep[0] = cnt, worst
        actg = np.empty(nv + 1, np.int64)
        nact = np.zeros(1, np.int64)
        stats = np.zeros(4, np.int64)
        qtmp = np.empty(nq)
        for kp in range(passes):
            if cnt == 0:
                break
            prm = prm_list[kp]
            nact[0] = 0
            hw = half_window + 4 * kp
            sg = sigma * (1 + kp)
            pad = int(np.ceil(3 * sg))
            wts = np.empty(2 * pad + 1)
            for i in range(2 * pad + 1):
                kk = i - pad
                wts[i] = np.exp(-0.5 * (kk / sg) ** 2)
            wts /= wts.sum()
            # windows (smooth._windows) from the flagged steps, in order
            wa = np.empty(cnt, np.int64)
            wb = np.empty(cnt, np.int64)
            nw = 0
            for t in range(T - 1):
                if flags[t]:
                    a, b = max(1, t - hw), min(T - 1, t + hw + 1)
                    if nw > 0 and a <= wb[nw - 1] + 2:
                        wb[nw - 1] = max(wb[nw - 1], b)
                    else:
                        wa[nw], wb[nw] = a, b
                        nw += 1
            for w in range(nw):
                a, b = wa[w], wb[w]
                # _smooth_window
                anchor = q[a - 1].copy()
                lo, hi = max(0, a - pad), min(T, b + pad)
                L = hi - lo
                V = np.empty((L + 2 * pad, nv))
                for t in range(L):
                    qdiff(anchor, q[lo + t], V[pad + t])
                for t in range(pad):
                    V[t] = V[pad]
                    V[pad + L + t] = V[pad + L - 1]
                n = b - a
                qs = np.empty((n, nq))
                vv = np.empty(nv)
                for i in range(n):
                    t = a + i
                    tl = t - lo + pad          # index of t in the padded V
                    tp = np.sin(np.pi * (i + 1) / (n + 1)) ** 2
                    for dd in range(nv):
                        acc = 0.0
                        for j in range(2 * pad + 1):
                            acc += V[tl - pad + j, dd] * wts[2 * pad - j]
                        vv[dd] = V[tl, dd] + tp * (acc - V[tl, dd])
                    qint(anchor, vv, 1.0, qs[i])
                old = q[a:b].copy()
                s0a, s0v = _window_score(mdl, wk, q, a, b, nv, st)
                _resolve(mdl, wk, qw, prm, opt, qs, targets, q, a, n, actg, nact, qtmp, stats)
                s1a, s1v = _window_score(mdl, wk, q, a, b, nv, st)
                if s1a < s0a and s1v <= s0v + 1e-3:
                    wins[irep[2], 0], wins[irep[2], 1] = a, b
                    irep[2] += 1
                else:
                    q[a:b] = old
                    irep[3] += 1
            joint_steps(q, nv, steps)
            cnt, worst = find_spikes(steps, T - 1, threshold, ratio, 7, flags)
        if T >= 2:
            joint_steps(q, nv, steps)
        cnt, worst = find_spikes(steps, T - 1, threshold, ratio, 7, flags)
        irep[1], frep[1] = cnt, worst


def supported(problem) -> bool:
    """The fused path covers the default problem: numba, free root + hinge/slide joints, no
    self-collision pairs, isotropic per-point weights."""
    if not (AVAILABLE and getattr(problem, "fast", False)):
        return False
    m = problem.kin.model
    if m.jnt_type[0] != 0 or any(t not in (2, 3) for t in m.jnt_type[1:]):
        return False
    if any(m.body_jntnum[b] > 1 and m.jnt_type[m.body_jntadr[b]] == 0 for b in range(m.nbody)):
        return False
    return True


class Fused:
    """Per-problem constant arrays for the fused kernels; ``workspace()`` makes per-thread buffers."""

    def __init__(self, problem):
        import mujoco
        p = problem
        m = p.kin.model
        self.problem = p
        nv, nq, nb = m.nv, m.nq, m.nbody
        self.nv, self.nq = nv, nq
        i64 = lambda a: np.ascontiguousarray(a, dtype=np.int64)  # noqa: E731
        f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)  # noqa: E731
        chains = []
        for b in range(nb):
            from .retarget import _dof_chain
            chains.append(_dof_chain(m, b) if b else [])
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
        # ground points grouped by body (CSR), with each body's reach below its origin
        gb = p._pb[p.S:]
        gbody = np.unique(gb)
        gpt = np.concatenate([np.flatnonzero(gb == b) for b in gbody]) if len(gb) else np.zeros(0, np.int64)
        gofs = np.r_[0, np.cumsum([np.sum(gb == b) for b in gbody])].astype(np.int64)
        ln = np.linalg.norm(p._pl[p.S:], axis=1) if len(gb) else np.zeros(0)
        reach = np.array([np.max(ln[gb == b] + grad[gb == b]) for b in gbody]) * (1 + 1e-9) + 1e-12
        lmax = np.array([np.max(ln[gb == b]) for b in gbody]) * (1 + 1e-9) + 1e-12
        self.mdl = (
            nb, i64(m.body_parentid), f64(m.body_pos), f64(m.body_quat), i64(m.body_jntadr), i64(m.body_jntnum),
            i64(m.jnt_type), f64(m.jnt_axis), f64(m.jnt_pos), i64(m.jnt_qposadr), i64(m.jnt_dofadr), f64(m.qpos0),
            i64(m.dof_parentid), i64(m.dof_bodyid), bchain, i64([len(c) for c in chains]),
            i64(p._pb), f64(p._pl), int(p.S), int(p.n_ground), grad, float(p.cfg.ground_height), f64(p.w_tgt),
            f64(p.q_nominal), f64(p.kin.q_min), f64(p.kin.q_max), i64(m.body_dofadr.clip(0)), i64(m.body_dofnum),
            anc, i64(nanc), i64(gbody), i64(gofs), i64(gpt), f64(reach), f64(lmax), i64(p._pb[p.S:]),
        )
        # nominal-pose target points for the yaw initialisation
        d = mujoco.MjData(m)
        d.qpos[:] = p.q_nominal
        mujoco.mj_kinematics(m, d)
        R = d.xmat[p._pb[: p.S]].reshape(-1, 3, 3)
        self.p0 = d.xpos[p._pb[: p.S]] + np.einsum("bij,bj->bi", R, p._pl[: p.S])
        self.bidx = i64(p.bidx)

    def workspace(self):
        p, nv, nq, nb = self.problem, self.nv, self.nq, self.mdl[0]
        P, G, nj = len(p._pb), p.n_ground, nv - 6
        wk = (np.zeros((nb, 3)), np.zeros((nb, 4)), np.zeros((nb, 9)), np.zeros((nv, 3)), np.zeros((nv, 3)),
              np.zeros(nv, np.int64), np.zeros(8), np.zeros((nv, 3)), np.zeros((nv, 3)), np.zeros((P, 3)),
              np.zeros((nv, nv)), np.zeros(nv), np.zeros(G), np.zeros((nb, 10)), np.zeros((nb, 6)),
              np.zeros((nb, 6)), np.zeros(nv), np.zeros(nj), np.zeros(nj), np.zeros(nj),
              np.zeros((max(G, 1), nv)), np.zeros(max(G, 1)), np.zeros(max(G, 1), np.int64),
              np.zeros(max(G, 1), np.int64), np.zeros(G + 2 * nj + 1, np.int64), np.zeros((max(G, 1), nv)),
              np.zeros(max(G, 1)), np.zeros(nq), np.zeros(nq), np.zeros(nv), np.zeros(nb), np.zeros((nb, 3)),
              np.zeros(max(G, 1)), np.zeros(nq), np.zeros(2, np.int64), np.zeros(nb, np.int64),
              np.zeros(max(G, 1), np.int64), np.zeros(nq), np.zeros(nq), np.zeros(nq), np.zeros(nv))
        n = nv
        qw = (np.zeros(n), np.zeros(n + 1, np.int64), np.zeros((n, n)), np.zeros(n), np.zeros(n), np.zeros(n),
              np.zeros(n + 1), np.zeros((n + 1, n)), np.zeros((n + 1, n + 1)), np.zeros((n + 1, n + 1)),
              np.zeros(n + 1), np.zeros(n + 1), np.zeros(n), np.zeros(n), np.zeros(n))
        return wk, qw, np.zeros(nv + 1, np.int64), np.zeros(1, np.int64), np.zeros(8, np.int64)

    def params(self, cfg, max_joint_step=None):
        """(ws, D, wreg, cap, bidx) like IKProblem.__init__ for a (possibly modified) IKConfig."""
        nv = self.nv
        reg = np.broadcast_to(np.asarray(cfg.joint_reg_weights, float), (nv - 6,)).copy()
        ws = np.full(nv, float(cfg.smoothing_weight) + float(cfg.accel_weight))
        ws[3:6] *= cfg.rot_smoothing_scale
        D = ws.copy()
        D[6:] += reg
        D += cfg.hessian_reg
        cap = -1.0 if max_joint_step is None else float(max_joint_step)
        alpha = cfg.accel_weight / max(cfg.smoothing_weight + cfg.accel_weight, 1e-12)
        return (ws, D, reg, cap, self.bidx), alpha

    @staticmethod
    def options(o, act_lim):
        return (int(o.max_iter), float(o.tol), bool(o.line_search), float(o.mu_init), float(o.c1), float(o.tau),
                int(o.max_ls_iter), float(o.qp_tol), -1.0 if act_lim is None else float(act_lim))

    def run(self, cfg, targets, wsp=None, q_prev=None, cold_start=True, info=None):
        """Retargeter.run(targets, q_prev, cold_start) in one GIL-free call. ``wsp``: a
        ``workspace()`` (per thread; it also carries the QP warm start between calls)."""
        from dataclasses import replace
        targets = targets_array(targets, self.problem.S)
        if q_prev is not None:
            q_prev = pose_array(q_prev, self.nq)
        wk, qw, actg, nact, stats = wsp if wsp is not None else self.workspace()
        prm, alpha = self.params(cfg, cfg.max_joint_step)
        act = cfg.constraint_activation
        o = self.options(cfg.sqp, act)
        ob = self.options(replace(cfg.sqp, max_iter=cfg.boost_iters), act)
        ow = self.options(replace(cfg.sqp, max_iter=1), act)
        targets = np.ascontiguousarray(targets, dtype=float).reshape(len(targets), -1, 3)
        out = np.empty((len(targets), self.nq))
        st0 = stats.copy()
        q0 = np.zeros(self.nq) if q_prev is None else np.ascontiguousarray(q_prev, dtype=float)
        run_clip(self.mdl, wk, qw, prm, o, ob, ow, self.p0, int(cfg.warmup_iters), float(cfg.warmup_tol),
                 int(cfg.boost_frames) if cold_start else 0, float(alpha), targets, q0, q_prev is None, out,
                 actg, nact, stats)
        if info is not None:
            d = stats - st0
            info.iters += int(d[0])
            info.ls_iters += int(d[1])
            info.qp_iters += int(d[2])
            info.qp_failures += int(d[3])
        qp_succeeded(int(stats[3] - st0[3]))
        return trajectory_array(out, len(targets), self.nq)

    def despike(self, cfg, targets, q, wsp=None, threshold=0.25, ratio=4.0, half_window=5, sigma=1.5,
                pull_weight=200.0, iters=2, passes=3):
        """smooth.despike (same arguments) in one GIL-free call; returns a DespikeReport."""
        from dataclasses import replace
        from .smooth import DespikeReport
        from .sqp import SQPOptions
        targets = targets_array(targets, self.problem.S)
        trajectory_array(q, len(targets), self.nq)
        wk, qw, _, _, _ = wsp if wsp is not None else self.workspace()
        prms = tuple(self.params(replace(cfg, smoothing_weight=pull_weight * 4 ** k, accel_weight=0.0,
                                         rot_smoothing_scale=1.0), None)[0] for k in range(max(passes, 1)))
        opt = self.options(SQPOptions(max_iter=iters, line_search=True), cfg.constraint_activation)
        irep, frep = np.zeros(4, np.int64), np.zeros(2)
        wins = np.zeros((max(len(q), 1), 2), np.int64)
        targets = np.ascontiguousarray(targets, dtype=float).reshape(len(targets), -1, 3)
        despike(self.mdl, wk, qw, prms, opt, targets, q, float(threshold), float(ratio), int(half_window),
                float(sigma), int(passes), irep, frep, wins)
        trajectory_array(q, len(targets), self.nq)
        return DespikeReport(spikes_before=int(irep[0]), spikes_after=int(irep[1]),
                             windows=[tuple(map(int, w)) for w in wins[: irep[2]]], worst_before=float(frep[0]),
                             worst_after=float(frep[1]), reverted=int(irep[3]))
