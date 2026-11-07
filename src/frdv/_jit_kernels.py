"""Optional numba kernels (``pip install frdv[fast]``); imported lazily through ``frdv._jit``.

Same math as the numpy implementations in qp.py / retarget.py, but fused into a
handful of compiled calls so per-frame cost is dominated by MuJoCo rather than
by numpy dispatch overhead. Linear algebra is hand-rolled (n <= ~60), so no
LAPACK/SciPy is needed. Disable with ``FRDV_NO_JIT=1``.
"""

from __future__ import annotations

import os

import numpy as np

import numba as nb

AVAILABLE = True

if AVAILABLE:
    # Allow reassociation/FMA for fast reductions, but keep IEEE inf/nan semantics: the QP uses inf
    # step lengths to detect infeasibility, which "ninf"/"nnan" would let LLVM optimise away.
    _jit = nb.njit(cache=True, nogil=True, fastmath={"reassoc", "contract", "arcp", "nsz"})

    # ------------------------------------------------------------------ linalg
    @_jit
    def _chol(M, n, L):
        """In-place lower Cholesky of M[:n,:n] into L. Returns False if not PD."""
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

    @_jit
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

    @_jit
    def _spd_inverse(H):
        n = H.shape[0]
        L = np.zeros((n, n))
        if not _chol(H, n, L):
            return L, False
        Hinv = np.empty((n, n))
        e = np.zeros(n)
        col = np.empty(n)
        for j in range(n):
            e[:] = 0.0
            e[j] = 1.0
            _chol_solve(L, n, e, col)
            Hinv[:, j] = col
        return Hinv, True

    # ------------------------------------------------------- stacked constraints
    # Row i of the virtual matrix [C ; +I_b ; -I_b] x >= [d ; lb ; -ub].
    @_jit
    def _row_dot(i, x, C, bidx, m, k):
        if i < m:
            s = 0.0
            for j in range(x.shape[0]):
                s += C[i, j] * x[j]
            return s
        if i < m + k:
            return x[bidx[i - m]]
        return -x[bidx[i - m - k]]

    @_jit
    def _row_rhs(i, d, lb, ub, m, k):
        if i < m:
            return d[i]
        if i < m + k:
            return lb[i - m]
        return -ub[i - m - k]

    @_jit
    def _hinv_row(i, L, C, bidx, m, k, out, tmp):
        """out = H^-1 a_i using the Cholesky factor L (two triangular solves, O(n^2))."""
        n = L.shape[0]
        if i < m:
            for r in range(n):
                tmp[r] = C[i, r]
        else:
            for r in range(n):
                tmp[r] = 0.0
            if i < m + k:
                tmp[bidx[i - m]] = 1.0
            else:
                tmp[bidx[i - m - k]] = -1.0
        _chol_solve(L, n, tmp, out)

    @_jit
    def _most_violated(x, C, d, bidx, lb, ub, m, k):
        best, which = 0.0, -1
        for i in range(m + 2 * k):
            s = _row_dot(i, x, C, bidx, m, k) - _row_rhs(i, d, lb, ub, m, k)
            if s < best:
                best, which = s, i
        return which, best

    @_jit
    def _build_M(act, na, HN, C, bidx, m, k, M):
        for a in range(na):
            for c in range(na):
                M[a, c] = _row_dot(act[a], HN[:, c], C, bidx, m, k)

    @_jit
    def gi_qp(H, g, C, d, bidx, lb, ub, active0, tol, max_iter):
        n = g.shape[0]
        m = C.shape[0]
        k = bidx.shape[0]
        # Only the Cholesky factor is formed; H^-1 is applied on demand (O(n^2) per vector)
        # instead of building the full inverse (O(n^3)) on every call.
        L = np.zeros((n, n))
        ok = _chol(H, n, L)
        x = np.zeros(n)
        act = np.empty(n + 1, np.int64)
        if not ok:
            return x, 3, 0, act, 0
        Hg = np.empty(n)
        _chol_solve(L, n, g, Hg)
        for r in range(n):
            x[r] = -Hg[r]
        tmp = np.empty(n)

        u = np.zeros(n + 1)
        HN = np.zeros((n, n + 1))
        M = np.zeros((n + 1, n + 1))
        Lm = np.zeros((n + 1, n + 1))
        rhs = np.zeros(n + 1)
        r = np.zeros(n + 1)
        z = np.empty(n)
        Hn = np.empty(n)
        na = 0
        it = 0
        mt = m + 2 * k

        # ---- warm start ------------------------------------------------------
        for w in active0:
            if w < mt and na < n and np.isfinite(_row_rhs(w, d, lb, ub, m, k)):
                act[na] = w
                na += 1
        while na > 0:
            for a in range(na):
                _hinv_row(act[a], L, C, bidx, m, k, HN[:, a], tmp)
            _build_M(act, na, HN, C, bidx, m, k, M)
            for a in range(na):
                rhs[a] = _row_rhs(act[a], d, lb, ub, m, k) + _row_dot(act[a], Hg, C, bidx, m, k)
            it += 1
            if not _chol(M, na, Lm):
                na = 0
                break
            _chol_solve(Lm, na, rhs, r)
            keep = 0
            for a in range(na):
                if r[a] >= 0.0:
                    act[keep] = act[a]
                    keep += 1
            if keep == na:
                for i in range(n):
                    s = -Hg[i]
                    for a in range(na):
                        s += HN[i, a] * r[a]
                    x[i] = s
                for a in range(na):
                    u[a] = r[a]
                break
            na = keep

        # ---- dual active-set iterations -------------------------------------
        while True:
            p, sp = _most_violated(x, C, d, bidx, lb, ub, m, k)
            if p < 0 or sp >= -tol:
                return x, 0, it, act, na
            for a in range(na):
                if act[a] == p:
                    return x, 0, it, act, na
            if it >= max_iter:
                return x, 2, it, act, na
            _hinv_row(p, L, C, bidx, m, k, Hn, tmp)
            u_p = 0.0
            while True:
                it += 1
                if na > 0:
                    _build_M(act, na, HN, C, bidx, m, k, M)
                    for a in range(na):
                        rhs[a] = _row_dot(act[a], Hn, C, bidx, m, k)
                    if not _chol(M, na, Lm):
                        return x, 2, it, act, na
                    _chol_solve(Lm, na, rhs, r)
                    for i in range(n):
                        s = Hn[i]
                        for a in range(na):
                            s -= HN[i, a] * r[a]
                        z[i] = s
                else:
                    for i in range(n):
                        z[i] = Hn[i]
                t1 = np.inf
                kdrop = -1
                for a in range(na):
                    if r[a] > 1e-14:
                        tj = u[a] / r[a]
                        if tj < t1:
                            t1 = tj
                            kdrop = a
                nz = _row_dot(p, z, C, bidx, m, k)
                if nz > 1e-14:
                    t2 = (_row_rhs(p, d, lb, ub, m, k) - _row_dot(p, x, C, bidx, m, k)) / nz
                else:
                    t2 = np.inf
                t = t1 if t1 < t2 else t2
                if t == np.inf:
                    return x, 1, it, act, na
                if t2 != np.inf:
                    for i in range(n):
                        x[i] += t * z[i]
                for a in range(na):
                    u[a] -= t * r[a]
                u_p += t
                if t2 != np.inf and t == t2:
                    act[na] = p
                    u[na] = u_p
                    HN[:, na] = Hn
                    na += 1
                    break
                for a in range(kdrop, na - 1):
                    act[a] = act[a + 1]
                    u[a] = u[a + 1]
                    HN[:, a] = HN[:, a + 1]
                na -= 1

    # ------------------------------------------------------------- IK kernels
    @_jit
    def points(xpos, xmat, body, local, out):
        for i in range(body.shape[0]):
            b = body[i]
            for r in range(3):
                out[i, r] = xpos[b, r] + xmat[b, 3 * r] * local[i, 0] + xmat[b, 3 * r + 1] * local[i, 1] \
                    + xmat[b, 3 * r + 2] * local[i, 2]

    @_jit
    def assemble(Jt, pos, targets, w3, D, w_smooth, s, w_reg, e, H, g):
        """H = Jt' W Jt + diag(D);  g = Jt' W r + w_s*s + [0, w_reg*e];  returns cost."""
        nv = H.shape[0]
        S = targets.shape[0]
        cost = 0.0
        for i in range(nv):
            g[i] = w_smooth[i] * s[i]
            cost += w_smooth[i] * s[i] * s[i]
            for j in range(i, nv):
                H[i, j] = 0.0
        for i in range(e.shape[0]):
            g[6 + i] += w_reg[i] * e[i]
            cost += w_reg[i] * e[i] * e[i]
        for p in range(S):
            for c in range(3):
                row = 3 * p + c
                w = w3[row]
                if w == 0.0:
                    continue
                res = pos[p, c] - targets[p, c]
                cost += w * res * res
                wr = w * res
                for i in range(nv):
                    a = Jt[row, i]
                    if a != 0.0:
                        g[i] += a * wr
                        wa = w * a
                        for j in range(i, nv):
                            H[i, j] += wa * Jt[row, j]
        for i in range(nv):
            H[i, i] += D[i]
            for j in range(i + 1, nv):
                H[j, i] = H[i, j]
        return 0.5 * cost

    @_jit
    def ground_values(pos, S, F, radius, h, out):
        P = pos.shape[0]
        for i in range(S, P):
            j = i - S
            v = pos[i, 2] - h
            if j >= F:
                v -= radius[j - F]
            out[j] = v

    @_jit
    def ground_rows(rows, pos, S, xpos, cbody, pt_map, Jp_u, Jr_u, C):
        """C[j] = z row of the Jacobian of ground point rows[j]."""
        nv = C.shape[1]
        for j in range(rows.shape[0]):
            gidx = rows[j]
            u = pt_map[S + gidx]
            b = cbody[gidx]
            ox = pos[S + gidx, 0] - xpos[b, 0]
            oy = pos[S + gidx, 1] - xpos[b, 1]
            for v in range(nv):
                C[j, v] = Jp_u[u, 2, v] + Jr_u[u, 0, v] * oy - Jr_u[u, 1, v] * ox

    @_jit
    def merit_cost(pos, targets, w, e, w_reg, s, w_smooth):
        cost = 0.0
        for p in range(targets.shape[0]):
            acc = 0.0
            for c in range(3):
                r = pos[p, c] - targets[p, c]
                acc += r * r
            cost += w[p] * acc
        for i in range(e.shape[0]):
            cost += w_reg[i] * e[i] * e[i]
        for i in range(s.shape[0]):
            cost += w_smooth[i] * s[i] * s[i]
        return 0.5 * cost

    # ------------------------------------------------- fused IK linearisation
    # Re-implements mj_jac's chain walk (engine_support.c) on top of d.cdof / d.subtree_com,
    # so every point Jacobian comes from one compiled call instead of one MuJoCo call each:
    #   jacp[:, i] = cdof_i[3:6] + cdof_i[0:3] x (p - subtree_com[root(body)])   for dofs i in chain(body)
    @_jit
    def _point_positions(xpos, xmat, body, local, pos):
        for i in range(body.shape[0]):
            b = body[i]
            for r in range(3):
                pos[i, r] = xpos[b, r] + xmat[b, 3 * r] * local[i, 0] + xmat[b, 3 * r + 1] * local[i, 1] \
                    + xmat[b, 3 * r + 2] * local[i, 2]

    @_jit
    def linearize_all(xpos, xmat, cdof, subtree_com, pt_body, pt_local, pt_root, chain, chain_len,
                      S, F, cap_radius, h, targets, w3, D, w_smooth, s, w_reg, e,
                      pos, Jt, H, g, c, C):
        """Fills pos, Jt (3S,nv), H, g, ground values c and their z-Jacobian rows C. Returns cost."""
        P = pt_body.shape[0]
        nv = H.shape[0]
        _point_positions(xpos, xmat, pt_body, pt_local, pos)
        # Tracking Jacobians.
        Jt[:, :] = 0.0
        for p in range(S):
            r0 = pt_root[p]
            ox = pos[p, 0] - subtree_com[r0, 0]
            oy = pos[p, 1] - subtree_com[r0, 1]
            oz = pos[p, 2] - subtree_com[r0, 2]
            for t in range(chain_len[p]):
                i = chain[p, t]
                wx, wy, wz = cdof[i, 0], cdof[i, 1], cdof[i, 2]
                Jt[3 * p, i] = cdof[i, 3] + wy * oz - wz * oy
                Jt[3 * p + 1, i] = cdof[i, 4] + wz * ox - wx * oz
                Jt[3 * p + 2, i] = cdof[i, 5] + wx * oy - wy * ox
        # Ground constraints: value and z row.
        C[:, :] = 0.0
        for p in range(S, P):
            j = p - S
            v = pos[p, 2] - h
            if j >= F:
                v -= cap_radius[j - F]
            c[j] = v
            r0 = pt_root[p]
            ox = pos[p, 0] - subtree_com[r0, 0]
            oy = pos[p, 1] - subtree_com[r0, 1]
            for t in range(chain_len[p]):
                i = chain[p, t]
                C[j, i] = cdof[i, 5] + cdof[i, 0] * oy - cdof[i, 1] * ox
        return assemble_chain(Jt, pos, targets, w3, D, w_smooth, s, w_reg, e, H, g, chain, chain_len)

    @_jit
    def assemble_chain(Jt, pos, targets, w3, D, w_smooth, s, w_reg, e, H, g, chain, chain_len):
        """``assemble`` exploiting sparsity: point p's Jacobian is non-zero only on its dof chain,
        so its Gauss-Newton block is accumulated over chain x chain (~12 x 12 instead of nv x nv)."""
        nv = H.shape[0]
        S = targets.shape[0]
        cost = 0.0
        for i in range(nv):
            g[i] = w_smooth[i] * s[i]
            cost += w_smooth[i] * s[i] * s[i]
            for j in range(nv):
                H[i, j] = 0.0
        for i in range(e.shape[0]):
            g[6 + i] += w_reg[i] * e[i]
            cost += w_reg[i] * e[i] * e[i]
        for p in range(S):
            r0, r1, r2 = 3 * p, 3 * p + 1, 3 * p + 2
            w0, w1, w2 = w3[r0], w3[r1], w3[r2]
            e0 = pos[p, 0] - targets[p, 0]
            e1 = pos[p, 1] - targets[p, 1]
            e2 = pos[p, 2] - targets[p, 2]
            cost += w0 * e0 * e0 + w1 * e1 * e1 + w2 * e2 * e2
            L = chain_len[p]
            for a in range(L):
                i = chain[p, a]
                a0, a1, a2 = w0 * Jt[r0, i], w1 * Jt[r1, i], w2 * Jt[r2, i]
                g[i] += a0 * e0 + a1 * e1 + a2 * e2
                for b in range(L):
                    j = chain[p, b]
                    if j >= i:
                        H[i, j] += a0 * Jt[r0, j] + a1 * Jt[r1, j] + a2 * Jt[r2, j]
        for i in range(nv):
            H[i, i] += D[i]
            for j in range(i + 1, nv):
                H[j, i] = H[i, j]
        return 0.5 * cost

    # ---------------------------------------- configuration space (free joint + hinge/slide)
    @_jit
    def qdiff(q0, q1, out):
        """mj_differentiatePos for [free joint | hinge/slide joints]: q1 = qint(q0, out)."""
        for i in range(3):
            out[i] = q1[i] - q0[i]
        # conj(q0) * q1
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

    @_jit
    def qint(q, dv, alpha, out):
        """mj_integratePos(q, alpha * dv) for [free joint | hinge/slide joints] into out."""
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

    @_jit
    def step_terms(q, q_ref, q_nom, q_prev, qmin, qmax, cap, s, e, lb, ub):
        """Smoothing residual s = q_ref -> q, regularisation e, and joint step bounds
        (limits, intersected with +-cap around q_prev when cap >= 0)."""
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

    @_jit
    def neg_viol(c, dneg):
        """dneg = -c; returns the L1 violation sum(max(0, -c))."""
        v = 0.0
        for i in range(c.shape[0]):
            dneg[i] = -c[i]
            if c[i] < 0.0:
                v -= c[i]
        return v

    @_jit
    def merit_q(xpos, xmat, pt_body, pt_local, S, F, cap_radius, h, targets, w, w_reg, w_smooth,
                q, q_ref, q_nom, s, e, pos):
        qdiff(q_ref, q, s)
        for i in range(e.shape[0]):
            e[i] = q[7 + i] - q_nom[7 + i]
        return merit_all(xpos, xmat, pt_body, pt_local, S, F, cap_radius, h, targets, w, e, w_reg, s,
                         w_smooth, pos)

    @_jit
    def merit_all(xpos, xmat, pt_body, pt_local, S, F, cap_radius, h, targets, w, e, w_reg, s, w_smooth, pos):
        _point_positions(xpos, xmat, pt_body, pt_local, pos)
        cost = merit_cost(pos, targets, w, e, w_reg, s, w_smooth)
        viol = 0.0
        for p in range(S, pt_body.shape[0]):
            j = p - S
            v = pos[p, 2] - h
            if j >= F:
                v -= cap_radius[j - F]
            if v < 0.0:
                viol -= v
        return cost, viol

if AVAILABLE:
    @_jit
    def bvh_fk(parents, offsets, chan_start, chan_n, chan_kind, chan_axis, data, pos, rot):
        """BVH forward kinematics: pos (T,J,3), rot (T,J,3,3) world frames.

        chan_kind[k] 0 = position, 1 = rotation; chan_axis[k] 0/1/2 = X/Y/Z, for data column k.
        Rotation channels compose left to right (R = R_c1 @ R_c2 @ ...)."""
        T, J = data.shape[0], parents.shape[0]
        R = np.empty((3, 3))
        tmp = np.empty((3, 3))
        tv = np.empty(3)
        for t in range(T):
            for j in range(J):
                for a in range(3):
                    tv[a] = offsets[j, a]
                    for b in range(3):
                        R[a, b] = 1.0 if a == b else 0.0
                for c in range(chan_n[j]):
                    k = chan_start[j] + c
                    v = data[t, k]
                    ax = chan_axis[k]
                    if chan_kind[k] == 0:
                        tv[ax] += v
                        continue
                    ang = v * (np.pi / 180.0)
                    co, si = np.cos(ang), np.sin(ang)
                    i1 = (ax + 1) % 3
                    i2 = (ax + 2) % 3
                    # R <- R @ Rc,  Rc: rotation about axis ax (columns i1, i2 mix)
                    for a in range(3):
                        x, y = R[a, i1], R[a, i2]
                        R[a, i1] = x * co + y * si
                        R[a, i2] = -x * si + y * co
                p = parents[j]
                if p < 0:
                    for a in range(3):
                        pos[t, j, a] = tv[a]
                        for b in range(3):
                            rot[t, j, a, b] = R[a, b]
                else:
                    for a in range(3):
                        pos[t, j, a] = pos[t, p, a] + rot[t, p, a, 0] * tv[0] + rot[t, p, a, 1] * tv[1] \
                            + rot[t, p, a, 2] * tv[2]
                        for b in range(3):
                            tmp[a, b] = rot[t, p, a, 0] * R[0, b] + rot[t, p, a, 1] * R[1, b] + rot[t, p, a, 2] * R[2, b]
                    for a in range(3):
                        for b in range(3):
                            rot[t, j, a, b] = tmp[a, b]

if AVAILABLE:
    @_jit
    def select_rows(c, C, act, glob_active, Csub, dsub, idx, local_of, a0):
        """Rows with c < act (QP constraint C dv >= -c) -> Csub/dsub; idx = their global ids,
        local_of = global -> local (-1 if pruned). The warm-start active set (global ids:
        rows < len(c), bounds len(c) + j) is mapped to local ids in a0. Returns (k, n_a0)."""
        m, nv = C.shape
        k = 0
        for i in range(m):
            if c[i] < act:
                for j in range(nv):
                    Csub[k, j] = C[i, j]
                dsub[k] = -c[i]
                idx[k] = i
                local_of[i] = k
                k += 1
            else:
                local_of[i] = -1
        na = 0
        for a in glob_active:
            if a < m:
                if local_of[a] >= 0:
                    a0[na] = local_of[a]
                    na += 1
            else:
                a0[na] = k + (a - m)
                na += 1
        return k, na

    @_jit
    def pruned_ok(c, C, act, dv, tol):
        """True if every pruned row (c >= act) still satisfies c + C dv >= -tol."""
        m, nv = C.shape
        for i in range(m):
            if c[i] >= act:
                s = c[i]
                for j in range(nv):
                    s += C[i, j] * dv[j]
                if s < -tol:
                    return False
        return True

    @_jit
    def to_global(act, na, k, idx, m, out):
        for i in range(na):
            a = act[i]
            out[i] = idx[a] if a < k else m + (a - k)
        return out[:na].copy()

if AVAILABLE:
    _POW10 = np.array([10.0 ** i for i in range(23)])

    # No fastmath here: m / 10^k must stay a correctly rounded division (that is what makes the
    # result identical to strtod / np.fromstring).
    @nb.njit(cache=True, nogil=True)
    def parse_decimals(buf, start, out):
        """Parse whitespace-separated decimals ([+-]digits[.digits][e[+-]digits]) from buf[start:]
        into out. Exact (correctly rounded): mantissa < 2^53 and |power of ten| <= 22, so the value
        is one IEEE division/multiplication of two exactly representable numbers. Returns the count,
        or -1 for anything else (caller falls back to numpy)."""
        n = buf.shape[0]
        i = start
        k = 0
        pw = _POW10
        while i < n:
            c = buf[i]
            if c == 32 or c == 10 or c == 13 or c == 9 or c == 11 or c == 12:
                i += 1
                continue
            neg = False
            if c == 45:
                neg = True
                i += 1
            elif c == 43:
                i += 1
            mant = 0
            nd = 0
            frac = 0
            while i < n and 48 <= buf[i] <= 57:
                if nd < 19:
                    mant = mant * 10 + (buf[i] - 48)
                nd += 1
                i += 1
            if i < n and buf[i] == 46:
                i += 1
                while i < n and 48 <= buf[i] <= 57:
                    if nd < 19:
                        mant = mant * 10 + (buf[i] - 48)
                    nd += 1
                    frac += 1
                    i += 1
            if nd == 0 or nd > 15 and mant >= 9007199254740992 or nd >= 19:
                return -1
            e = 0
            if i < n and (buf[i] == 101 or buf[i] == 69):
                i += 1
                eneg = False
                if i < n and buf[i] == 45:
                    eneg = True
                    i += 1
                elif i < n and buf[i] == 43:
                    i += 1
                ne = 0
                while i < n and 48 <= buf[i] <= 57:
                    e = e * 10 + (buf[i] - 48)
                    ne += 1
                    i += 1
                    if e > 400:
                        return -1
                if ne == 0:
                    return -1
                if eneg:
                    e = -e
            if i < n:
                c = buf[i]
                if not (c == 32 or c == 10 or c == 13 or c == 9 or c == 11 or c == 12):
                    return -1
            p10 = e - frac
            if p10 == 0:
                v = float(mant)
            elif -22 <= p10 < 0:
                v = float(mant) / pw[-p10]
            elif 0 < p10 <= 22:
                v = float(mant) * pw[p10]
            else:
                return -1
            if k >= out.shape[0]:
                return -1
            out[k] = -v if neg else v
            k += 1
        return k

    @_jit
    def rotate_points(P, R, s, out):
        for i in range(P.shape[0]):
            x, y, z = P[i, 0], P[i, 1], P[i, 2]
            for r in range(3):
                out[i, r] = s * (R[r, 0] * x + R[r, 1] * y + R[r, 2] * z)

    @_jit
    def bvh_fk_sel(parents, offsets, chan_start, chan_n, chan_kind, chan_axis, data, slot, pos, rot):
        """bvh_fk, but world rotations are kept only for joints with slot[j] >= 0 (rot[:, slot[j]])."""
        T, J = data.shape[0], parents.shape[0]
        R = np.empty((3, 3))
        W = np.empty((J, 3, 3))
        tv = np.empty(3)
        for t in range(T):
            for j in range(J):
                for a in range(3):
                    tv[a] = offsets[j, a]
                    for b in range(3):
                        R[a, b] = 1.0 if a == b else 0.0
                for c in range(chan_n[j]):
                    k = chan_start[j] + c
                    v = data[t, k]
                    ax = chan_axis[k]
                    if chan_kind[k] == 0:
                        tv[ax] += v
                        continue
                    ang = v * (np.pi / 180.0)
                    co, si = np.cos(ang), np.sin(ang)
                    i1 = (ax + 1) % 3
                    i2 = (ax + 2) % 3
                    for a in range(3):
                        x, y = R[a, i1], R[a, i2]
                        R[a, i1] = x * co + y * si
                        R[a, i2] = -x * si + y * co
                p = parents[j]
                if p < 0:
                    for a in range(3):
                        pos[t, j, a] = tv[a]
                        for b in range(3):
                            W[j, a, b] = R[a, b]
                else:
                    for a in range(3):
                        pos[t, j, a] = pos[t, p, a] + W[p, a, 0] * tv[0] + W[p, a, 1] * tv[1] + W[p, a, 2] * tv[2]
                        for b in range(3):
                            W[j, a, b] = W[p, a, 0] * R[0, b] + W[p, a, 1] * R[1, b] + W[p, a, 2] * R[2, b]
                sl = slot[j]
                if sl >= 0:
                    for a in range(3):
                        for b in range(3):
                            rot[t, sl, a, b] = W[j, a, b]

    @_jit
    def limb_plan(P, joint, anchor, kind, ratio, rot_id, off, R, out):
        """motion.limb_rescale's joint assignments, frame by frame (see motion._run_plan)."""
        T = P.shape[0]
        for t in range(T):
            for n in range(joint.shape[0]):
                j, a, k = joint[n], anchor[n], kind[n]
                if k == 2:
                    for c in range(3):
                        out[t, j, c] = ratio[n] * P[t, j, c]
                elif k == 1:
                    Rt = R[rot_id[n], t]
                    for c in range(3):
                        out[t, j, c] = out[t, a, c] + (Rt[c, 0] * off[n, 0] + Rt[c, 1] * off[n, 1] + Rt[c, 2] * off[n, 2])
                else:
                    r = ratio[n]
                    for c in range(3):
                        out[t, j, c] = out[t, a, c] + r * (P[t, j, c] - P[t, a, c])

    @_jit
    def finish_targets(P, fac, idx, shift, dz, offset, extra, X):
        for t in range(P.shape[0]):
            for n in range(idx.shape[0]):
                j = idx[n]
                for c in range(3):
                    v = P[t, j, c] * fac[c] - shift[c]
                    if c == 2:
                        v -= dz
                    X[t, n, c] = (v + offset[c]) + extra[c]
