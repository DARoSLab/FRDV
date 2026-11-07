/* frdv native kernels: the fused per-clip solver (see _fused.py for the numba version and the
 * math) plus the BVH / target-building helpers. Built on first use by frdv/_native.py with the
 * system C compiler and called through ctypes (which releases the GIL), so clips run on plain
 * threads with no per-frame Python.
 *
 * Everything mirrors _fused.py line by line; only the code generation differs (no tuple argument
 * marshalling, no index wraparound checks, no refcounting).
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int64_t i64;

#if defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

typedef struct {
    i64 nbody, nv, nq, nj, S, G, Lc, ngb;
    const i64 *bparent, *bjnt0, *bjntn, *jtype, *jqadr, *jdadr, *dparent, *dbody, *bchain, *bchain_len,
        *pb, *anc, *nanc, *gbody, *gofs, *gpt, *gpb, *bdof0, *bdofn;
    const double *bpos, *bquat, *jaxis, *jpos, *qpos0, *pl, *grad, *wpt, *q_nom, *qmin, *qmax, *reach,
        *lmax, *p0;
    double h;
    const double *bmat;   /* (nbody, 9) rotation of body_quat */
    const i64 *bflag;     /* bit 0: body_quat is identity; bit 1: single hinge with jnt_pos = 0 */
    /* The ancestors of a dof (and the dof chain of a body) form a few runs of consecutive indices.
     * arng[k] = runs (lo, hi) of dof k's ancestors, nearest first; brng[b] = runs of body b's chain
     * (its own dofs included). Matrices are stored dense so these runs are contiguous rows. */
    i64 maxr;
    const i64 *arng, *narng, *brng, *nbrng;
} Model;

#define ARNG(m, k, r, e) ((m)->arng[((k) * (m)->maxr + (r)) * 2 + (e)])
#define BRNG(m, b, r, e) ((m)->brng[((b) * (m)->maxr + (r)) * 2 + (e)])

typedef struct {
    const double *ws, *D, *wreg;
    double cap;
    const i64 *bidx;
    i64 nb;  /* number of bounded variables */
} Prm;

typedef struct {
    i64 max_iter;
    double tol;
    i64 line_search;
    double mu_init, c1, tau;
    i64 max_ls;
    double qp_tol, act_lim;
} Opt;

typedef struct {
    /* kinematics */
    double *xpos, *xquat, *xmat, *dax, *dpt, *sw, *sv, *pos, *fkq;
    i64 *dkind, *near;
    int fk_dofs, fk_pts;
    /* linearisation */
    double *Hc, *Hd, *g, *c, *bK, *bF, *Vb, *s, *e, *lb, *ub;
    /* constraint rows */
    double *Csub, *dsub, *Cfull, *dneg;
    i64 *idx, *local_of, *a0, *rbs;
    /* qp */
    double *x, *F, *Dg, *wg, *xi, *u, *Wa, *M, *Lm, *rhs, *r, *zeta, *Wp, *isd;
    i64 *act;
    /* qp warm start (global ids), persists across calls */
    i64 *actg;
    i64 nact;
    /* frame / clip temporaries */
    double *q, *qt, *qa, *qb, *qref, *tmp, *cz;
    void *block;
} Work;

/* ------------------------------------------------------------------ small helpers */
static inline double dmin(double a, double b) { return a < b ? a : b; }
static inline double dmax(double a, double b) { return a > b ? a : b; }

void qdiff(const double *q0, const double *q1, double *out, i64 nq) {
    for (int i = 0; i < 3; i++) out[i] = q1[i] - q0[i];
    double aw = q0[3], ax = -q0[4], ay = -q0[5], az = -q0[6];
    double bw = q1[3], bx = q1[4], by = q1[5], bz = q1[6];
    double w = aw * bw - ax * bx - ay * by - az * bz;
    double x = aw * bx + bw * ax + ay * bz - az * by;
    double y = aw * by + bw * ay + az * bx - ax * bz;
    double z = aw * bz + bw * az + ax * by - ay * bx;
    double sn = sqrt(x * x + y * y + z * z);
    if (sn < 1e-15) {
        out[3] = out[4] = out[5] = 0.0;
    } else {
        double ang = 2.0 * atan2(sn, w);
        if (ang > M_PI) ang -= 2.0 * M_PI;
        double f = ang / sn;
        out[3] = x * f; out[4] = y * f; out[5] = z * f;
    }
    for (i64 i = 7; i < nq; i++) out[i - 1] = q1[i] - q0[i];
}

void qint(const double *q, const double *dv, double alpha, double *out, i64 nq) {
    for (int i = 0; i < 3; i++) out[i] = q[i] + alpha * dv[i];
    double vx = dv[3], vy = dv[4], vz = dv[5];
    double nrm = sqrt(vx * vx + vy * vy + vz * vz);
    double aw = q[3], ax = q[4], ay = q[5], az = q[6], w, x, y, z;
    if (nrm < 1e-15) {
        w = aw; x = ax; y = ay; z = az;
    } else {
        double half = 0.5 * alpha * nrm;
        double sh = sin(half) / nrm;
        double bw = cos(half), bx = vx * sh, by = vy * sh, bz = vz * sh;
        w = aw * bw - ax * bx - ay * by - az * bz;
        x = aw * bx + bw * ax + ay * bz - az * by;
        y = aw * by + bw * ay + az * bx - ax * bz;
        z = aw * bz + bw * az + ax * by - ay * bx;
    }
    double n4 = sqrt(w * w + x * x + y * y + z * z);
    out[3] = w / n4; out[4] = x / n4; out[5] = y / n4; out[6] = z / n4;
    for (i64 i = 7; i < nq; i++) out[i] = q[i] + alpha * dv[i - 1];
}

static void step_terms(const Model *m, const double *q, const double *q_ref, const double *q_prev, double cap,
                       double *s, double *e, double *lb, double *ub) {
    qdiff(q_ref, q, s, m->nq);
    for (i64 i = 0; i < m->nj; i++) {
        double qj = q[7 + i];
        e[i] = qj - m->q_nom[7 + i];
        double lo = m->qmin[i], hi = m->qmax[i];
        if (cap >= 0.0) {
            double qp = dmin(dmax(q_prev[7 + i], lo), hi);
            lo = dmax(lo, qp - cap);
            hi = dmin(hi, qp + cap);
        }
        lb[i] = lo - qj;
        ub[i] = hi - qj;
    }
}

/* ------------------------------------------------------------------ kinematics */
static inline void rotvec(const double *q, double v0, double v1, double v2, double *o) {
    if (v0 == 0.0 && v1 == 0.0 && v2 == 0.0) { o[0] = o[1] = o[2] = 0.0; return; }
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double t0 = w * v0 + y * v2 - z * v1;
    double t1 = w * v1 + z * v0 - x * v2;
    double t2 = w * v2 + x * v1 - y * v0;
    o[0] = v0 + 2.0 * (y * t2 - z * t1);
    o[1] = v1 + 2.0 * (z * t0 - x * t2);
    o[2] = v2 + 2.0 * (x * t1 - y * t0);
}

static inline void quat2mat(const double *q, double *m) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    if (w == 1.0 && x == 0.0 && y == 0.0 && z == 0.0) {
        for (int i = 0; i < 9; i++) m[i] = 0.0;
        m[0] = m[4] = m[8] = 1.0;
        return;
    }
    double q00 = w * w, q01 = w * x, q02 = w * y, q03 = w * z;
    double q11 = x * x, q12 = x * y, q13 = x * z, q22 = y * y, q23 = y * z, q33 = z * z;
    m[0] = q00 + q11 - q22 - q33;
    m[4] = q00 - q11 + q22 - q33;
    m[8] = q00 - q11 - q22 + q33;
    m[1] = 2.0 * (q12 - q03);
    m[2] = 2.0 * (q13 + q02);
    m[3] = 2.0 * (q12 + q03);
    m[5] = 2.0 * (q23 - q01);
    m[6] = 2.0 * (q13 - q02);
    m[7] = 2.0 * (q23 + q01);
}

static inline void normalize4(double *q) {
    double n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n < 1e-15) { q[0] = 1.0; q[1] = q[2] = q[3] = 0.0; }
    else if (fabs(n - 1.0) > 1e-15) { q[0] /= n; q[1] /= n; q[2] /= n; q[3] /= n; }
}

static inline void mulquat(const double *a, const double *b, double *out) {
    double w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    double x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    double y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    double z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
    out[0] = w; out[1] = x; out[2] = y; out[3] = z;
}

/* mj_kinematics for [free | hinge | slide] trees; with dofs, every dof's world axis and anchor.
 * Bodies with one hinge at their origin (the common case) use rotation matrices directly:
 * R = R_parent R_body Rot(axis, q - q0); others follow mj_kinematics' quaternion path. */
static void fk(const Model *m, Work *W, const double *q, int dofs) {
    double *xpos = W->xpos, *xquat = W->xquat, *xmat = W->xmat, *dax = W->dax, *dpt = W->dpt;
    i64 *dkind = W->dkind;
    memcpy(W->fkq, q, sizeof(double) * m->nq);
    W->fk_dofs = dofs;
    W->fk_pts = 0;
    xpos[0] = xpos[1] = xpos[2] = 0.0;
    xquat[0] = 1.0; xquat[1] = xquat[2] = xquat[3] = 0.0;
    quat2mat(xquat, xmat);
    for (i64 i = 1; i < m->nbody; i++) {
        i64 p = m->bparent[i], j0 = m->bjnt0[i], nj = m->bjntn[i];
        double *xq = xquat + 4 * i, *xp = xpos + 3 * i, *xm = xmat + 9 * i;
        if (nj > 0 && m->jtype[j0] == 0) {
            i64 qa = m->jqadr[j0], da = m->jdadr[j0];
            xp[0] = q[qa]; xp[1] = q[qa + 1]; xp[2] = q[qa + 2];
            xq[0] = q[qa + 3]; xq[1] = q[qa + 4]; xq[2] = q[qa + 5]; xq[3] = q[qa + 6];
            normalize4(xq);
            quat2mat(xq, xm);
            if (dofs) {
                for (int k = 0; k < 3; k++) {
                    dkind[da + k] = 1;
                    dax[3 * (da + k)] = k == 0; dax[3 * (da + k) + 1] = k == 1; dax[3 * (da + k) + 2] = k == 2;
                    dkind[da + 3 + k] = 0;
                    dax[3 * (da + 3 + k)] = xm[k];
                    dax[3 * (da + 3 + k) + 1] = xm[3 + k];
                    dax[3 * (da + 3 + k) + 2] = xm[6 + k];
                    for (int a = 0; a < 3; a++) dpt[3 * (da + 3 + k) + a] = xp[a];
                }
            }
            continue;
        }
        const double *pm = xmat + 9 * p, *pp = xpos + 3 * p, *bp = m->bpos + 3 * i;
        for (int a = 0; a < 3; a++)
            xp[a] = pp[a] + pm[3 * a] * bp[0] + pm[3 * a + 1] * bp[1] + pm[3 * a + 2] * bp[2];
        if (m->bflag[i] & 2) {
            /* no joint, or one hinge at the body origin (set for every body or none, so parents
             * on this path never need xquat) */
            double Rpre[9];
            if (m->bflag[i] & 1) memcpy(Rpre, pm, sizeof(Rpre));
            else {
                const double *B = m->bmat + 9 * i;
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        Rpre[3 * r + c] = pm[3 * r] * B[c] + pm[3 * r + 1] * B[3 + c] + pm[3 * r + 2] * B[6 + c];
            }
            if (nj == 0) { memcpy(xm, Rpre, sizeof(Rpre)); continue; }
            i64 qa = m->jqadr[j0], da = m->jdadr[j0];
            const double *ja = m->jaxis + 3 * j0;
            double ang = q[qa] - m->qpos0[qa];
            double sn = sin(ang), cs = cos(ang), vc = 1.0 - cs, x = ja[0], y = ja[1], z = ja[2];
            double Rl[9] = {cs + x * x * vc, x * y * vc - z * sn, x * z * vc + y * sn,
                            y * x * vc + z * sn, cs + y * y * vc, y * z * vc - x * sn,
                            z * x * vc - y * sn, z * y * vc + x * sn, cs + z * z * vc};
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    xm[3 * r + c] = Rpre[3 * r] * Rl[c] + Rpre[3 * r + 1] * Rl[3 + c] + Rpre[3 * r + 2] * Rl[6 + c];
            if (dofs) {
                dkind[da] = 0;
                for (int a = 0; a < 3; a++) {
                    dax[3 * da + a] = Rpre[3 * a] * x + Rpre[3 * a + 1] * y + Rpre[3 * a + 2] * z;
                    dpt[3 * da + a] = xp[a];
                }
            }
            continue;
        }
        mulquat(xquat + 4 * p, m->bquat + 4 * i, xq);
        for (i64 jj = 0; jj < nj; jj++) {
            i64 jid = j0 + jj, qa = m->jqadr[jid], da = m->jdadr[jid];
            const double *ja = m->jaxis + 3 * jid, *jp = m->jpos + 3 * jid;
            double ax[3], v[3], an[3];
            rotvec(xq, ja[0], ja[1], ja[2], ax);
            rotvec(xq, jp[0], jp[1], jp[2], v);
            an[0] = xp[0] + v[0]; an[1] = xp[1] + v[1]; an[2] = xp[2] + v[2];
            double ang = q[qa] - m->qpos0[qa];
            if (m->jtype[jid] == 2) { /* slide */
                xp[0] += ax[0] * ang; xp[1] += ax[1] * ang; xp[2] += ax[2] * ang;
                if (dofs) dkind[da] = 1;
            } else {                  /* hinge */
                if (ang != 0.0) {
                    double sh = sin(0.5 * ang), ql[4], tq[4];
                    ql[0] = cos(0.5 * ang); ql[1] = ja[0] * sh; ql[2] = ja[1] * sh; ql[3] = ja[2] * sh;
                    mulquat(xq, ql, tq);
                    xq[0] = tq[0]; xq[1] = tq[1]; xq[2] = tq[2]; xq[3] = tq[3];
                }
                rotvec(xq, jp[0], jp[1], jp[2], v);
                xp[0] = an[0] - v[0]; xp[1] = an[1] - v[1]; xp[2] = an[2] - v[2];
                if (dofs) dkind[da] = 0;
            }
            if (dofs) {
                dax[3 * da] = ax[0]; dax[3 * da + 1] = ax[1]; dax[3 * da + 2] = ax[2];
                dpt[3 * da] = an[0]; dpt[3 * da + 1] = an[1]; dpt[3 * da + 2] = an[2];
            }
        }
        normalize4(xq);
        quat2mat(xq, xm);
    }
}

static void points(const Model *m, const Work *W, double *pos, i64 n) {
    for (i64 i = 0; i < n; i++) {
        i64 b = m->pb[i];
        const double *xm = W->xmat + 9 * b, *xp = W->xpos + 3 * b, *l = m->pl + 3 * i;
        for (int r = 0; r < 3; r++)
            pos[3 * i + r] = xp[r] + xm[3 * r] * l[0] + xm[3 * r + 1] * l[1] + xm[3 * r + 2] * l[2];
    }
}

/* ground values for every body that can come within act of the floor (all if allb); L1 violation */
static double ground_near(const Model *m, Work *W, double act, int allb) {
    const double *xpos = W->xpos, *xmat = W->xmat, *pl = m->pl;
    double *c = W->c, viol = 0.0, h = m->h;
    i64 S = m->S;
    for (i64 n = 0; n < m->ngb; n++) {
        i64 b = m->gbody[n];
        if (allb || xpos[3 * b + 2] - h - m->reach[n] < act) {
            W->near[b] = 1;
            double m6 = xmat[9 * b + 6], m7 = xmat[9 * b + 7], m8 = xmat[9 * b + 8], zb = xpos[3 * b + 2];
            for (i64 u = m->gofs[n]; u < m->gofs[n + 1]; u++) {
                i64 k = m->gpt[u], p = S + k;
                double v = zb + m6 * pl[3 * p] + m7 * pl[3 * p + 1] + m8 * pl[3 * p + 2] - h - m->grad[k];
                c[k] = v;
                if (v < 0.0) viol -= v;
            }
        } else {
            W->near[b] = 0;
        }
    }
    return viol;
}

/* min(0, smallest ground value): only bodies that can reach below the floor are evaluated
 * (despike's window score only uses the violation max(0, -min c)) */
static double ground_min(const Model *m, const Work *W) {
    double mn = 0.0, h = m->h;
    const double *xpos = W->xpos, *pl = m->pl;
    for (i64 n = 0; n < m->ngb; n++) {
        i64 b = m->gbody[n];
        if (xpos[3 * b + 2] - h - m->reach[n] >= 0.0) continue;
        const double *xm = W->xmat + 9 * b;
        for (i64 u = m->gofs[n]; u < m->gofs[n + 1]; u++) {
            i64 k = m->gpt[u];
            const double *l = pl + 3 * (m->S + k);
            double v = xpos[3 * b + 2] + xm[6] * l[0] + xm[7] * l[1] + xm[8] * l[2] - h - m->grad[k];
            if (v < mn) mn = v;
        }
    }
    return mn;
}

/* tree-sparse Gauss-Newton H (compressed: Hd diagonal, Hc[j, t] = H[j, anc[j, t]]) and g */
static double linearize(const Model *m, Work *W, const double *targets, const Prm *P, const double *s,
                        const double *e) {
    i64 nv = m->nv, nb = m->nbody, S = m->S;
    const double *xpos = W->xpos, *dax = W->dax, *dpt = W->dpt, *pos = W->pos;
    double *sw = W->sw, *sv = W->sv, *Hc = W->Hc, *Hd = W->Hd, *g = W->g, *bK = W->bK, *bF = W->bF;
    double *sw0 = sw, *sw1 = sw + nv, *sw2 = sw + 2 * nv, *sv0 = sv, *sv1 = sv + nv, *sv2 = sv + 2 * nv;
    i64 root = m->dbody[0];
    double o0 = xpos[3 * root], o1 = xpos[3 * root + 1], o2 = xpos[3 * root + 2];
    for (i64 i = 0; i < nv; i++) {
        double a0 = dax[3 * i], a1 = dax[3 * i + 1], a2 = dax[3 * i + 2];
        if (W->dkind[i] == 1) {
            sw0[i] = sw1[i] = sw2[i] = 0.0;
            sv0[i] = a0; sv1[i] = a1; sv2[i] = a2;
        } else {
            sw0[i] = a0; sw1[i] = a1; sw2[i] = a2;
            double r0 = o0 - dpt[3 * i], r1 = o1 - dpt[3 * i + 1], r2 = o2 - dpt[3 * i + 2];
            sv0[i] = a1 * r2 - a2 * r1;
            sv1[i] = a2 * r0 - a0 * r2;
            sv2[i] = a0 * r1 - a1 * r0;
        }
    }
    memset(bK, 0, sizeof(double) * 10 * nb);
    memset(bF, 0, sizeof(double) * 6 * nb);
    double cost = 0.0;
    for (i64 p = 0; p < S; p++) {
        double w = m->wpt[p];
        i64 b = m->pb[p];
        double d0 = pos[3 * p] - o0, d1 = pos[3 * p + 1] - o1, d2 = pos[3 * p + 2] - o2;
        double r0 = pos[3 * p] - targets[3 * p], r1 = pos[3 * p + 1] - targets[3 * p + 1],
               r2 = pos[3 * p + 2] - targets[3 * p + 2];
        cost += w * (r0 * r0 + r1 * r1 + r2 * r2);
        double *K = bK + 10 * b, *Fb = bF + 6 * b;
        K[0] += w;
        K[1] += w * d0; K[2] += w * d1; K[3] += w * d2;
        K[4] += w * (d1 * d1 + d2 * d2);
        K[5] += w * (d0 * d0 + d2 * d2);
        K[6] += w * (d0 * d0 + d1 * d1);
        K[7] -= w * d0 * d1;
        K[8] -= w * d0 * d2;
        K[9] -= w * d1 * d2;
        Fb[0] += w * (d1 * r2 - d2 * r1);
        Fb[1] += w * (d2 * r0 - d0 * r2);
        Fb[2] += w * (d0 * r1 - d1 * r0);
        Fb[3] += w * r0; Fb[4] += w * r1; Fb[5] += w * r2;
    }
    for (i64 b = nb - 1; b > 0; b--) {
        i64 p = m->bparent[b];
        for (int k = 0; k < 10; k++) bK[10 * p + k] += bK[10 * b + k];
        for (int k = 0; k < 6; k++) bF[6 * p + k] += bF[6 * b + k];
    }
    for (i64 j = 0; j < nv; j++) {
        i64 b = m->dbody[j];
        const double *K = bK + 10 * b, *Fb = bF + 6 * b;
        double Wt = K[0], m0 = K[1], m1 = K[2], m2 = K[3];
        double Qxx = K[4], Qyy = K[5], Qzz = K[6], Qxy = K[7], Qxz = K[8], Qyz = K[9];
        double w0 = sw0[j], w1 = sw1[j], w2 = sw2[j];
        double v0 = sv0[j], v1 = sv1[j], v2 = sv2[j];
        double t0 = Qxx * w0 + Qxy * w1 + Qxz * w2 + (m1 * v2 - m2 * v1);
        double t1 = Qxy * w0 + Qyy * w1 + Qyz * w2 + (m2 * v0 - m0 * v2);
        double t2 = Qxz * w0 + Qyz * w1 + Qzz * w2 + (m0 * v1 - m1 * v0);
        double t3 = Wt * v0 - (m1 * w2 - m2 * w1);
        double t4 = Wt * v1 - (m2 * w0 - m0 * w2);
        double t5 = Wt * v2 - (m0 * w1 - m1 * w0);
        Hd[j] = w0 * t0 + w1 * t1 + w2 * t2 + v0 * t3 + v1 * t4 + v2 * t5 + P->D[j];
        double *RESTRICT Hrow = Hc + nv * j;
        for (i64 r = 0; r < m->narng[j]; r++)
            for (i64 i = ARNG(m, j, r, 0); i <= ARNG(m, j, r, 1); i++)
                Hrow[i] = sw0[i] * t0 + sw1[i] * t1 + sw2[i] * t2 + sv0[i] * t3 + sv1[i] * t4 + sv2[i] * t5;
        g[j] = w0 * Fb[0] + w1 * Fb[1] + w2 * Fb[2] + v0 * Fb[3] + v1 * Fb[4] + v2 * Fb[5] + P->ws[j] * s[j];
    }
    double creg = 0.0;
    for (i64 i = 0; i < nv; i++) creg += P->ws[i] * s[i] * s[i];
    for (i64 i = 0; i < m->nj; i++) {
        g[6 + i] += P->wreg[i] * e[i];
        creg += P->wreg[i] * e[i] * e[i];
    }
    return 0.5 * (cost + creg);
}

/* z row of the Jacobian of ground point k (dense, nv) */
static void ground_row(const Model *m, const Work *W, i64 k, double *row) {
    i64 nv = m->nv, root = m->dbody[0], p = m->S + k, b = m->pb[p];
    const double *xm = W->xmat + 9 * b, *xp = W->xpos + 3 * b, *l = m->pl + 3 * p;
    memset(row, 0, sizeof(double) * nv);
    double d0 = xp[0] + xm[0] * l[0] + xm[1] * l[1] + xm[2] * l[2] - W->xpos[3 * root];
    double d1 = xp[1] + xm[3] * l[0] + xm[4] * l[1] + xm[5] * l[2] - W->xpos[3 * root + 1];
    const double *sw0 = W->sw, *sw1 = W->sw + nv, *sv2 = W->sv + 2 * nv;
    for (i64 r = 0; r < m->nbrng[b]; r++)
        for (i64 i = BRNG(m, b, r, 0); i <= BRNG(m, b, r, 1); i++) row[i] = sv2[i] + sw0[i] * d1 - sw1[i] * d0;
}

/* every ground row with c >= act still satisfies c + J_z dv >= -tol */
static int pruned_ok(const Model *m, Work *W, double act, const double *dv, double tol) {
    double *Vb = W->Vb;
    const double *xpos = W->xpos, *xmat = W->xmat, *sw = W->sw, *sv = W->sv, *c = W->c, *pl = m->pl;
    i64 root = m->dbody[0], S = m->S;
    for (int a = 0; a < 6; a++) Vb[a] = 0.0;
    for (i64 b = 1; b < m->nbody; b++) {
        i64 p = m->bparent[b];
        for (int a = 0; a < 6; a++) Vb[6 * b + a] = Vb[6 * p + a];
        for (i64 i = m->bdof0[b]; i < m->bdof0[b] + m->bdofn[b]; i++) {
            double x = dv[i];
            for (int a = 0; a < 3; a++) {
                Vb[6 * b + a] += sw[a * m->nv + i] * x;
                Vb[6 * b + 3 + a] += sv[a * m->nv + i] * x;
            }
        }
    }
    double o0 = xpos[3 * root], o1 = xpos[3 * root + 1], h = m->h;
    for (i64 n = 0; n < m->ngb; n++) {
        i64 b = m->gbody[n];
        const double *xm = xmat + 9 * b;
        double wx = Vb[6 * b], wy = Vb[6 * b + 1];
        double A = Vb[6 * b + 5] + wx * (xpos[3 * b + 1] - o1) - wy * (xpos[3 * b] - o0);
        double u0 = wx * xm[3] - wy * xm[0], u1 = wx * xm[4] - wy * xm[1], u2 = wx * xm[5] - wy * xm[2];
        if (W->near[b]) {
            for (i64 t = m->gofs[n]; t < m->gofs[n + 1]; t++) {
                i64 k = m->gpt[t];
                if (c[k] >= act) {
                    const double *l = pl + 3 * (S + k);
                    double vz = A + u0 * l[0] + u1 * l[1] + u2 * l[2];
                    if (c[k] + vz < -tol) return 0;
                }
            }
        } else {
            double un = sqrt(u0 * u0 + u1 * u1 + u2 * u2);
            if (xpos[3 * b + 2] - h - m->reach[n] + A - un * m->lmax[n] >= -tol) continue;
            for (i64 t = m->gofs[n]; t < m->gofs[n + 1]; t++) {
                i64 k = m->gpt[t];
                const double *l = pl + 3 * (S + k);
                double ck = xpos[3 * b + 2] + xm[6] * l[0] + xm[7] * l[1] + xm[8] * l[2] - h - m->grad[k];
                double vz = A + u0 * l[0] + u1 * l[1] + u2 * l[2];
                if (ck + vz < -tol) return 0;
            }
        }
    }
    return 1;
}

/* (cost, L1 violation) at q; leaves kinematics, target points and near-floor c of q in W */
static void merit(const Model *m, Work *W, const double *q, const double *q_ref, const double *targets,
                  const Prm *P, double act, double *cost_out, double *viol_out) {
    fk(m, W, q, 1);
    points(m, W, W->pos, m->S);
    double viol = ground_near(m, W, act, act < 0.0);
    W->fk_pts = 1;
    double *s = W->s, *e = W->e;
    qdiff(q_ref, q, s, m->nq);
    double cost = 0.0;
    for (i64 p = 0; p < m->S; p++) {
        double acc = 0.0;
        for (int a = 0; a < 3; a++) {
            double r = W->pos[3 * p + a] - targets[3 * p + a];
            acc += r * r;
        }
        cost += m->wpt[p] * acc;
    }
    for (i64 i = 0; i < m->nj; i++) {
        double ei = q[7 + i] - m->q_nom[7 + i];
        e[i] = ei;
        cost += P->wreg[i] * ei * ei;
    }
    for (i64 i = 0; i < m->nv; i++) cost += P->ws[i] * s[i] * s[i];
    *cost_out = 0.5 * cost;
    *viol_out = viol;
}

/* ------------------------------------------------------------------ tree-sparse L'DL */
static inline void axpy(double *RESTRICT y, const double *RESTRICT x, double a, i64 lo, i64 hi) {
    for (i64 j = lo; j < hi; j++) y[j] -= a * x[j];
}

/* H = L'DL along the tree (mj_factorM), dense rows: F[k, j] for j an ancestor of k */
static int ltdl(const Model *m, const double *Hc, const double *Hd, double *F, double *Dg) {
    i64 n = m->nv;
    for (i64 k = 0; k < n; k++) {
        Dg[k] = Hd[k];
        for (i64 r = 0; r < m->narng[k]; r++) {
            i64 lo = ARNG(m, k, r, 0), hi = ARNG(m, k, r, 1);
            memcpy(F + n * k + lo, Hc + n * k + lo, sizeof(double) * (hi - lo + 1));
        }
    }
    for (i64 k = n - 1; k >= 0; k--) {
        double dk = Dg[k];
        if (!(dk > 1e-300)) return 0;
        double *Fk = F + n * k;
        i64 nr = m->narng[k];
        for (i64 r = 0; r < nr; r++) {
            i64 lo = ARNG(m, k, r, 0);
            for (i64 i = ARNG(m, k, r, 1); i >= lo; i--) {
                double fki = Fk[i], tmp = fki / dk;
                double *Fi = F + n * i;
                Dg[i] -= tmp * fki;
                axpy(Fi, Fk, tmp, lo, i);                         /* rest of this run */
                for (i64 r2 = r + 1; r2 < nr; r2++)               /* and the runs further up */
                    axpy(Fi, Fk, tmp, ARNG(m, k, r2, 0), ARNG(m, k, r2, 1) + 1);
                Fk[i] = tmp;
            }
        }
    }
    return 1;
}

/* x[anc(k)] -= F[k, anc(k)] x[k] */
static inline void bwd_one(const Model *m, const double *F, double *x, i64 k) {
    double xk = x[k];
    if (xk == 0.0) return;
    const double *Fk = F + m->nv * k;
    for (i64 r = 0; r < m->narng[k]; r++) axpy(x, Fk, xk, ARNG(m, k, r, 0), ARNG(m, k, r, 1) + 1);
}

/* x <- L'^-1 x for x supported on lst (descending, closed under ancestors) */
static inline void ltdl_bwd(const Model *m, const double *F, double *x, const i64 *lst, i64 nl) {
    for (i64 u = 0; u < nl; u++) bwd_one(m, F, x, lst[u]);
}

static inline double dot_r(const double *RESTRICT a, const double *RESTRICT b, i64 lo, i64 hi) {
    double s0 = 0.0, s1 = 0.0;
    i64 j = lo;
    for (; j + 1 < hi; j += 2) { s0 += a[j] * b[j]; s1 += a[j + 1] * b[j + 1]; }
    if (j < hi) s0 += a[j] * b[j];
    return s0 + s1;
}

/* x <- L^-1 x */
static inline void lfwd(const Model *m, const double *F, double *x) {
    i64 n = m->nv;
    for (i64 k = 0; k < n; k++) {
        const double *Fk = F + n * k;
        double s = 0.0;
        for (i64 r = 0; r < m->narng[k]; r++) s += dot_r(Fk, x, ARNG(m, k, r, 0), ARNG(m, k, r, 1) + 1);
        x[k] -= s;
    }
}

static inline double dot(const double *a, const double *b, i64 n) {
    double s0 = 0.0, s1 = 0.0;
    i64 t = 0;
    for (; t + 1 < n; t += 2) {
        s0 += a[t] * b[t];
        s1 += a[t + 1] * b[t + 1];
    }
    if (t < n) s0 += a[t] * b[t];
    return s0 + s1;
}

static int chol(const double *Mm, i64 n, i64 ld, double *L) {
    for (i64 j = 0; j < n; j++) {
        double s = Mm[ld * j + j];
        for (i64 k = 0; k < j; k++) s -= L[ld * j + k] * L[ld * j + k];
        if (s <= 1e-300) return 0;
        double ljj = sqrt(s);
        L[ld * j + j] = ljj;
        for (i64 i = j + 1; i < n; i++) {
            double t = Mm[ld * i + j];
            for (i64 k = 0; k < j; k++) t -= L[ld * i + k] * L[ld * j + k];
            L[ld * i + j] = t / ljj;
        }
    }
    return 1;
}

static void chol_solve(const double *L, i64 n, i64 ld, const double *b, double *out) {
    for (i64 i = 0; i < n; i++) {
        double s = b[i];
        for (i64 k = 0; k < i; k++) s -= L[ld * i + k] * out[k];
        out[i] = s / L[ld * i + i];
    }
    for (i64 i = n - 1; i >= 0; i--) {
        double s = out[i];
        for (i64 k = i + 1; k < n; k++) s -= L[ld * k + i] * out[k];
        out[i] = s / L[ld * i + i];
    }
}

typedef struct {
    const double *C, *d;
    i64 m;           /* general rows */
    const i64 *rb;   /* body of each general row (its non-zeros are on that body's dof chain) */
    const i64 *bidx;
    i64 k;           /* bounded variables */
    const double *lb, *ub;
} Rows;

static inline double row_dot(const Model *m, const Rows *R, i64 i, const double *x) {
    if (i < R->m) {
        i64 b = R->rb[i];
        const double *Ci = R->C + m->nv * i;
        double s = 0.0;
        for (i64 r = 0; r < m->nbrng[b]; r++) s += dot_r(Ci, x, BRNG(m, b, r, 0), BRNG(m, b, r, 1) + 1);
        return s;
    }
    if (i < R->m + R->k) return x[R->bidx[i - R->m]];
    return -x[R->bidx[i - R->m - R->k]];
}

static inline double row_rhs(const Rows *R, i64 i) {
    if (i < R->m) return R->d[i];
    if (i < R->m + R->k) return R->lb[i - R->m];
    return -R->ub[i - R->m - R->k];
}

/* most violated row of [C ; I_b ; -I_b] x >= [d ; lb ; -ub] (first of equals, as a single scan) */
static inline i64 most_violated(const Model *m, const Rows *R, const double *x, double *sp_out) {
    double sp = 0.0;
    i64 p = -1, mm = R->m, k = R->k;
    for (i64 i = 0; i < mm; i++) {
        double v = row_dot(m, R, i, x) - R->d[i];
        if (v < sp) { sp = v; p = i; }
    }
    for (i64 j = 0; j < k; j++) {
        double v = x[R->bidx[j]] - R->lb[j];
        if (v < sp) { sp = v; p = mm + j; }
    }
    for (i64 j = 0; j < k; j++) {
        double v = -x[R->bidx[j]] - -R->ub[j];
        if (v < sp) { sp = v; p = mm + k + j; }
    }
    *sp_out = sp;
    return p;
}

/* out = D^-1/2 L^-T a_i */
static void wrow(const Model *m, const Work *W, const Rows *R, i64 i, double *out) {
    i64 n = m->nv, b, j;
    if (i < R->m) {
        memcpy(out, R->C + n * i, sizeof(double) * n);
        b = R->rb[i];
    } else {
        memset(out, 0, sizeof(double) * n);
        if (i < R->m + R->k) { j = R->bidx[i - R->m]; out[j] = 1.0; }
        else { j = R->bidx[i - R->m - R->k]; out[j] = -1.0; }
        b = m->dbody[j];
    }
    ltdl_bwd(m, W->F, out, m->bchain + m->Lc * b, m->bchain_len[b]);
    for (i64 r = 0; r < n; r++) out[r] *= W->isd[r];
}

static inline void x_of(const Model *m, Work *W) {
    for (i64 i = 0; i < m->nv; i++) W->x[i] = W->isd[i] * W->xi[i];
    lfwd(m, W->F, W->x);
}

/* Goldfarb-Idnani dual active set in the scaled space (see _fused.gi_qp). x in W->x, active
 * set in W->act. Returns status; *iters_out, *na_out. */
static int gi_qp(const Model *m, Work *W, const Rows *R, const i64 *active0, i64 n0, double tol, i64 max_iter,
                 i64 *iters_out, i64 *na_out) {
    i64 n = m->nv, ld = n + 1;
    double *x = W->x, *wg = W->wg, *xi = W->xi, *u = W->u, *Wa = W->Wa, *M = W->M, *Lm = W->Lm,
           *rhs = W->rhs, *r = W->r, *zeta = W->zeta, *Wp = W->Wp;
    i64 *act = W->act;
    memset(x, 0, sizeof(double) * n);
    *iters_out = 0; *na_out = 0;
    if (!ltdl(m, W->Hc, W->Hd, W->F, W->Dg)) return 3;
    for (i64 i = 0; i < n; i++) { W->isd[i] = 1.0 / sqrt(W->Dg[i]); wg[i] = W->g[i]; }
    for (i64 k = n - 1; k >= 0; k--) bwd_one(m, W->F, wg, k);
    for (i64 i = 0; i < n; i++) { wg[i] *= W->isd[i]; xi[i] = -wg[i]; }
    i64 na = 0, it = 0, mt = R->m + 2 * R->k;
    for (i64 wi = 0; wi < n0; wi++) {
        i64 w = active0[wi];
        if (w < mt && na < n && isfinite(row_rhs(R, w))) act[na++] = w;
    }
    for (i64 a = 0; a < na; a++) wrow(m, W, R, act[a], Wa + n * a);
    while (na > 0) {
        for (i64 a = 0; a < na; a++) {
            for (i64 b = 0; b <= a; b++) M[ld * a + b] = M[ld * b + a] = dot(Wa + n * a, Wa + n * b, n);
            rhs[a] = row_rhs(R, act[a]) + dot(Wa + n * a, wg, n);
        }
        it++;
        if (!chol(M, na, ld, Lm)) { na = 0; break; }
        chol_solve(Lm, na, ld, rhs, r);
        i64 keep = 0;
        for (i64 a = 0; a < na; a++) {
            if (r[a] >= 0.0) {
                if (keep != a) {
                    act[keep] = act[a];
                    memcpy(Wa + n * keep, Wa + n * a, sizeof(double) * n);
                }
                keep++;
            }
        }
        if (keep == na) {
            for (i64 i = 0; i < n; i++) {
                double s = -wg[i];
                for (i64 a = 0; a < na; a++) s += Wa[n * a + i] * r[a];
                xi[i] = s;
            }
            for (i64 a = 0; a < na; a++) u[a] = r[a];
            break;
        }
        na = keep;
    }
    for (;;) {
        x_of(m, W);
        double sp;
        i64 p = most_violated(m, R, x, &sp);
        *iters_out = it; *na_out = na;
        if (p < 0 || sp >= -tol) return 0;
        for (i64 a = 0; a < na; a++)
            if (act[a] == p) return 0;
        if (it >= max_iter) return 2;
        wrow(m, W, R, p, Wp);
        double u_p = 0.0;
        for (;;) {
            it++;
            if (na > 0) {
                for (i64 a = 0; a < na; a++) {
                    for (i64 b = 0; b <= a; b++) M[ld * a + b] = M[ld * b + a] = dot(Wa + n * a, Wa + n * b, n);
                    rhs[a] = dot(Wa + n * a, Wp, n);
                }
                if (!chol(M, na, ld, Lm)) { x_of(m, W); *iters_out = it; *na_out = na; return 2; }
                chol_solve(Lm, na, ld, rhs, r);
                for (i64 i = 0; i < n; i++) {
                    double s = Wp[i];
                    for (i64 a = 0; a < na; a++) s -= Wa[n * a + i] * r[a];
                    zeta[i] = s;
                }
            } else {
                memcpy(zeta, Wp, sizeof(double) * n);
            }
            double t1 = INFINITY;
            i64 kdrop = -1;
            for (i64 a = 0; a < na; a++) {
                if (r[a] > 1e-14) {
                    double tj = u[a] / r[a];
                    if (tj < t1) { t1 = tj; kdrop = a; }
                }
            }
            double nz = dot(Wp, zeta, n), t2;
            if (nz > 1e-14) t2 = (row_rhs(R, p) - dot(Wp, xi, n)) / nz;
            else t2 = INFINITY;
            double t = t1 < t2 ? t1 : t2;
            if (t == INFINITY) { x_of(m, W); *iters_out = it; *na_out = na; return 1; }
            if (t2 != INFINITY)
                for (i64 i = 0; i < n; i++) xi[i] += t * zeta[i];
            for (i64 a = 0; a < na; a++) u[a] -= t * r[a];
            u_p += t;
            if (t2 != INFINITY && t == t2) {
                act[na] = p;
                u[na] = u_p;
                memcpy(Wa + n * na, Wp, sizeof(double) * n);
                na++;
                break;
            }
            for (i64 a = kdrop; a < na - 1; a++) {
                act[a] = act[a + 1];
                u[a] = u[a + 1];
                memcpy(Wa + n * a, Wa + n * (a + 1), sizeof(double) * n);
            }
            na--;
        }
    }
}

/* ------------------------------------------------------------------ one frame */
static void solve_frame(const Model *m, Work *W, const Prm *P, const Opt *O, const double *q0,
                        const double *q_ref, const double *q_prev, const double *targets, double *qout,
                        i64 *stats) {
    i64 nq = m->nq, nv = m->nv, G = m->G;
    double *q = W->q, *qt = W->qt, *c = W->c, *x = W->x;
    double act_lim = O->act_lim;
    int allb = act_lim < 0.0;
    memcpy(q, q0, sizeof(double) * nq);
    for (i64 itr = 0; itr < O->max_iter; itr++) {
        stats[0]++;
        int same = W->fk_dofs && W->fk_pts && memcmp(q, W->fkq, sizeof(double) * nq) == 0;
        double viol = 0.0;
        if (same) {
            for (i64 nn = 0; nn < m->ngb; nn++) {
                if (!W->near[m->gbody[nn]]) continue;
                for (i64 u = m->gofs[nn]; u < m->gofs[nn + 1]; u++)
                    if (c[m->gpt[u]] < 0.0) viol -= c[m->gpt[u]];
            }
        } else {
            fk(m, W, q, 1);
            points(m, W, W->pos, m->S);
            viol = ground_near(m, W, act_lim, allb);
        }
        W->fk_pts = 0;
        step_terms(m, q, q_ref, q_prev, P->cap, W->s, W->e, W->lb, W->ub);
        double cost = linearize(m, W, targets, P, W->s, W->e);
        int ok = 0, status = 0;
        i64 iters = 0, na = 0;
        if (act_lim >= 0.0 && G > 0) {
            i64 k = 0;
            for (i64 i = 0; i < G; i++) {
                if (W->near[m->gpb[i]] && c[i] < act_lim) {
                    ground_row(m, W, i, W->Csub + nv * k);
                    W->dsub[k] = -c[i];
                    W->idx[k] = i;
                    W->rbs[k] = m->gpb[i];
                    W->local_of[i] = k;
                    k++;
                } else {
                    W->local_of[i] = -1;
                }
            }
            i64 n0 = 0;
            for (i64 ai = 0; ai < W->nact; ai++) {
                i64 a = W->actg[ai];
                if (a < G) {
                    if (W->local_of[a] >= 0) W->a0[n0++] = W->local_of[a];
                } else {
                    W->a0[n0++] = k + (a - G);
                }
            }
            Rows R = {W->Csub, W->dsub, k, W->rbs, P->bidx, P->nb, W->lb, W->ub};
            status = gi_qp(m, W, &R, W->a0, n0, O->qp_tol, 200, &iters, &na);
            stats[4] += k;
            stats[5] += na;
            if (status == 0 && pruned_ok(m, W, act_lim, x, 1e-9)) {
                for (i64 i = 0; i < na; i++) {
                    i64 a = W->act[i];
                    W->actg[i] = a < k ? W->idx[a] : G + (a - k);
                }
                W->nact = na;
                ok = 1;
            } else {
                stats[2] += iters;
            }
        }
        if (!ok) {
            stats[6]++;
            ground_near(m, W, act_lim, 1);
            for (i64 i = 0; i < G; i++) {
                ground_row(m, W, i, W->Cfull + nv * i);
                W->dneg[i] = -c[i];
            }
            for (i64 i = 0; i < W->nact; i++) W->a0[i] = W->actg[i];
            Rows R = {W->Cfull, W->dneg, G, m->gpb, P->bidx, P->nb, W->lb, W->ub};
            status = gi_qp(m, W, &R, W->a0, W->nact, O->qp_tol, 200, &iters, &na);
            for (i64 i = 0; i < na; i++) W->actg[i] = W->act[i];
            W->nact = na;
        }
        stats[2] += iters;
        if (status != 0) stats[3]++;
        double step = 0.0;
        for (i64 i = 0; i < nv; i++)
            if (fabs(x[i]) > step) step = fabs(x[i]);
        double alpha = 1.0;
        if (O->line_search) {
            double mu = O->mu_init, phi0 = cost + mu * viol;
            double gd = 0.0;
            for (i64 i = 0; i < nv; i++) gd += W->g[i] * x[i];
            double dphi = gd - mu * viol;
            if (dphi < 0.0) {
                int found = 0;
                for (i64 l = 0; l < O->max_ls; l++) {
                    qint(q, x, alpha, qt, nq);
                    double cm, vm;
                    merit(m, W, qt, q_ref, targets, P, act_lim, &cm, &vm);
                    stats[7]++;
                    if (cm + mu * vm <= phi0 + O->c1 * alpha * dphi) { found = 1; break; }
                    alpha *= O->tau;
                    stats[1]++;
                }
                if (!found) alpha = 1.0;
            }
        }
        qint(q, x, alpha, qt, nq);
        memcpy(q, qt, sizeof(double) * nq);
        if (step < O->tol) break;
    }
    memcpy(qout, q, sizeof(double) * nq);
}

static void set_ref(const Model *m, const double *q_prev, const double *q_prev2, double alpha, int has2,
                    double *tmp, double *q_ref) {
    if (!has2 || alpha == 0.0) {
        memcpy(q_ref, q_prev, sizeof(double) * m->nq);
    } else {
        qdiff(q_prev2, q_prev, tmp, m->nq);
        for (i64 i = 0; i < m->nv; i++) tmp[i] = alpha * tmp[i];
        qint(q_prev, tmp, 1.0, q_ref, m->nq);
    }
}

static void init_yaw(const Model *m, const double *targets, double *out) {
    i64 S = m->S;
    const double *p0 = m->p0, *w = m->wpt, *qn = m->q_nom;
    double sw = 0, pa0 = 0, pa1 = 0, pa2 = 0, ta0 = 0, ta1 = 0;
    for (i64 i = 0; i < S; i++) {
        sw += w[i];
        pa0 += w[i] * p0[3 * i]; pa1 += w[i] * p0[3 * i + 1]; pa2 += w[i] * p0[3 * i + 2];
        ta0 += w[i] * targets[3 * i]; ta1 += w[i] * targets[3 * i + 1];
    }
    pa0 /= sw; pa1 /= sw; pa2 /= sw; ta0 /= sw; ta1 /= sw;
    double sxx = 0, sxy = 0;
    for (i64 i = 0; i < S; i++) {
        double a0 = p0[3 * i] - pa0, a1 = p0[3 * i + 1] - pa1;
        double b0 = targets[3 * i] - ta0, b1 = targets[3 * i + 1] - ta1;
        sxx += w[i] * (a0 * b0 + a1 * b1);
        sxy += w[i] * (a0 * b1 - a1 * b0);
    }
    double yaw = atan2(sxy, sxx);
    memcpy(out, qn, sizeof(double) * m->nq);
    double zw = cos(yaw / 2), zz = sin(yaw / 2);
    double a0 = qn[3], a1 = qn[4], a2 = qn[5], a3 = qn[6];
    out[3] = zw * a0 - zz * a3;
    out[4] = zw * a1 - zz * a2;
    out[5] = zw * a2 + zz * a1;
    out[6] = zw * a3 + zz * a0;
    double cy = cos(yaw), sy = sin(yaw), v0 = pa0 - qn[0], v1 = pa1 - qn[1];
    out[0] = ta0 - (cy * v0 - sy * v1);
    out[1] = ta1 - (sy * v0 + cy * v1);
    (void)pa2;
}

/* Retargeter.run: warm start (or q_init), boosted first frames, then one opt solve per frame */
void frdv_run_clip(const Model *m, Work *W, const Prm *P, const Opt *opt, const Opt *opt_boost,
                   const Opt *opt_warm, i64 warmup_iters, double warmup_tol, i64 boost_frames, double alpha,
                   const double *targets, i64 T, const double *q_init, i64 do_warm, double *out, i64 *stats) {
    i64 nq = m->nq, nv = m->nv, tsz = 3 * m->S;
    double *qa = W->qa, *qb = W->qb, *qref = W->qref, *tmp = W->tmp;
    if (T == 0) return;
    if (do_warm) {
        init_yaw(m, targets, qa);
        for (i64 it = 0; it < warmup_iters; it++) {
            solve_frame(m, W, P, opt_warm, qa, qa, qa, targets, qb, stats);
            qdiff(qa, qb, tmp, nq);
            double mx = 0.0;
            for (i64 i = 0; i < nv; i++)
                if (fabs(tmp[i]) > mx) mx = fabs(tmp[i]);
            memcpy(qa, qb, sizeof(double) * nq);
            if (mx < warmup_tol) break;
        }
    } else {
        memcpy(qa, q_init, sizeof(double) * nq);
    }
    for (i64 t = 0; t < T; t++) {
        const Opt *o = t < boost_frames ? opt_boost : opt;
        set_ref(m, qa, t >= 2 ? out + nq * (t - 2) : qa, alpha, t >= 2, tmp, qref);
        solve_frame(m, W, P, o, qa, qref, qa, targets + tsz * t, out + nq * t, stats);
        memcpy(qa, out + nq * t, sizeof(double) * nq);
    }
}

/* ------------------------------------------------------------------ despike (smooth.despike) */
static void joint_steps(const double *q, i64 T, i64 nq, double *out) {
    i64 nv = nq - 1;
    for (i64 t = 0; t + 1 < T; t++) {
        const double *a = q + nq * t, *b = q + nq * (t + 1);
        double *o = out + nv * t;
        for (int i = 0; i < 3; i++) o[i] = b[i] - a[i];
        for (i64 i = 7; i < nq; i++) o[i - 1] = b[i] - a[i];
        double na = sqrt(a[3] * a[3] + a[4] * a[4] + a[5] * a[5] + a[6] * a[6]);
        double nb = sqrt(b[3] * b[3] + b[4] * b[4] + b[5] * b[5] + b[6] * b[6]);
        double w0 = a[3] / na, x0 = -a[4] / na, y0 = -a[5] / na, z0 = -a[6] / na;
        double w1 = b[3] / nb, x1 = b[4] / nb, y1 = b[5] / nb, z1 = b[6] / nb;
        double w = w0 * w1 - (x0 * x1 + y0 * y1 + z0 * z1);
        double vx = w0 * x1 + w1 * x0 + (y0 * z1 - z0 * y1);
        double vy = w0 * y1 + w1 * y0 + (z0 * x1 - x0 * z1);
        double vz = w0 * z1 + w1 * z0 + (x0 * y1 - y0 * x1);
        if (w < 0) { w = -w; vx = -vx; vy = -vy; vz = -vz; }
        double sa = sqrt(vx * vx + vy * vy + vz * vz);
        double ang = 2.0 * atan2(sa, w);
        double sc = sa > 1e-15 ? ang / dmax(sa, 1e-300) : 2.0;
        o[3] = vx * sc; o[4] = vy * sc; o[5] = vz * sc;
    }
}

static i64 find_spikes(const double *steps, i64 T, i64 nv, double threshold, double ratio, i64 k, i64 *flags,
                       double *worst_out) {
    for (i64 t = 0; t < T; t++) flags[t] = 0;
    *worst_out = 0.0;
    if (T < 3) return 0;
    double nbuf[64];
    double worst = 0.0;
    i64 cnt = 0;
    for (i64 t = 0; t < T; t++) {
        for (i64 dd = 0; dd < nv; dd++) {
            double v = fabs(steps[nv * t + dd]);
            if (v > threshold) {
                i64 mm = 0;
                for (i64 o = -k; o <= k; o++) {
                    if (o == 0) continue;
                    i64 tt = t + o;
                    tt = tt < 0 ? 0 : (tt > T - 1 ? T - 1 : tt);
                    double x = fabs(steps[nv * tt + dd]);
                    i64 j = mm++;     /* insertion sort */
                    while (j > 0 && nbuf[j - 1] > x) { nbuf[j] = nbuf[j - 1]; j--; }
                    nbuf[j] = x;
                }
                double med = 0.5 * (nbuf[k - 1] + nbuf[k]);
                double base = dmax(med, 0.02);
                if (v > ratio * base) {
                    if (!flags[t]) { flags[t] = 1; cnt++; }
                    if (v > worst) worst = v;
                }
            }
        }
    }
    *worst_out = worst;
    return cnt;
}

static void window_score(const Model *m, Work *W, const double *q, i64 T, i64 a, i64 b, double *st,
                         double *acc_out, double *viol_out) {
    i64 nq = m->nq, nv = m->nv;
    i64 lo = a - 2 > 0 ? a - 2 : 0, hi = b + 2 < T ? b + 2 : T, n = hi - lo;
    double acc = 0.0;
    if (n - 1 > 1) {
        joint_steps(q + nq * lo, n, nq, st);
        for (i64 t = 0; t < n - 2; t++)
            for (i64 i = 6; i < nv; i++) {
                double v = fabs(st[nv * (t + 1) + i] - st[nv * t + i]);
                if (v > acc) acc = v;
            }
    }
    double viol = 0.0;
    for (i64 t = a; t < b; t++) {
        fk(m, W, q + nq * t, 0);
        double mn = ground_min(m, W);
        viol = dmax(viol, -mn);
    }
    *acc_out = acc;
    *viol_out = dmax(viol, 0.0);
}

void frdv_despike(const Model *m, Work *W, const Prm *prms, const Opt *opt, const double *targets, i64 T,
                  double *q, double threshold, double ratio, i64 half_window, double sigma, i64 passes,
                  i64 *irep, double *frep, i64 *wins) {
    i64 nq = m->nq, nv = m->nv, tsz = 3 * m->S;
    i64 Ts = T - 1 > 1 ? T - 1 : 1;
    double *steps = malloc(sizeof(double) * nv * Ts);
    i64 *flags = calloc(Ts, sizeof(i64));
    double *st = malloc(sizeof(double) * nv * (T + 64));
    double *qtmp = malloc(sizeof(double) * nq);
    double *anchor = malloc(sizeof(double) * nq);
    double *vv = malloc(sizeof(double) * nv);
    i64 stats[8] = {0};
    double worst;
    if (T >= 2) joint_steps(q, T, nq, steps);
    i64 cnt = find_spikes(steps, T - 1, nv, threshold, ratio, 7, flags, &worst);
    irep[0] = cnt; frep[0] = worst;
    i64 save_nact = W->nact;
    i64 save_act[256];
    for (i64 i = 0; i < save_nact && i < 256; i++) save_act[i] = W->actg[i];
    for (i64 kp = 0; kp < passes; kp++) {
        if (cnt == 0) break;
        const Prm *P = prms + kp;
        W->nact = 0;
        i64 hw = half_window + 4 * kp;
        double sg = sigma * (1 + kp);
        i64 pad = (i64)ceil(3 * sg);
        double *wts = malloc(sizeof(double) * (2 * pad + 1)), wsum = 0.0;
        for (i64 i = 0; i < 2 * pad + 1; i++) {
            double kk = (double)(i - pad);
            wts[i] = exp(-0.5 * (kk / sg) * (kk / sg));
            wsum += wts[i];
        }
        for (i64 i = 0; i < 2 * pad + 1; i++) wts[i] /= wsum;
        i64 *wa = malloc(sizeof(i64) * cnt), *wb = malloc(sizeof(i64) * cnt), nw = 0;
        for (i64 t = 0; t < T - 1; t++) {
            if (!flags[t]) continue;
            i64 a = t - hw > 1 ? t - hw : 1, b = t + hw + 1 < T - 1 ? t + hw + 1 : T - 1;
            if (nw > 0 && a <= wb[nw - 1] + 2) { if (b > wb[nw - 1]) wb[nw - 1] = b; }
            else { wa[nw] = a; wb[nw] = b; nw++; }
        }
        for (i64 w = 0; w < nw; w++) {
            i64 a = wa[w], b = wb[w], n = b - a;
            memcpy(anchor, q + nq * (a - 1), sizeof(double) * nq);
            i64 lo = a - pad > 0 ? a - pad : 0, hi = b + pad < T ? b + pad : T, L = hi - lo;
            double *V = malloc(sizeof(double) * nv * (L + 2 * pad));
            for (i64 t = 0; t < L; t++) qdiff(anchor, q + nq * (lo + t), V + nv * (pad + t), nq);
            for (i64 t = 0; t < pad; t++) {
                memcpy(V + nv * t, V + nv * pad, sizeof(double) * nv);
                memcpy(V + nv * (pad + L + t), V + nv * (pad + L - 1), sizeof(double) * nv);
            }
            double *qs = malloc(sizeof(double) * nq * n), *old = malloc(sizeof(double) * nq * n);
            for (i64 i = 0; i < n; i++) {
                i64 t = a + i, tl = t - lo + pad;
                double tp = sin(M_PI * (double)(i + 1) / (double)(n + 1));
                tp = tp * tp;
                for (i64 dd = 0; dd < nv; dd++) {
                    double acc = 0.0;
                    for (i64 j = 0; j < 2 * pad + 1; j++) acc += V[nv * (tl - pad + j) + dd] * wts[2 * pad - j];
                    vv[dd] = V[nv * tl + dd] + tp * (acc - V[nv * tl + dd]);
                }
                qint(anchor, vv, 1.0, qs + nq * i, nq);
            }
            memcpy(old, q + nq * a, sizeof(double) * nq * n);
            double s0a, s0v, s1a, s1v;
            window_score(m, W, q, T, a, b, st, &s0a, &s0v);
            for (i64 i = 0; i < n; i++) {
                solve_frame(m, W, P, opt, qs + nq * i, qs + nq * i, qs + nq * i, targets + tsz * (a + i), qtmp,
                            stats);
                memcpy(q + nq * (a + i), qtmp, sizeof(double) * nq);
            }
            window_score(m, W, q, T, a, b, st, &s1a, &s1v);
            if (s1a < s0a && s1v <= s0v + 1e-3) {
                wins[2 * irep[2]] = a; wins[2 * irep[2] + 1] = b;
                irep[2]++;
            } else {
                memcpy(q + nq * a, old, sizeof(double) * nq * n);
                irep[3]++;
            }
            free(V); free(qs); free(old);
        }
        free(wts); free(wa); free(wb);
        joint_steps(q, T, nq, steps);
        cnt = find_spikes(steps, T - 1, nv, threshold, ratio, 7, flags, &worst);
    }
    if (T >= 2) joint_steps(q, T, nq, steps);
    cnt = find_spikes(steps, T - 1, nv, threshold, ratio, 7, flags, &worst);
    irep[1] = cnt; frep[1] = worst;
    W->nact = save_nact;
    for (i64 i = 0; i < save_nact && i < 256; i++) W->actg[i] = save_act[i];
    W->fk_dofs = 0;
    free(steps); free(flags); free(st); free(qtmp); free(anchor); free(vv);
}

/* ------------------------------------------------------------------ workspace */
Work *frdv_work_new(const Model *m) {
    i64 nb = m->nbody, nv = m->nv, nq = m->nq, nj = m->nj, G = m->G > 0 ? m->G : 1, P = m->S + m->G;
    /* generous upper bounds of what is carved below */
    i64 nd = 40 * nb + 16 * nv + 4 * (nv + 2) * (nv + 2) + 2 * nv * nv + 2 * G * nv + 4 * G + 3 * P + 8 * nq + 4 * nj + 64;
    i64 ni = 2 * nv + nb + 4 * G + 2 * nj + 2 * (nv + 1) + 64;
    Work *W = calloc(1, sizeof(Work));
    double *d = calloc(nd + 16, sizeof(double));
    i64 *ii = calloc(ni + 16, sizeof(i64));
    W->block = d;
#define TAKE(ptr, n) do { ptr = d; d += (n); } while (0)
#define TAKEI(ptr, n) do { ptr = ii; ii += (n); } while (0)
    TAKE(W->xpos, 3 * nb); TAKE(W->xquat, 4 * nb); TAKE(W->xmat, 9 * nb); TAKE(W->dax, 3 * nv);
    TAKE(W->dpt, 3 * nv); TAKE(W->sw, 3 * nv); TAKE(W->sv, 3 * nv); TAKE(W->pos, 3 * P); TAKE(W->fkq, nq);
    TAKE(W->Hc, nv * nv); TAKE(W->Hd, nv); TAKE(W->g, nv); TAKE(W->c, G); TAKE(W->bK, 10 * nb);
    TAKE(W->bF, 6 * nb); TAKE(W->Vb, 6 * nb); TAKE(W->s, nv); TAKE(W->e, nj); TAKE(W->lb, nj); TAKE(W->ub, nj);
    TAKE(W->Csub, G * nv); TAKE(W->Cfull, G * nv); TAKE(W->dsub, G); TAKE(W->dneg, G);
    TAKE(W->x, nv); TAKE(W->F, nv * nv); TAKE(W->Dg, nv); TAKE(W->wg, nv); TAKE(W->xi, nv); TAKE(W->isd, nv);
    TAKE(W->u, nv + 1); TAKE(W->Wa, (nv + 1) * nv); TAKE(W->M, (nv + 1) * (nv + 1)); TAKE(W->Lm, (nv + 1) * (nv + 1));
    TAKE(W->rhs, nv + 1); TAKE(W->r, nv + 1); TAKE(W->zeta, nv); TAKE(W->Wp, nv); TAKE(W->tmp, nv);
    TAKE(W->q, nq); TAKE(W->qt, nq); TAKE(W->qa, nq); TAKE(W->qb, nq); TAKE(W->qref, nq); TAKE(W->cz, G);
    (void)W->cz;
    i64 *iblock = ii;
    TAKEI(W->dkind, nv); TAKEI(W->near, nb); TAKEI(W->idx, G); TAKEI(W->local_of, G); TAKEI(W->rbs, G);
    TAKEI(W->a0, G + 2 * nj + 1); TAKEI(W->act, nv + 1); TAKEI(W->actg, nv + 1);
    (void)iblock;
#undef TAKE
#undef TAKEI
    W->nact = 0;
    return W;
}

void frdv_work_free(Work *W) {
    if (!W) return;
    /* the int block pointer is dkind (first carved) */
    free(W->dkind);
    free(W->block);
    free(W);
}

i64 frdv_work_nact(const Work *W) { return W->nact; }

/* ------------------------------------------------------------------ motion helpers */
void frdv_bvh_fk_sel(const i64 *parents, const double *offsets, const i64 *chan_start, const i64 *chan_n,
                     const i64 *chan_kind, const i64 *chan_axis, const double *data, i64 T, i64 J, i64 nch,
                     const i64 *slot, i64 nsel, double *pos, double *rot) {
    double *Wr = malloc(sizeof(double) * 9 * J);
    for (i64 t = 0; t < T; t++) {
        for (i64 j = 0; j < J; j++) {
            double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, tv[3] = {offsets[3 * j], offsets[3 * j + 1], offsets[3 * j + 2]};
            for (i64 c = 0; c < chan_n[j]; c++) {
                i64 k = chan_start[j] + c;
                double v = data[nch * t + k];
                i64 ax = chan_axis[k];
                if (chan_kind[k] == 0) { tv[ax] += v; continue; }
                double ang = v * (M_PI / 180.0), co = cos(ang), si = sin(ang);
                i64 i1 = (ax + 1) % 3, i2 = (ax + 2) % 3;
                for (int a = 0; a < 3; a++) {
                    double x = R[3 * a + i1], y = R[3 * a + i2];
                    R[3 * a + i1] = x * co + y * si;
                    R[3 * a + i2] = -x * si + y * co;
                }
            }
            i64 p = parents[j];
            double *Wj = Wr + 9 * j, *pj = pos + 3 * (J * t + j);
            if (p < 0) {
                for (int a = 0; a < 3; a++) pj[a] = tv[a];
                memcpy(Wj, R, sizeof(R));
            } else {
                const double *Wp = Wr + 9 * p, *pp = pos + 3 * (J * t + p);
                for (int a = 0; a < 3; a++) {
                    pj[a] = pp[a] + Wp[3 * a] * tv[0] + Wp[3 * a + 1] * tv[1] + Wp[3 * a + 2] * tv[2];
                    for (int b = 0; b < 3; b++)
                        Wj[3 * a + b] = Wp[3 * a] * R[b] + Wp[3 * a + 1] * R[3 + b] + Wp[3 * a + 2] * R[6 + b];
                }
            }
            if (slot[j] >= 0) memcpy(rot + 9 * (nsel * t + slot[j]), Wj, sizeof(double) * 9);
        }
    }
    free(Wr);
}

void frdv_rotate_points(const double *P, i64 n, const double *R, double s, double *out) {
    for (i64 i = 0; i < n; i++) {
        double x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
        for (int r = 0; r < 3; r++) out[3 * i + r] = s * (R[3 * r] * x + R[3 * r + 1] * y + R[3 * r + 2] * z);
    }
}

void frdv_limb_plan(const double *P, i64 T, i64 J, const i64 *joint, const i64 *anchor, const i64 *kind,
                    const double *ratio, const i64 *rot_id, const double *off, i64 nplan, const double *R,
                    double *out) {
    for (i64 t = 0; t < T; t++) {
        const double *Pt = P + 3 * J * t;
        double *Ot = out + 3 * J * t;
        for (i64 n = 0; n < nplan; n++) {
            i64 j = joint[n], a = anchor[n], k = kind[n];
            if (k == 2) {
                for (int c = 0; c < 3; c++) Ot[3 * j + c] = ratio[n] * Pt[3 * j + c];
            } else if (k == 1) {
                const double *Rt = R + 9 * (T * rot_id[n] + t), *o = off + 3 * n;
                for (int c = 0; c < 3; c++)
                    Ot[3 * j + c] = Ot[3 * a + c] + (Rt[3 * c] * o[0] + Rt[3 * c + 1] * o[1] + Rt[3 * c + 2] * o[2]);
            } else {
                double r = ratio[n];
                for (int c = 0; c < 3; c++) Ot[3 * j + c] = Ot[3 * a + c] + r * (Pt[3 * j + c] - Pt[3 * a + c]);
            }
        }
    }
}

void frdv_finish_targets(const double *P, i64 T, i64 J, const double *fac, const i64 *idx, i64 n,
                         const double *shift, double dz, const double *offset, const double *extra, double *X) {
    for (i64 t = 0; t < T; t++)
        for (i64 i = 0; i < n; i++) {
            i64 j = idx[i];
            for (int c = 0; c < 3; c++) {
                double v = P[3 * (J * t + j) + c] * fac[c] - shift[c];
                if (c == 2) v -= dz;
                X[3 * (n * t + i) + c] = (v + offset[c]) + extra[c];
            }
        }
}

/* ------------------------------------------------------------------ BVH numbers */
/* Exact (correctly rounded, like strtod / np.fromstring): mantissa < 2^53 and |power of ten| <= 22,
 * so each value is one IEEE division / multiplication of two exactly representable numbers. Fast
 * math must stay off here (a reciprocal would break the rounding). Returns the count, or -1. */
#pragma float_control(precise, on, push)
i64 frdv_parse_decimals(const unsigned char *buf, i64 n, i64 start, double *out, i64 cap) {
    static const double pw[23] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12,
                                  1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
    i64 i = start, k = 0;
    while (i < n) {
        unsigned char c = buf[i];
        if (c == 32 || c == 10 || c == 13 || c == 9 || c == 11 || c == 12) { i++; continue; }
        int neg = 0;
        if (c == '-') { neg = 1; i++; } else if (c == '+') { i++; }
        uint64_t mant = 0;
        i64 nd = 0, frac = 0;
        while (i < n && buf[i] >= '0' && buf[i] <= '9') {
            if (nd < 19) mant = mant * 10 + (uint64_t)(buf[i] - '0');
            nd++; i++;
        }
        if (i < n && buf[i] == '.') {
            i++;
            while (i < n && buf[i] >= '0' && buf[i] <= '9') {
                if (nd < 19) mant = mant * 10 + (uint64_t)(buf[i] - '0');
                nd++; frac++; i++;
            }
        }
        if (nd == 0 || nd >= 19 || mant >= 9007199254740992ULL) return -1;
        i64 e = 0;
        if (i < n && (buf[i] == 'e' || buf[i] == 'E')) {
            i++;
            int eneg = 0;
            if (i < n && buf[i] == '-') { eneg = 1; i++; } else if (i < n && buf[i] == '+') { i++; }
            i64 ne = 0;
            while (i < n && buf[i] >= '0' && buf[i] <= '9') {
                e = e * 10 + (buf[i] - '0'); ne++; i++;
                if (e > 400) return -1;
            }
            if (ne == 0) return -1;
            if (eneg) e = -e;
        }
        if (i < n) {
            c = buf[i];
            if (!(c == 32 || c == 10 || c == 13 || c == 9 || c == 11 || c == 12)) return -1;
        }
        i64 p10 = e - frac;
        double v;
        if (p10 == 0) v = (double)mant;
        else if (p10 < 0 && p10 >= -22) v = (double)mant / pw[-p10];
        else if (p10 > 0 && p10 <= 22) v = (double)mant * pw[p10];
        else return -1;
        if (k >= cap) return -1;
        out[k++] = neg ? -v : v;
    }
    return k;
}
#pragma float_control(pop)

/* BVH forward kinematics + rigid markers: pos (T, J + nm, 3); marker i = pos[parent] + R_parent off_i */
void frdv_bvh_fk_markers(const i64 *parents, const double *offsets, const i64 *chan_start, const i64 *chan_n,
                         const i64 *chan_kind, const i64 *chan_axis, const double *data, i64 T, i64 J, i64 nch,
                         const i64 *mk_parent, const double *mk_off, i64 nm, double *pos) {
    double *Wr = malloc(sizeof(double) * 9 * J);
    i64 JJ = J + nm;
    for (i64 t = 0; t < T; t++) {
        double *pt = pos + 3 * JJ * t;
        for (i64 j = 0; j < J; j++) {
            double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, tv[3] = {offsets[3 * j], offsets[3 * j + 1], offsets[3 * j + 2]};
            for (i64 c = 0; c < chan_n[j]; c++) {
                i64 k = chan_start[j] + c;
                double v = data[nch * t + k];
                i64 ax = chan_axis[k];
                if (chan_kind[k] == 0) { tv[ax] += v; continue; }
                double ang = v * (M_PI / 180.0), co = cos(ang), si = sin(ang);
                i64 i1 = (ax + 1) % 3, i2 = (ax + 2) % 3;
                for (int a = 0; a < 3; a++) {
                    double x = R[3 * a + i1], y = R[3 * a + i2];
                    R[3 * a + i1] = x * co + y * si;
                    R[3 * a + i2] = -x * si + y * co;
                }
            }
            i64 p = parents[j];
            double *Wj = Wr + 9 * j, *pj = pt + 3 * j;
            if (p < 0) {
                for (int a = 0; a < 3; a++) pj[a] = tv[a];
                memcpy(Wj, R, sizeof(R));
            } else {
                const double *Wp = Wr + 9 * p, *pp = pt + 3 * p;
                for (int a = 0; a < 3; a++) {
                    pj[a] = pp[a] + Wp[3 * a] * tv[0] + Wp[3 * a + 1] * tv[1] + Wp[3 * a + 2] * tv[2];
                    for (int b = 0; b < 3; b++)
                        Wj[3 * a + b] = Wp[3 * a] * R[b] + Wp[3 * a + 1] * R[3 + b] + Wp[3 * a + 2] * R[6 + b];
                }
            }
        }
        for (i64 i = 0; i < nm; i++) {
            i64 p = mk_parent[i];
            const double *Wp = Wr + 9 * p, *pp = pt + 3 * p, *o = mk_off + 3 * i;
            double *pm = pt + 3 * (J + i);
            for (int a = 0; a < 3; a++) pm[a] = pp[a] + (Wp[3 * a] * o[0] + Wp[3 * a + 1] * o[1] + Wp[3 * a + 2] * o[2]);
        }
    }
    free(Wr);
}

/* k-th smallest of v[0..n) (v is reordered) */
static double select_kth(double *v, i64 n, i64 k) {
    i64 lo = 0, hi = n - 1;
    while (hi > lo) {
        double a = v[lo], b = v[(lo + hi) / 2], c = v[hi];
        double piv = a < b ? (b < c ? b : (a < c ? c : a)) : (a < c ? a : (b < c ? c : b));
        i64 i = lo, j = hi;
        while (i <= j) {
            while (v[i] < piv) i++;
            while (v[j] > piv) j--;
            if (i <= j) { double t = v[i]; v[i] = v[j]; v[j] = t; i++; j--; }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else return v[k];
    }
    return v[k];
}

/* np.median(np.linalg.norm(P[:, a] - P[:, b], axis=-1)) for each (a, b) pair */
#pragma float_control(precise, on, push)
#pragma clang fp contract(off)
void frdv_bone_medians(const double *P, i64 T, i64 J, const i64 *ia, const i64 *ib, i64 nb, double *out) {
    double *v = malloc(sizeof(double) * (T > 0 ? T : 1));
    for (i64 n = 0; n < nb; n++) {
        for (i64 t = 0; t < T; t++) {
            const double *pa = P + 3 * (J * t + ia[n]), *pb = P + 3 * (J * t + ib[n]);
            double x = pa[0] - pb[0], y = pa[1] - pb[1], z = pa[2] - pb[2];
            double s = x * x;
            s = s + y * y;
            s = s + z * z;
            v[t] = sqrt(s);
        }
        if (T % 2) out[n] = select_kth(v, T, T / 2);
        else {
            double hi = select_kth(v, T, T / 2), lo = v[0];
            for (i64 t = 1; t < T / 2; t++) if (v[t] > lo) lo = v[t];   /* max of the lower half */
            out[n] = (lo + hi) / 2.0;
        }
    }
    free(v);
}
#pragma float_control(pop)

/* limb_rescale's torso rotation: frame F_t (columns: lat x up, lat, up) of the performer, times Fr'.
 * lat = P[la] - P[lb]; up = mid(P[u1], P[u2]) - mid(P[v1], P[v2]), or +z if u1 < 0. */
void frdv_torso_frames(const double *P, i64 T, i64 J, i64 la, i64 lb, i64 u1, i64 u2, i64 v1, i64 v2,
                       const double *Fr, double *out) {
    for (i64 t = 0; t < T; t++) {
        const double *pt = P + 3 * J * t;
        double lat[3], up[3], F[9];
        for (int a = 0; a < 3; a++) {
            lat[a] = pt[3 * la + a] - pt[3 * lb + a];
            if (u1 >= 0)
                up[a] = 0.5 * (pt[3 * u1 + a] + pt[3 * u2 + a]) - 0.5 * (pt[3 * v1 + a] + pt[3 * v2 + a]);
            else
                up[a] = a == 2 ? 1.0 : 0.0;
        }
        double nl = dmax(sqrt(lat[0] * lat[0] + lat[1] * lat[1] + lat[2] * lat[2]), 1e-9);
        for (int a = 0; a < 3; a++) lat[a] /= nl;
        double d = up[0] * lat[0] + up[1] * lat[1] + up[2] * lat[2];
        for (int a = 0; a < 3; a++) up[a] -= d * lat[a];
        double nu = dmax(sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]), 1e-9);
        for (int a = 0; a < 3; a++) up[a] /= nu;
        double cr[3] = {lat[1] * up[2] - lat[2] * up[1], lat[2] * up[0] - lat[0] * up[2], lat[0] * up[1] - lat[1] * up[0]};
        for (int a = 0; a < 3; a++) { F[3 * a] = cr[a]; F[3 * a + 1] = lat[a]; F[3 * a + 2] = up[a]; }
        double *o = out + 9 * t;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                o[3 * i + j] = F[3 * i] * Fr[3 * j] + F[3 * i + 1] * Fr[3 * j + 1] + F[3 * i + 2] * Fr[3 * j + 2];
    }
}
