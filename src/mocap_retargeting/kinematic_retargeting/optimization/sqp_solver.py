"""SQP and QP configuration, solver statistics, and sequential quadratic programming."""

import logging
import time
from dataclasses import dataclass, field

import casadi as ca
import numpy as np
import osqp
from omegaconf import OmegaConf
from scipy import sparse

logger = logging.getLogger(__name__)


@dataclass
class QPSolverCfg:
    verbose: bool = False


@dataclass
class OSQPSolverCfg(QPSolverCfg):
    alpha: float = 1.6
    eps_abs: float = 1e-5
    eps_rel: float = 1e-5
    eps_prim_inf: float = 1e-4
    eps_dual_inf: float = 1e-4
    rho: float = 0.1
    sigma: float = 1e-6
    scaling: int = 10
    max_iter: int = 4000
    time_limit: float = 0
    adaptive_rho_interval: int = 0
    check_termination: int = 25
    adaptive_rho: bool = True
    warm_starting: bool = True


@dataclass
class SQPSolverCfg:
    max_sqp_iter: int = 1
    tol: float = 1e-2
    mu_init: float = 10.0
    mu_multiplier: float = 1.1
    tau: float = 0.5
    c1: float = 1e-4
    max_ls_iter: int = 10
    use_line_search: bool = False

    qp_solver: OSQPSolverCfg = field(default_factory=OSQPSolverCfg)


@dataclass
class SQPStats:
    sqp_iters: int = 0
    total_ls_iters: int = 0
    final_cost: float = 0.0
    final_violation: float = 0.0
    converged: bool = False
    qp_func_time: float = 0.0
    line_search_time: float = 0.0
    qp_solve_time: float = 0.0
    qp_status: str = ""
    qp_failed: bool = False


def load_solver_config(cfg):
    cfg = cfg if OmegaConf.is_config(cfg) else OmegaConf.create(cfg)
    sqp_args = OmegaConf.to_container(
        cfg.get("sqp", OmegaConf.create({})), resolve=True
    )
    qp_conf = sqp_args.pop("qp_solver", {})

    qp_type = qp_conf.get("type", "osqp")
    if qp_type != "osqp":
        raise ValueError(f"Unsupported QP solver {qp_type!r}; only 'osqp' is supported.")
    qp_solver_cfg = OSQPSolverCfg(**qp_conf.get("osqp", {}))

    return SQPSolverCfg(qp_solver=qp_solver_cfg, **sqp_args)


class SQP:
    def __init__(self, qp_func_or_path, ls_func_or_path):
        if isinstance(qp_func_or_path, str) and isinstance(ls_func_or_path, str):
            self.qp_func = ca.Function.load(qp_func_or_path)
            self.ls_func = ca.Function.load(ls_func_or_path)
        else:
            self.qp_func = qp_func_or_path
            self.ls_func = ls_func_or_path
        self.prob = None
        self.last_P_nnz_size = None

    def solve(
        self, x0, p, cfg: SQPSolverCfg = SQPSolverCfg(), verbose=False, *, context=""
    ):
        if not isinstance(cfg.qp_solver, OSQPSolverCfg):
            raise ValueError("SQP requires an OSQPSolverCfg configuration.")
        x = x0.copy()
        stats = SQPStats()

        for sqp_iter in range(cfg.max_sqp_iter):
            stats.sqp_iters += 1
            t0 = time.perf_counter()
            res = self.qp_func(x=x, p=p)
            stats.qp_func_time += time.perf_counter() - t0
            cost_hess = np.array(res["cost_hess"])
            cost_jac = np.array(res["cost_jac"]).flatten()
            constraint_jac = np.array(res["constraint_jac"])
            constraints = np.array(res["constraints"]).flatten()
            lb = np.array(res["lb"]).flatten()
            ub = np.array(res["ub"]).flatten()

            P = sparse.csc_matrix(cost_hess)
            q = cost_jac
            A = sparse.csc_matrix(constraint_jac)
            l = lb - constraints
            u = ub - constraints

            needs_setup = self.prob is None or self.last_P_nnz_size != P.nnz
            if self.prob is None:
                self.prob = osqp.OSQP()
            if needs_setup:
                osqp_settings = {
                    "verbose": cfg.qp_solver.verbose,
                    "alpha": cfg.qp_solver.alpha,
                    "eps_abs": cfg.qp_solver.eps_abs,
                    "eps_rel": cfg.qp_solver.eps_rel,
                    "eps_prim_inf": cfg.qp_solver.eps_prim_inf,
                    "eps_dual_inf": cfg.qp_solver.eps_dual_inf,
                    "rho": cfg.qp_solver.rho,
                    "sigma": cfg.qp_solver.sigma,
                    "scaling": cfg.qp_solver.scaling,
                    "max_iter": cfg.qp_solver.max_iter,
                    "time_limit": cfg.qp_solver.time_limit,
                    "adaptive_rho_interval": cfg.qp_solver.adaptive_rho_interval,
                    "check_termination": cfg.qp_solver.check_termination,
                    "adaptive_rho": cfg.qp_solver.adaptive_rho,
                    "warm_start": cfg.qp_solver.warm_starting,
                }
                self.prob.setup(P, q, A, l, u, **osqp_settings)
            else:
                self.prob.update(Px=P.data, q=q, Ax=A.data, l=l, u=u)

            self.last_P_nnz_size = P.nnz

            t0 = time.perf_counter()
            result = self.prob.solve()
            stats.qp_solve_time += time.perf_counter() - t0
            stats.qp_status = result.info.status
            if stats.qp_status not in (
                "solved", "maximum iterations reached", "solved inaccurate"
            ):
                stats.qp_failed = True
                logger.warning(
                    "OSQP failed (%s) at SQP iteration %d%s. Returning the current pose.",
                    stats.qp_status,
                    sqp_iter,
                    f" | {context}" if context else "",
                )
                break

            dx = result.x
            y_dual = result.y

            norm_dx = np.linalg.norm(dx)
            if norm_dx < cfg.tol:
                if verbose:
                    print(
                        f"SQP converged at iter {sqp_iter}: norm(dx)={norm_dx:.6f} < {cfg.tol}"
                    )
                stats.converged = True
                break

            # --- Backtracking Line Search ---
            t0 = time.perf_counter()
            if not cfg.use_line_search:
                x = x + dx
                stats.line_search_time += time.perf_counter() - t0
                continue

            alpha = 1.0

            mu = cfg.mu_init
            if y_dual is not None:
                mu = max(mu, np.max(np.abs(y_dual)) * cfg.mu_multiplier)

            def compute_merit(x_val, p_val):
                res_ls = self.ls_func(x=x_val, p=p_val)
                cost_val = float(res_ls["cost"])
                g_val = np.array(res_ls["constraints"]).flatten()
                lb_val = np.array(res_ls["lb"]).flatten()
                ub_val = np.array(res_ls["ub"]).flatten()

                violation = np.sum(np.maximum(0, lb_val - g_val)) + np.sum(
                    np.maximum(0, g_val - ub_val)
                )
                res = {
                    "cost": cost_val,
                    "violation": violation,
                    "merit": cost_val + mu * violation,
                }
                return res

            merit_0 = compute_merit(x, p)

            # Directional derivative of merit function for Armijo condition
            grad_J_dot_dx = np.dot(cost_jac, dx)
            violation_0 = merit_0["violation"]
            D_phi = grad_J_dot_dx - mu * violation_0

            step_accepted = False
            for i in range(cfg.max_ls_iter):
                x_new = x + alpha * dx
                merit = compute_merit(x_new, p)

                # Armijo condition
                if merit["merit"] <= merit_0["merit"] + cfg.c1 * alpha * D_phi:
                    if verbose:
                        print(
                            f"SQP {sqp_iter}: LS accepted alpha={alpha}, merit={merit['merit']:.4f} < {merit_0['merit']:.4f}"
                        )
                    x = x_new
                    step_accepted = True
                    break

                alpha *= cfg.tau
                stats.total_ls_iters += 1

            if not step_accepted:
                if verbose:
                    print(f"SQP {sqp_iter}: Line search failed. Taking full step.")
                x = x + dx
            stats.line_search_time += time.perf_counter() - t0

        # Compute final stats
        res_final = self.ls_func(x=x, p=p)
        cost_val = float(res_final["cost"])
        g_val = np.array(res_final["constraints"]).flatten()
        lb_val = np.array(res_final["lb"]).flatten()
        ub_val = np.array(res_final["ub"]).flatten()
        violation = np.sum(np.maximum(0, lb_val - g_val)) + np.sum(
            np.maximum(0, g_val - ub_val)
        )
        stats.final_cost = cost_val
        stats.final_violation = violation

        if verbose:
            print(f"stats: {stats}")
        return x, stats
