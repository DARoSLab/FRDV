"""OSQP solves, failure diagnostics, and configuration validation."""

import casadi as ca
import numpy as np
import pytest
from omegaconf import OmegaConf

from mocap_retargeting.kinematic_retargeting.optimization.sqp_solver import (
    OSQPSolverCfg,
    SQP,
    SQPSolverCfg,
    load_solver_config,
)
from mocap_retargeting.kinematic_retargeting.optimization.trajectory_solver import (
    RetargetingSolver,
)


def scalar_solver(*, infeasible=False, parameter_count=1):
    x = ca.SX.sym("x", 1)
    p = ca.SX.sym("p", parameter_count)
    cost = 0.5 * (x[0] - p[0]) ** 2
    constraints = ca.vertcat(x, x)
    lb = ca.DM([1, -ca.inf] if infeasible else [-10, -ca.inf])
    ub = ca.DM([ca.inf, 0] if infeasible else [ca.inf, 10])
    qp = ca.Function(
        "qp", [x, p],
        [ca.hessian(cost, x)[0], ca.gradient(cost, x),
         ca.jacobian(constraints, x), constraints, lb, ub],
        ["x", "p"],
        ["cost_hess", "cost_jac", "constraint_jac", "constraints", "lb", "ub"],
    )
    ls = ca.Function(
        "ls", [x, p], [cost, constraints, lb, ub],
        ["x", "p"], ["cost", "constraints", "lb", "ub"],
    )
    return SQP(qp, ls)


def solver_config():
    return SQPSolverCfg(
        max_sqp_iter=5, tol=1e-8,
        qp_solver=OSQPSolverCfg(time_limit=1.0),
    )


def test_osqp_setup_and_cached_updates_solve_without_failure_logs(caplog):
    solver = scalar_solver()
    q, first = solver.solve(np.zeros(1), np.array([1.0]), cfg=solver_config())
    np.testing.assert_allclose(q, [1.0], atol=1e-5)
    cached_problem = solver.prob
    q, second = solver.solve(q, np.array([-2.0]), cfg=solver_config())
    np.testing.assert_allclose(q, [-2.0], atol=1e-5)
    assert solver.prob is cached_problem
    assert first.qp_status == second.qp_status == "solved"
    assert not first.qp_failed and not second.qp_failed
    assert not caplog.records


def test_infeasible_qp_logs_without_verbose_and_keeps_the_current_pose(caplog):
    solver = scalar_solver(infeasible=True)
    initial = np.zeros(1)
    q, stats = solver.solve(
        initial, np.array([1.0]), cfg=solver_config(),
        context="walk.npz | chunk 2 | source frame 57",
    )
    np.testing.assert_array_equal(q, initial)
    assert stats.qp_failed and stats.qp_status == "primal infeasible"
    assert not stats.converged
    assert "walk.npz | chunk 2 | source frame 57" in caplog.text
    assert "OSQP failed (primal infeasible)" in caplog.text


def test_chunk_failure_logs_include_source_frames_warmup_and_summary(caplog):
    cfg = OmegaConf.create({
        "data_type": "amass", "verbose": False, "warmup_iterations": 2,
        "optimizer": {"sqp_boost": 1, "sqp_boost_frames": 0},
    })
    mocap = np.zeros((2, 1, 3))
    retargeter = RetargetingSolver(
        cfg, scalar_solver(infeasible=True, parameter_count=4), mocap,
        np.zeros(1), {"Pelvis": "pelvis"}, solver_config(),
        context="walk.npz | chunk 2", start_index=57,
    )
    q, returned_mocap = retargeter.retarget(["Pelvis"])
    np.testing.assert_array_equal(q, np.zeros((2, 1)))
    np.testing.assert_array_equal(returned_mocap, mocap)
    assert "source frame 57 (chunk frame 0) | warmup iteration 0" in caplog.text
    assert "source frame 58 (chunk frame 1)" in caplog.text
    assert "2/2 frame solves and 2/2 warmup solves failed" in caplog.text


def test_solver_config_defaults_to_osqp_and_resolves_parent_interpolations():
    assert isinstance(load_solver_config({}).qp_solver, OSQPSolverCfg)
    cfg = OmegaConf.create({
        "limit": 250,
        "sqp": {"qp_solver": {"type": "osqp", "osqp": {"max_iter": "${limit}"}}},
    })
    assert load_solver_config(cfg).qp_solver.max_iter == 250


def test_unsupported_solver_configuration_raises_a_clear_error():
    with pytest.raises(ValueError, match="only 'osqp' is supported"):
        load_solver_config({"sqp": {"qp_solver": {"type": "unsupported"}}})
