import os
import sys

# Prevent OpenMP runtime conflict between libomp and libiomp5
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import time
import numpy as np
import hydra
from omegaconf import DictConfig
import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as ca
import mujoco
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy import sparse
import osqp
import matplotlib.pyplot as plt
from tqdm import tqdm

from mocap_retargeting.data_loading.lafan1.loader import Lafan1Loader
from mocap_retargeting.mujoco_integration.mujoco_interface import mj_to_pin, pin_to_mj
from mocap_retargeting.mujoco_integration.mujoco_interface import get_body_geoms, get_ground_geoms
from mocap_retargeting import PROJECT_ROOT
from mocap_retargeting.kinematic_retargeting.optimization.sqp_solver import SQP
from mocap_retargeting.kinematic_retargeting.optimization.sqp_solver import SQPSolverCfg, OSQPSolverCfg

np.set_printoptions(precision=4, suppress=True)
def casadi_pin_aba_func(model, contact_link_names):
    q = ca.SX.sym("q", model.nq)
    v = ca.SX.sym("v", model.nv)
    tau = ca.SX.sym("tau", model.nv - 6)
    fext_world = ca.SX.sym(
        "fext", 3 * len(contact_link_names)
    )  # external linear forces on contact links
    pin_data = model.createData()
    f_ext = [cpin.Force.Zero() for _ in range(model.njoints)]

    cpin.forwardKinematics(model, pin_data, q)  # Update frame placements

    for i, link_name in enumerate(contact_link_names):
        link_id = model.getFrameId(link_name)
        parent_jt_id = model.frames[link_id].parentJoint

        cpin.updateFramePlacement(model, pin_data, link_id)  # Update frame placement

        force = fext_world[3 * i : 3 * i + 3]
        fext_wal = ca.vertcat(force, ca.SX.zeros(3))
        fext_wal = cpin.Force(fext_wal)
        # Create world aligned frame R=I and p = pos of link
        wal_frame = cpin.SE3(ca.SX.eye(3), pin_data.oMf[link_id].translation)
        # Transform force from world aligned local frame to world frame
        fext_world_i = wal_frame.act(fext_wal)
        # Transform force to parent joint frame
        f_parent_joint = pin_data.oMi[parent_jt_id].actInv(fext_world_i)
        f_ext[parent_jt_id] += cpin.Force(f_parent_joint)

    tau_full = ca.vertcat(ca.SX.zeros(6), tau)
    a = cpin.aba(model, pin_data, q, v, tau_full, f_ext)

    return ca.Function("FD", [q, v, tau, fext_world], [a])


def add_contact_constraint_flat(opti, ugrf_k, csk, mu):
    num_contacts = csk.shape[0]

    for foot in range(num_contacts):
        grf = ugrf_k[3 * foot : 3 * foot + 3]
        f_x, f_y, f_z = grf[0], grf[1], grf[2]
        csk_foot = csk[foot]

        opti.subject_to(csk_foot * f_z >= 0)
        opti.subject_to(csk_foot * f_x <= mu * csk_foot * f_z)
        opti.subject_to(csk_foot * f_y <= mu * csk_foot * f_z)
        opti.subject_to(csk_foot * f_x >= -mu * csk_foot * f_z)
        opti.subject_to(csk_foot * f_y >= -mu * csk_foot * f_z)
        opti.subject_to((1 - csk_foot) * grf == 0)


class DynFilter:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.setup_models()
        self.foot_links = list(cfg.get("foot_links", []))

    def setup_models(self):
        print(f"[INFO] Loading robot model from: {self.cfg.mujoco_dst_xml_path}")
        if not os.path.exists(self.cfg.mujoco_dst_xml_path):
            raise FileNotFoundError(
                f"[ERROR] Model file not found: {self.cfg.mujoco_dst_xml_path}"
            )

        # Pinocchio
        self.pin_model = pin.buildModelFromMJCF(self.cfg.mujoco_dst_xml_path)
        self.pin_data = self.pin_model.createData()

        # CasADi Pinocchio
        self.cpin_model = cpin.Model(self.pin_model)
        self.cpin_data = self.cpin_model.createData()

        # MuJoCo
        self.mj_model = mujoco.MjModel.from_xml_path(self.cfg.mujoco_dst_xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.nq = self.pin_model.nq
        self.nv = self.pin_model.nv
        self.nu = self.nq - 7  # Number of actuated joints (assuming free flyer root)

        print(f"[INFO] Model loaded. nq={self.nq}, nv={self.nv}")

    def load_motion(self, motion_path=None):
        if motion_path is None:
            motion_path = self.cfg.get("motion_path", None)

        if not motion_path or not os.path.exists(motion_path):
            # Fallback logic could go here, but for now raise error
            raise FileNotFoundError(f"Motion file not found: {motion_path}")

        self.motion_path = motion_path
        print(f"[INFO] Loading motion data from {motion_path}")
        data = np.load(motion_path, allow_pickle=True).item()

        q_traj_full = data["q"]
        self.fps = data.get("fps", 30)
        self.dt = 1.0 / self.fps
        self.mocap_data = data.get("mocap", {})
        self.skeleton_data = data.get("skeleton", [])

        # Slicing
        start_idx = self.cfg.get("start_index", 0)
        end_idx = self.cfg.get("end_index", -1)
        if end_idx == -1:
            end_idx = len(q_traj_full)

        self.q_traj_raw = q_traj_full[start_idx:end_idx]
        self.T = len(self.q_traj_raw)

        if self.q_traj_raw.shape[1] != self.nq:
            raise ValueError(
                f"Motion dimension {self.q_traj_raw.shape[1]} does not match model nq {self.nq}"
            )

        print(
            f"[INFO] Motion loaded. Shape: {self.q_traj_raw.shape}, FPS: {self.fps}, T: {self.T}"
        )

    def filter_trajectory(self):
        print(f"[INFO] Filtering trajectory...")
        window_length = self.cfg.get("filter_window", 5)
        polyorder = self.cfg.get("filter_poly", 3)

        # Adjust window size
        if window_length > self.T:
            window_length = self.T | 1
            if window_length < polyorder + 2:
                window_length = polyorder + 2
                if window_length % 2 == 0:
                    window_length += 1

        self.q_filtered = savgol_filter(
            self.q_traj_raw, window_length, polyorder, axis=0
        )

        has_root = self.nq >= 7
        if has_root:
            # Normalize quaternions (indices 3:7)
            quats = self.q_filtered[:, 3:7]
            norms = np.linalg.norm(quats, axis=1, keepdims=True)
            self.q_filtered[:, 3:7] = quats / (norms + 1e-8)

    def compute_kinematics(self):
        print(f"[INFO] Computing derivatives (v, a)...")
        # Initialize storage
        self.v_filtered = np.zeros((self.T, self.nv))
        self.a_filtered = np.zeros((self.T, self.nv))
        self.v_raw = np.zeros((self.T, self.nv))
        self.a_raw = np.zeros((self.T, self.nv))

        # Compute v (filtered & raw)
        for t in range(self.T - 1):
            self.v_filtered[t] = (
                pin.difference(
                    self.pin_model, self.q_filtered[t], self.q_filtered[t + 1]
                )
                / self.dt
            )
            self.v_raw[t] = (
                pin.difference(
                    self.pin_model, self.q_traj_raw[t], self.q_traj_raw[t + 1]
                )
                / self.dt
            )

        # Pad last v
        self.v_filtered[-1] = self.v_filtered[-2]
        self.v_raw[-1] = self.v_raw[-2]

        # Compute a
        for t in range(self.T - 1):
            self.a_filtered[t] = (self.v_filtered[t + 1] - self.v_filtered[t]) / self.dt
            self.a_raw[t] = (self.v_raw[t + 1] - self.v_raw[t]) / self.dt

        # Pad last a
        self.a_filtered[-1] = self.a_filtered[-2]
        self.a_raw[-1] = self.a_raw[-2]

    def detect_contacts(self):
        print("[INFO] Detecting contacts in MuJoCo...")
        foot_geoms = {}
        for body_name in self.foot_links:
            bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid == -1:
                print(f"[WARNING] Foot link '{body_name}' not found.")
                continue
            foot_geoms[body_name] = get_body_geoms(self.mj_model, bid)

        all_foot_geoms = set()
        for gids in foot_geoms.values():
            all_foot_geoms.update(gids)

        ground_geoms = set(get_ground_geoms(self.mj_model))
        if not ground_geoms and self.mj_model.ngeom > 0:
            ground_geoms.add(0)

        if not foot_geoms:
            raise ValueError("[ERROR] No foot geoms found.")

        self.num_feet = len(self.foot_links)
        self.contacts = np.zeros((self.T, self.num_feet), dtype=bool)

        for t in tqdm(range(self.T), desc="Contact Detection"):
            q_pin = self.q_filtered[t]  # Use filtered q for contact detection
            q_mj = pin_to_mj(q_pin)

            self.mj_data.qpos = q_mj
            self.mj_data.qvel[:] = 0
            mujoco.mj_forward(self.mj_model, self.mj_data)

            for i in range(self.mj_data.ncon):
                con = self.mj_data.contact[i]
                g1, g2 = con.geom1, con.geom2

                is_g1_foot = g1 in all_foot_geoms
                is_g2_foot = g2 in all_foot_geoms
                is_g1_ground = g1 in ground_geoms
                is_g2_ground = g2 in ground_geoms

                detected_foot_geom = -1
                if is_g1_foot and is_g2_ground:
                    detected_foot_geom = g1
                elif is_g2_foot and is_g1_ground:
                    detected_foot_geom = g2

                if detected_foot_geom != -1:
                    for idx, fname in enumerate(self.foot_links):
                        if detected_foot_geom in foot_geoms[fname]:
                            self.contacts[t, idx] = True
                            break

        # Debounce
        kernel_size = self.cfg.get("contact_filter_size", 10)
        if kernel_size > 1:
            print(f"[INFO] Applying median filter to contacts (kernel={kernel_size})")
            self.contacts = median_filter(self.contacts, size=(kernel_size, 1))




    def compute_contact_jacobians(self, q):
        Jac_list = []
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        for fname in self.foot_links:
            frame_id = self.pin_model.getFrameId(fname)
            J_frame = pin.computeFrameJacobian(
                self.pin_model,
                self.pin_data,
                q,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            Jac_list.append(J_frame[3:6, :])  # Use only linear part
        Jac = np.vstack(Jac_list)
        return Jac

    def solve_dynamics_qp(self):
        """
        Solve inverse dynamics using direct QP formulation with OSQP.

        Decision variables: x = [tau (nu), grf (3*num_feet)]

        Dynamics equation: M * a + h = S^T * tau + Jc^T * grf
        where h = C(q,v) + g(q) (from RNEA with zero acceleration)

        We minimize weighted residual of dynamics + regularization on tau and grf.
        """
        print(f"[INFO] Solving Inverse Dynamics (QP with OSQP)...")

        # Get weights from config
        tau_weight = self.cfg.get("optimizer", {}).get("tau_weight", 10)
        grf_weight = self.cfg.get("optimizer", {}).get("grf_weight", 10)
        mu = self.cfg.get("optimizer", {}).get("friction_coef", 0.7)

        fd_func = casadi_pin_aba_func(self.cpin_model, self.foot_links)


        # Initialize storage
        self.grf_history = np.zeros((self.T, 3 * self.num_feet))
        self.tau_history = np.zeros((self.T, self.nu))
        self.acc_error_history = np.zeros((self.T, self.nv))

        # Decision variable dimensions
        n_tau = self.nu
        n_grf = 3 * self.num_feet
        n_x = n_tau + n_grf

        # Build friction cone constraints (linearized)
        # For each foot: |fx| <= mu*fz, |fy| <= mu*fz, fz >= 0
        # Linearized as: fx - mu*fz <= 0, -fx - mu*fz <= 0, etc.
        n_friction_constraints = 5 * self.num_feet  # 4 friction pyramid + 1 normal force >= 0

        def build_friction_constraints():
            """Build G matrix and h vector for friction cone: Gx <= h"""
            G_list = []
            for i in range(self.num_feet):
                # Indices for this foot's GRF in decision variable
                idx_fx = n_tau + i * 3
                idx_fy = n_tau + i * 3 + 1
                idx_fz = n_tau + i * 3 + 2

                # fx - mu*fz <= 0
                row1 = np.zeros(n_x)
                row1[idx_fx] = 1.0
                row1[idx_fz] = -mu
                G_list.append(row1)

                # -fx - mu*fz <= 0
                row2 = np.zeros(n_x)
                row2[idx_fx] = -1.0
                row2[idx_fz] = -mu
                G_list.append(row2)

                # fy - mu*fz <= 0
                row3 = np.zeros(n_x)
                row3[idx_fy] = 1.0
                row3[idx_fz] = -mu
                G_list.append(row3)

                # -fy - mu*fz <= 0
                row4 = np.zeros(n_x)
                row4[idx_fy] = -1.0
                row4[idx_fz] = -mu
                G_list.append(row4)

                # -fz <= 0 (fz >= 0)
                row5 = np.zeros(n_x)
                row5[idx_fz] = -1.0
                G_list.append(row5)

            G = np.vstack(G_list)
            h = np.zeros(n_friction_constraints)
            return G, h

        G_friction, h_friction = build_friction_constraints()

        # OSQP solver instance (will be reused with warm starting)
        prob = None
        last_P_nnz = -1
        last_A_rows = -1

        for t in tqdm(range(self.T), desc="Dynamics QP Solve"):
            try:
                q = self.q_filtered[t]
                v = self.v_filtered[t]
                a = self.a_filtered[t]
                contacts = self.contacts[t]

                # Compute dynamics quantities
                # RNEA: tau_rnea = M*a + C + g
                tau_rnea = pin.rnea(self.pin_model, self.pin_data, q, v, a)

                # Mass matrix for weighting
                pin.crba(self.pin_model, self.pin_data, q)
                M = self.pin_data.M.copy()

                # Contact Jacobian (3*num_feet x nv) - linear velocity part
                Jc = self.compute_contact_jacobians(q)

                # Selection matrix S: tau_full = S^T * tau_actuated
                # For floating base: S = [0_{nu x 6} | I_{nu x nu}]
                # S^T maps actuated torques to full torque vector
                S_T = np.zeros((self.nv, n_tau))
                S_T[6:, :] = np.eye(n_tau)

                # Dynamics: tau_rnea = S^T * tau + Jc^T * grf
                # Build A matrix: [S^T | Jc^T] so A @ x = tau_rnea
                A_dyn = np.hstack([S_T, Jc.T])
                b_dyn = tau_rnea

                # QP cost: min 0.5 * x^T P x + q^T x
                # We want to minimize ||A_dyn @ x - b_dyn||^2_M + tau_weight*||tau||^2 + grf_weight*||grf||^2
                # = 0.5 * x^T (A^T M A + W_reg) x - b^T M A x + const

                W_tau = tau_weight * np.eye(n_tau)
                W_grf = grf_weight * np.eye(n_grf)
                W_reg = sparse.block_diag([W_tau, W_grf]).toarray()

                P = A_dyn.T @ M @ A_dyn + W_reg
                q_vec = -A_dyn.T @ M @ b_dyn

                # Apply contact mask: if foot not in contact, force GRF to zero
                # Add equality constraints for non-contact feet
                A_eq_list = []
                b_eq_list = []
                for i in range(self.num_feet):
                    if not contacts[i]:
                        # grf[i*3:i*3+3] = 0
                        for j in range(3):
                            row = np.zeros(n_x)
                            row[n_tau + i * 3 + j] = 1.0
                            A_eq_list.append(row)
                            b_eq_list.append(0.0)

                if A_eq_list:
                    A_eq = np.vstack(A_eq_list)
                    b_eq = np.array(b_eq_list)
                else:
                    A_eq = np.zeros((0, n_x))
                    b_eq = np.zeros(0)

                # OSQP format: min 0.5 x^T P x + q^T x  s.t.  l <= Ax <= u
                # Combine inequality (G @ x <= h) and equality (A_eq @ x = b_eq)
                A_combined = np.vstack([G_friction, A_eq])
                l_combined = np.concatenate([
                    -np.inf * np.ones(n_friction_constraints),
                    b_eq
                ])
                u_combined = np.concatenate([
                    h_friction,
                    b_eq
                ])


                P_sparse = sparse.csc_matrix(P)
                A_sparse = sparse.csc_matrix(A_combined)

                # Check if sparsity pattern changed (requires re-setup)
                need_setup = True or  (
                    prob is None
                    or P_sparse.nnz != last_P_nnz
                    or A_sparse.shape[0] != last_A_rows
                )

                if need_setup:
                    prob = osqp.OSQP()
                    prob.setup(
                        P_sparse, q_vec, A_sparse, l_combined, u_combined,
                        verbose=False,
                        warm_start=True,
                        eps_abs=1e-3,
                        eps_rel=1e-3,
                        max_iter=100,
                    )
                    last_P_nnz = P_sparse.nnz
                    last_A_rows = A_sparse.shape[0]
                else:
                    prob.update(
                        Px=P_sparse.data,
                        q=q_vec,
                        Ax=A_sparse.data,
                        l=l_combined,
                        u=u_combined,
                    )

                result = prob.solve()

                if result.info.status in ["solved", "solved inaccurate", "maximum iterations reached"]:
                    x_sol = result.x
                    self.tau_history[t] = x_sol[:n_tau]
                    self.grf_history[t] = x_sol[n_tau:]
                    acc_pred = fd_func(q, v, self.tau_history[t], self.grf_history[t])
                    self.acc_error_history[t] = a - acc_pred.full().flatten()

                else:
                    print(f"[WARNING] OSQP failed at frame {t}: {result.info.status}")
                    if t > 0:
                        self.tau_history[t] = self.tau_history[t - 1]
                        self.grf_history[t] = self.grf_history[t - 1]

            except KeyboardInterrupt:
                print("\n[INFO] Optimization interrupted by user. Stopping early...")
                break
            except Exception as e:
                print(f"[WARNING] QP solve failed at frame {t}: {e}")
                if t > 0:
                    self.tau_history[t] = self.tau_history[t - 1]
                    self.grf_history[t] = self.grf_history[t - 1]

    def solve_dynamics(self):
        print(f"[INFO] Solving Inverse Dynamics (CasADi)...")
        opti = ca.Opti()

        taus = opti.variable(self.nu)
        grfs = opti.variable(3 * self.num_feet)

        qddot_param = opti.parameter(self.nv)
        q_param = opti.parameter(self.nq)
        v_param = opti.parameter(self.nv)
        mass_matrix_param = opti.parameter(self.nv, self.nv)
        mu_param = opti.parameter(1)
        cs_param = opti.parameter(self.num_feet)
        tau_weights = opti.parameter(1)
        grf_weights = opti.parameter(1)

        fd_func = casadi_pin_aba_func(self.cpin_model, self.foot_links)

        acc_pred = fd_func(q_param, v_param, taus, grfs)
        acc_error = qddot_param - acc_pred

        residuals = ca.vertcat(acc_error, grfs, taus)

        W_grf = grf_weights * ca.MX.eye(3 * self.num_feet)
        W_tau = tau_weights * ca.MX.eye(self.nu)

        weights_block = ca.diagcat(mass_matrix_param, W_grf, W_tau)

        cost = ca.mtimes(ca.mtimes(residuals.T, weights_block), residuals)

        opti.minimize(cost)
        add_contact_constraint_flat(opti, grfs, cs_param, mu_param)

        c = opti.f
        x = opti.x
        p = opti.p
        g = opti.g
        lb = opti.lbg
        ub = opti.ubg

        solver_type = "ipopt" #self.cfg.get("optimizer").get("solver", "ipopt").lower()
        print(f"[INFO] Using solver: {solver_type}")

        sqp = None
        if solver_type == "sqp":
            cost_jac = ca.gradient(c, x)
            residual_jac = ca.jacobian(residuals, x)
            constraint_jac = ca.jacobian(g, x)
            cost_hess = ca.triu(
                residual_jac.T @ weights_block @ residual_jac
            )  # gauss-newton hessian

            qp_func_dyn = ca.Function(
                "qp_func_dyn",
                [x, p],
                [
                    g,
                    cost_jac,
                    cost_hess,
                    constraint_jac,
                    lb,
                    ub,
                ],
                ["x", "p"],
                [
                    "constraints",
                    "cost_jac",
                    "cost_hess",
                    "constraint_jac",
                    "lb",
                    "ub",
                ],
            ).expand()

            ls_func_dyn = ca.Function(
                "ls_func_dyn",
                [x, p],
                [c, g, lb, ub],
                ["x", "p"],
                [
                    "cost",
                    "constraints",
                    "lb",
                    "ub",
                ],
            ).expand()

            if self.cfg.verbose:
                print(f"[INFO] qp_func_dyn num_instr: {qp_func_dyn.n_instructions()}")
                print(f"[INFO] ls_func_dyn num_instr: {ls_func_dyn.n_instructions()}")

            if self.cfg.get("jit_compile", False):
                robot_name = self.cfg.get("robot", "unknown_robot")
                jit_base_dir = os.path.abspath(f"log/casadi_jit/{robot_name}_dyn")
                os.makedirs(jit_base_dir, exist_ok=True)

                print(
                    f"[INFO] JIT Compiling dynfilter functions for {robot_name} in {jit_base_dir}..."
                )

                def compile_and_load(func, name, recompile):
                    cwd = os.getcwd()
                    try:
                        os.chdir(jit_base_dir)

                        src_name = f"{name}.c"
                        if recompile:
                            func.generate(src_name)
                        if sys.platform == "darwin":  # macOS
                            lib_name = f"{name}.dylib"
                            compile_cmd = (
                                f"gcc -fPIC -dynamiclib -O3 {src_name} -o {lib_name}"
                            )
                        else:  # Linux
                            lib_name = f"{name}.so"
                            compile_cmd = f"gcc -fPIC -shared -O3 {src_name} -o {lib_name}"

                        if os.path.exists(lib_name) and not recompile:
                            print(f"[INFO] Using cached {name} in {jit_base_dir}")
                            return ca.external(name, os.path.join(jit_base_dir, lib_name))

                        ret_code = os.system(compile_cmd)
                        if ret_code != 0:
                            raise RuntimeError(f"Compilation failed for {name}")

                        return ca.external(name, os.path.join(jit_base_dir, lib_name))

                    finally:
                        os.chdir(cwd)

                qp_func_dyn = compile_and_load(
                    qp_func_dyn, "qp_func_dyn", recompile=self.cfg.get("jit_recompile", True)
                )
                ls_func_dyn = compile_and_load(
                    ls_func_dyn, "ls_func_dyn", recompile=self.cfg.get("jit_recompile", True)
                )

            sqp = SQP(qp_func_dyn, ls_func_dyn)

            # Initialize SQP config
            sqp_conf = self.cfg.get("sqp", {})

            qp_conf = sqp_conf.get("qp_solver", {})
            if qp_conf.get("type", "osqp") != "osqp":
                raise ValueError("Only the OSQP backend is supported.")
            osqp_conf = qp_conf.get("osqp", {})
            qp_solver_cfg = OSQPSolverCfg(
                verbose=osqp_conf.get("verbose", False),
                alpha=osqp_conf.get("alpha", 1.6),
                eps_abs=osqp_conf.get("eps_abs", 1e-5),
                eps_rel=osqp_conf.get("eps_rel", 1e-5),
                eps_prim_inf=osqp_conf.get("eps_prim_inf", 1e-4),
                eps_dual_inf=osqp_conf.get("eps_dual_inf", 1e-4),
                rho=osqp_conf.get("rho", 0.1),
                sigma=osqp_conf.get("sigma", 1e-6),
                scaling=osqp_conf.get("scaling", 10),
                max_iter=osqp_conf.get("max_iter", 4000),
                time_limit=osqp_conf.get("time_limit", 0),
                adaptive_rho_interval=osqp_conf.get("adaptive_rho_interval", 0),
                check_termination=osqp_conf.get("check_termination", 25),
                adaptive_rho=osqp_conf.get("adaptive_rho", True),
                warm_starting=osqp_conf.get("warm_starting", True),
            )

            self.sqp_cfg = SQPSolverCfg(
                max_sqp_iter=sqp_conf.get("max_sqp_iter", 20),
                tol=sqp_conf.get("tol", 1e-4),
                mu_init=sqp_conf.get("mu_init", 10.0),
                mu_multiplier=sqp_conf.get("mu_multiplier", 1.1),
                tau=sqp_conf.get("tau", 0.5),
                c1=sqp_conf.get("c1", 1e-4),
                max_ls_iter=sqp_conf.get("max_ls_iter", 10),
                use_line_search=sqp_conf.get("use_line_search", True),
                qp_solver=qp_solver_cfg,
            )

        else:
            p_opts = {"expand": True, "print_time": False}
            s_opts = {"max_iter": 100, "print_level": 0, "tol": 1e-3}
            opti.solver("ipopt", p_opts, s_opts)

        opti.set_value(mu_param, 0.7)
        opti.set_value(tau_weights, 0.0001)
        opti.set_value(grf_weights, 0.0001)

        self.grf_history = np.zeros((self.T, 3 * self.num_feet))
        self.tau_history = np.zeros((self.T, self.nu))
        self.acc_error_history = np.zeros((self.T, self.nv))

        for t in tqdm(range(self.T), desc="Dynamics Solve"):
            try:
                pin.crba(self.pin_model, self.pin_data, self.q_filtered[t])

                opti.set_value(q_param, self.q_filtered[t])
                opti.set_value(v_param, self.v_filtered[t])
                opti.set_value(qddot_param, self.a_filtered[t])
                opti.set_value(mass_matrix_param, self.pin_data.M)
                opti.set_value(cs_param, self.contacts[t])

                M_flat = self.pin_data.M.reshape(-1, 1)

                sqp_params = ca.vertcat(
                    self.a_filtered[t].reshape(-1, 1),
                    self.q_filtered[t].reshape(-1, 1),
                    self.v_filtered[t].reshape(-1, 1),
                    M_flat,
                    0.7,  # mu
                    self.contacts[t].astype(float).reshape(-1, 1),  # contacts
                    0.0001,  # tau_weight
                    0.0001,  # grf_weight
                )
                sqp_x0 = np.concatenate(
                    (
                        self.tau_history[t - 1].reshape(-1),
                        self.grf_history[t - 1].reshape(-1),
                    )
                )

                # Warm start
                if t > 0:
                    opti.set_initial(taus, self.tau_history[t - 1])
                    opti.set_initial(grfs, self.grf_history[t - 1])

                if solver_type == "sqp":
                    M_flat = self.pin_data.M.reshape(-1, 1)

                    sqp_params = ca.vertcat(
                        self.a_filtered[t].reshape(-1, 1),
                        self.q_filtered[t].reshape(-1, 1),
                        self.v_filtered[t].reshape(-1, 1),
                        M_flat,
                        0.7,  # mu
                        self.contacts[t].astype(float).reshape(-1, 1),  # contacts
                        0.0001,  # tau_weight
                        0.0001,  # grf_weight
                    )

                    if t > 0:
                        sqp_x0 = np.concatenate(
                            (
                                self.tau_history[t - 1].reshape(-1),
                                self.grf_history[t - 1].reshape(-1),
                            )
                        )
                    else:
                        sqp_x0 = np.zeros(self.nu + 3 * self.num_feet)

                    try:
                        sqp_sol, stats = sqp.solve(sqp_x0, sqp_params, cfg=self.sqp_cfg)
                        self.tau_history[t] = sqp_sol[: self.nu]

                        self.grf_history[t] = sqp_sol[self.nu :]
                    except Exception as e:
                        print(f"[WARNING] SQP Solver failed at frame {t}: {e}")
                        if t > 0:
                            self.tau_history[t] = self.tau_history[t - 1]
                            self.grf_history[t] = self.grf_history[t - 1]
                else:
                    try:
                        sol = opti.solve()
                        self.tau_history[t] = sol.value(taus)
                        self.grf_history[t] = sol.value(grfs)
                        self.acc_error_history[t] = sol.value(acc_error)
                    except Exception as e:
                        print(f"[WARNING] IPOPT Solver failed at frame {t}: {e}")
                        if t > 0:
                            self.tau_history[t] = self.tau_history[t - 1]
                            self.grf_history[t] = self.grf_history[t - 1]
            except KeyboardInterrupt:
                print("\n[INFO] Optimization interrupted by user. Stopping early...")
                break

    def plot_results(self):
        print("[INFO] Plotting results...")
        plot_dir = os.path.join(
            os.path.dirname(self.motion_path),
            "plots_" + os.path.basename(self.motion_path).replace(".npy", ""),
        )
        os.makedirs(plot_dir, exist_ok=True)

        time_vals = np.arange(self.T) * self.dt

        for i in range(self.nv):
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            title_suffix = ""
            is_angular_root = i >= 3 and i < 6 and self.nq >= 7

            if is_angular_root:
                axs[0].text(0.5, 0.5, "Orientation (Quaternion)", ha="center")
            else:
                q_idx = i
                if self.nq >= 7 and i >= 6:
                    q_idx = i + 1
                axs[0].plot(
                    time_vals,
                    self.q_traj_raw[:, q_idx],
                    label="Raw",
                    color="gray",
                    alpha=0.5,
                )
                axs[0].plot(
                    time_vals,
                    self.q_filtered[:, q_idx],
                    label="Filtered",
                    color="red",
                    alpha=0.8,
                )
                axs[0].set_ylabel("Position")
                axs[0].legend()

            axs[1].plot(
                time_vals, self.v_raw[:, i], label="Raw", color="gray", alpha=0.5
            )
            axs[1].plot(
                time_vals,
                self.v_filtered[:, i],
                label="Filtered",
                color="blue",
                alpha=0.8,
            )
            axs[1].set_ylabel("Velocity")

            axs[2].plot(
                time_vals, self.a_raw[:, i], label="Raw", color="gray", alpha=0.5
            )
            axs[2].plot(
                time_vals,
                self.a_filtered[:, i],
                label="Filtered",
                color="green",
                alpha=0.8,
            )
            axs[2].set_ylabel("Acceleration")
            axs[2].set_xlabel("Time (s)")

            fig.suptitle(f"DOF {i}")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"dof_{i:02d}_kinematics.png"))
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4))
        for i, fname in enumerate(self.foot_links):
            ax.plot(
                time_vals,
                self.contacts[:, i].astype(float) * 0.8 + i,
                label=fname,
                alpha=0.9,
            )
        ax.set_yticks(range(self.num_feet))
        ax.set_yticklabels(self.foot_links)
        ax.set_title("Ground Contacts")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "contacts.png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4))
        for i, fname in enumerate(self.foot_links):
            grf_z = self.grf_history[:, i * 3 + 2]
            ax.plot(time_vals, grf_z, label=f"{fname} Fz")
        ax.set_title("Vertical Ground Reaction Forces")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "grf_z.png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4))
        for i in range(min(self.nu, 10)):
            ax.plot(time_vals, self.tau_history[:, i], label=f"Tau {i}")
        ax.set_title("Joint Torques (First 10)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "torques.png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4))
        err_norm = np.linalg.norm(self.acc_error_history, axis=1)
        ax.plot(time_vals, err_norm, label="Norm(Acc Error)", color="red")
        ax.set_title("Acceleration Error Norm")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (m/s^2)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "acc_error.png"))
        plt.close(fig)

        print(f"[INFO] Plots saved to {plot_dir}")

    def save_results(self):
        output_path = self.motion_path.replace(".npy", "_processed.npy")
        save_data = {
            "q": self.q_filtered,
            "v": self.v_filtered,
            "a": self.a_filtered,
            "q_raw": self.q_traj_raw,
            "v_raw": self.v_raw,
            "a_raw": self.a_raw,
            "contacts": self.contacts,
            "tau": self.tau_history,
            "grf": self.grf_history,
            "acc_error": self.acc_error_history,
            "fps": self.fps,
            "mocap": self.mocap_data,
            "skeleton": self.skeleton_data,
            "foot_names": self.foot_links,
        }
        np.save(output_path, save_data)
        print(f"[INFO] Processed data saved to {output_path}")

    def run(self):
        self.load_motion()
        self.filter_trajectory()
        self.compute_kinematics()
        self.detect_contacts()
        self.solve_dynamics_qp()
        self.plot_results()
        self.save_results()


@hydra.main(
    version_base=None,
    config_path=os.path.join(PROJECT_ROOT, "conf"),
    config_name="config",
)
def main(cfg: DictConfig):
    try:
        df = DynFilter(cfg)
        df.run()
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
