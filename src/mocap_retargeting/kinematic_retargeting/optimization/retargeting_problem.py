"""Robot setup, symbolic kinematics, retargeting objectives/constraints, and JIT compilation."""

import os
import sys

import casadi as ca
import mujoco
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
from scipy.spatial.transform import Rotation as R

from mocap_retargeting.kinematic_retargeting.optimization.capsule_collision import (
    create_capsule_distance_function,
)
from mocap_retargeting.kinematic_retargeting.optimization.sqp_solver import (
    load_solver_config,
)


class RobotModel:
    """Own the robot models, mapped sites, limits, and nominal pose."""

    def __init__(self, cfg):
        self.cfg = cfg
        if not os.path.exists(self.cfg.mujoco_dst_xml_path):
            raise FileNotFoundError(
                f"Mujoco model file not found: {self.cfg.mujoco_dst_xml_path} Please run process --config-name <config_name> first."
            )

        self.capsules = []
        if self.cfg.get("use_capsule_constraint", True):
            mj_model = mujoco.MjModel.from_xml_path(self.cfg.mujoco_dst_xml_path)
            for i in range(mj_model.ngeom):
                if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    bid = mj_model.geom_bodyid[i]
                    body_name = mujoco.mj_id2name(
                        mj_model, mujoco.mjtObj.mjOBJ_BODY, bid
                    )

                    size = mj_model.geom_size[i]
                    radius = size[0]
                    half_len = size[1]

                    pos_local = mj_model.geom_pos[i]
                    quat_local = mj_model.geom_quat[i]

                    r = R.from_quat(
                        [quat_local[1], quat_local[2], quat_local[3], quat_local[0]]
                    )
                    mat_local = r.as_matrix()

                    p1_cap = np.array([0, 0, -half_len])
                    p2_cap = np.array([0, 0, half_len])

                    p1_body = pos_local + mat_local @ p1_cap
                    p2_body = pos_local + mat_local @ p2_cap

                    self.capsules.append(
                        {
                            "body_name": body_name,
                            "p1": p1_body,
                            "p2": p2_body,
                            "radius": radius,
                        }
                    )

            if self.cfg.verbose:
                print(f"[INFO] Loaded {len(self.capsules)} capsules from Mujoco model.")
                # print("Capsules: ", str(self.capsules))

        pinmodel = pin.buildModelFromMJCF(self.cfg.mujoco_dst_xml_path)
        self.pinmodel = pinmodel
        self.robot_name = self.cfg.get("robot", "unknown_robot")
        self.cpin_model = cpin.Model(pinmodel)
        self.cpin_data = self.cpin_model.createData()
        self.nq = pinmodel.nq
        self.nv = pinmodel.nv
        self.q_min = np.array(pinmodel.lowerPositionLimit)[7:]
        self.q_max = np.array(pinmodel.upperPositionLimit)[7:]

        if "t_pose" in pinmodel.referenceConfigurations:
            if self.cfg.verbose:
                print("[INFO] Found keyframe 't_pose' in Pinocchio model")
            self.q0_robot = np.array(pinmodel.referenceConfigurations["t_pose"])
        else:
            raise ValueError("Keyframe 't_pose' not found.")

        self.active_mapping = {}
        for item in self.cfg.mapping:
            if "bvh" in item:
                self.active_mapping[item["bvh"]] = f"mocap_{item['bvh']}"


class RobotKinematics:
    """Build symbolic FK for mapped robot links, feet, and collision geometry."""

    def __init__(self, robot, cfg):
        self.robot = robot
        self.cfg = cfg
        q_sym = ca.SX.sym("q", self.robot.nq)
        cpin.forwardKinematics(self.robot.cpin_model, self.robot.cpin_data, q_sym)
        cpin.updateFramePlacements(self.robot.cpin_model, self.robot.cpin_data)
        pos_sym_list = []
        out_names = []
        for item in self.cfg.mapping:
            if "robot" in item and "bvh" in item:
                r_name = item["robot"]
                m_name = item["bvh"]
                try:
                    if self.cfg.verbose:
                        print(f"Processing {r_name} -> {m_name}")

                    frame_id = self.robot.cpin_model.getFrameId(r_name)

                    try:
                        pos_sym = self.robot.cpin_data.oMf[frame_id].translation
                    except:
                        print(
                            f"ERROR: Could not get translation for frame '{r_name}'. Skipping."
                        )
                        quit(-1)
                    pos_sym_list.append(pos_sym)
                    out_names.append(f"mocap_{m_name}")
                except ValueError:
                    print(f"ERROR: Frame '{r_name}' not found in model. Skipping.")
                    quit(-1)

        foot_pos_sym_list = []
        for foot_link in self.cfg.foot_links:
            try:
                frame_id = self.robot.cpin_model.getFrameId(foot_link)
                pos_sym = self.robot.cpin_data.oMf[frame_id].translation
                foot_pos_sym_list.append(pos_sym)
            except ValueError:
                print(f"ERROR: Frame '{foot_link}' not found in model. Skipping.")
                quit(-1)

        foot_poses = ca.vertcat(*foot_pos_sym_list)
        self.fn_fk_sites = ca.Function(
            "fn_fk_sites", [q_sym], pos_sym_list, ["q"], out_names
        )
        self.fn_fk_feet = ca.Function("fn_fk_feet", [q_sym], [foot_poses])

        if self.cfg.get("use_capsule_constraint", True):
            cap_z_list = []
            transformed_capsules = []

            for cap in self.robot.capsules:
                try:
                    if self.robot.cpin_model.existFrame(cap["body_name"]):
                        fid = self.robot.cpin_model.getFrameId(cap["body_name"])

                        transform = self.robot.cpin_data.oMf[fid]

                        p1 = transform.translation + ca.mtimes(
                            transform.rotation, ca.DM(cap["p1"])
                        )
                        p2 = transform.translation + ca.mtimes(
                            transform.rotation, ca.DM(cap["p2"])
                        )

                        r_val = cap["radius"]

                        cap_z_list.append(p1[2] - r_val)
                        cap_z_list.append(p2[2] - r_val)

                        transformed_capsules.append(
                            {
                                "p1": p1,
                                "p2": p2,
                                "radius": r_val,
                                "body_name": cap["body_name"],
                            }
                        )
                except Exception as e:
                    pass

            if len(cap_z_list) > 0:
                capsule_z_vec = ca.vertcat(*cap_z_list)
            else:
                capsule_z_vec = ca.SX([])

            self.fn_capsule_z = ca.Function(
                "fn_capsule_z", [q_sym], [capsule_z_vec], ["q"], ["cap_z"]
            )

            if self.cfg.get("use_capsule_self_collision", False):
                collision_pairs_cfg = self.cfg.get("collision_pairs")
                allowed_pairs_set = set()

                if collision_pairs_cfg is not None:
                    for p in collision_pairs_cfg:
                        if len(p) == 2:
                            allowed_pairs_set.add(frozenset(p))

                kept_pairs = 0

                caps1_p1 = []
                caps1_p2 = []
                caps1_r = []
                caps2_p1 = []
                caps2_p2 = []
                caps2_r = []

                num_caps = len(transformed_capsules)
                for i in range(num_caps):
                    for j in range(i + 1, num_caps):
                        cap1 = transformed_capsules[i]
                        cap2 = transformed_capsules[j]

                        bn1 = cap1["body_name"]
                        bn2 = cap2["body_name"]

                        if bn1 == bn2:
                            continue

                        if allowed_pairs_set is not None:
                            if frozenset([bn1, bn2]) not in allowed_pairs_set:
                                continue

                        caps1_p1.append(cap1["p1"])
                        caps1_p2.append(cap1["p2"])
                        caps1_r.append(cap1["radius"])

                        caps2_p1.append(cap2["p1"])
                        caps2_p2.append(cap2["p2"])
                        caps2_r.append(cap2["radius"])

                        kept_pairs += 1

                if self.cfg.verbose:
                    print(f"[INFO] Self-collision: Kept {kept_pairs} pairs.")

                if kept_pairs > 0:
                    vec_c1_p1 = ca.horzcat(*caps1_p1)
                    vec_c1_p2 = ca.horzcat(*caps1_p2)
                    vec_c1_r = ca.horzcat(*[ca.DM(r) for r in caps1_r])

                    vec_c2_p1 = ca.horzcat(*caps2_p1)
                    vec_c2_p2 = ca.horzcat(*caps2_p2)
                    vec_c2_r = ca.horzcat(*[ca.DM(r) for r in caps2_r])

                    dist_func_base = create_capsule_distance_function()
                    mapped_dist_func = dist_func_base.map(kept_pairs)

                    dists_row = mapped_dist_func(
                        vec_c1_p1, vec_c1_p2, vec_c1_r, vec_c2_p1, vec_c2_p2, vec_c2_r
                    )

                    self_col_vec = dists_row.T
                else:
                    self_col_vec = ca.SX([])

                self.fn_self_collision = ca.Function(
                    "fn_self_collision", [q_sym], [self_col_vec], ["q"], ["dists"]
                )


def compile_and_load(jit_base_dir, func, name, recompile):
    cwd = os.getcwd()
    try:
        os.chdir(jit_base_dir)

        src_name = f"{name}.c"
        if recompile:
            func.generate(src_name)
        if sys.platform == "darwin":  # macOS
            lib_name = f"{name}.dylib"
            compile_cmd = f"gcc -fPIC -dynamiclib -O3 {src_name} -o {lib_name}"
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


class RetargetingProblem:
    """Compose robot assets, symbolic kinematics, and optimization functions."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.current_weights = {}
        self.current_smoothing_weight = cfg.get("optimizer", {}).get(
            "smoothing_weight", 0.1
        )
        self.robot = RobotModel(cfg)
        for bvh_link, site_name in self.robot.active_mapping.items():
            weight = cfg.optimizer.tracking_default_weight
            for item in cfg.mapping:
                if item.get("bvh") == bvh_link:
                    weight = item.get("weight", cfg.optimizer.tracking_default_weight)
                    break
            print(f"Set'{bvh_link}' weight {weight}")
            self.current_weights[site_name] = weight
        self.kinematics = RobotKinematics(self.robot, cfg)
        self.sqp_cfg = load_solver_config(cfg)

    def generate_solver_functions(self):
        self.opti = ca.Opti()

        self.q = self.opti.variable(self.robot.nq)

        self.p_targets = {}
        self.p_q_prev = self.opti.parameter(self.robot.nq)

        total_cost = 0

        residuals = {}
        fk_poses = self.kinematics.fn_fk_sites(q=self.q)
        for bvh_link, site_name in self.robot.active_mapping.items():
            p_target = self.opti.parameter(3)
            self.p_targets[site_name] = p_target

            fk_pos = fk_poses[site_name]

            weight = self.current_weights[site_name]
            err = fk_pos - p_target
            total_cost += weight * ca.mtimes(err.T, err)
            residuals["target_" + site_name] = {
                "error": err,
                "weight": weight * np.ones(3),
            }

        q_joints = self.q[7:]
        q_nominal_joints = self.robot.q0_robot[7:]
        diff = q_joints - q_nominal_joints
        total_cost += 1.0 * ca.mtimes(diff.T, diff)
        residuals["joints"] = {"error": diff, "weight": np.ones(len(q_nominal_joints))}

        root_ori_quat = self.q[3:7]

        self.opti.subject_to(root_ori_quat.T @ root_ori_quat == 1.0)
        feet_pos_height = self.kinematics.fn_fk_feet(self.q)[2::3]

        smoothing_weight = self.current_smoothing_weight
        diff_prev = self.q - self.p_q_prev
        total_cost += smoothing_weight * ca.mtimes(diff_prev.T, diff_prev)

        residuals["smoothing"] = {
            "error": diff_prev,
            "weight": smoothing_weight * np.ones(self.robot.nq),
        }

        self.opti.subject_to(
            self.opti.bounded(self.robot.q_min, self.q[7:], self.robot.q_max)
        )
        self.opti.subject_to(feet_pos_height >= 0.0)

        if self.cfg.get("use_capsule_constraint", True):
            capsule_z_vec = self.kinematics.fn_capsule_z(self.q)
            if capsule_z_vec.shape[0] > 0:
                self.opti.subject_to(capsule_z_vec >= 0.0)

            if hasattr(self.kinematics, "fn_self_collision"):
                dists_vec = self.kinematics.fn_self_collision(self.q)
                if dists_vec.shape[0] > 0:
                    min_dist = self.cfg.get("self_collision_margin", 0.02)
                    self.opti.subject_to(dists_vec >= min_dist)

        self.opti.minimize(total_cost)

        full_residual = ca.vertcat(*[err["error"] for err in residuals.values()])
        full_W = ca.diag(ca.vertcat(*[err["weight"] for err in residuals.values()]))

        cost = 0.5 * ca.mtimes(ca.mtimes(full_residual.T, full_W), full_residual)
        c = cost
        x = self.opti.x
        p = self.opti.p
        g = self.opti.g
        lb = self.opti.lbg
        ub = self.opti.ubg

        large_num = 1e20

        lb = ca.fmax(lb, -large_num)
        ub = ca.fmin(ub, large_num)

        cost_jac = ca.gradient(c, x)
        residual_jac = ca.jacobian(full_residual, x)
        constraint_jac = ca.jacobian(g, x)
        cost_hess = ca.triu(
            residual_jac.T @ full_W @ residual_jac
        )  # gauss-newton hessian
        qp_func = ca.Function(
            "qp_func",
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

        ls_func = ca.Function(
            "ls_func",
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
            print(f"[INFO] qp_func num_instr: {qp_func.n_instructions()}")
            print(f"[INFO] ls_func num_instr: {ls_func.n_instructions()}")

        if self.cfg.get("jit_compile", False):
            jit_base_dir = os.path.abspath(f"log/casadi_jit/{self.robot.robot_name}")
            os.makedirs(jit_base_dir, exist_ok=True)

            print(
                f"[INFO] JIT Compiling functions for {self.robot.robot_name} in {jit_base_dir}..."
            )

            qp_func = compile_and_load(
                jit_base_dir,
                qp_func,
                "qp_func",
                recompile=self.cfg.get("jit_recompile", True),
            )
            ls_func = compile_and_load(
                jit_base_dir,
                ls_func,
                "ls_func",
                recompile=self.cfg.get("jit_recompile", True),
            )

        return qp_func, ls_func

    def generate_stitch_solver_functions(self, T):
        """
        Generate trajectory optimization functions for stitching T frames.

        Args:
            T: Number of frames in the trajectory

        Returns:
            qp_func, ls_func: Casadi functions for QP and line search
        """
        if T < 2:
            raise ValueError(f"T must be at least 2 for stitching, got T={T}")

        opti = ca.Opti()

        # Decision variables: T frames, each with nq DOFs
        q_vars = []
        for t in range(T):
            q_vars.append(opti.variable(self.robot.nq))

        # Parameters: fixed first and last frames
        q_first = opti.parameter(self.robot.nq)
        q_last = opti.parameter(self.robot.nq)

        # Optional: original trajectory for regularization
        q_original = opti.parameter(T * self.robot.nq)

        total_cost = 0

        # 1. Smoothness cost: sum of squared velocities
        smoothness_weight = 1.0
        for t in range(T - 1):
            vel = q_vars[t + 1] - q_vars[t]
            total_cost += smoothness_weight * ca.mtimes(vel.T, vel)

        # 2. Regularization cost: stay close to original trajectory
        reg_weight = 0.1
        for t in range(T):
            q_orig_t = q_original[t * self.robot.nq : (t + 1) * self.robot.nq]
            diff = q_vars[t] - q_orig_t
            total_cost += reg_weight * ca.mtimes(diff.T, diff)

        # 3. Joint regularization: stay near nominal pose
        joint_reg_weight = 0.01
        for t in range(T):
            q_joints_t = q_vars[t][7:]
            q_nominal_joints = self.robot.q0_robot[7:]
            diff = q_joints_t - q_nominal_joints
            total_cost += joint_reg_weight * ca.mtimes(diff.T, diff)

        # Constraints for each frame
        for t in range(T):
            q_t = q_vars[t]

            # Quaternion normalization for root orientation
            root_ori_quat = q_t[3:7]
            opti.subject_to(root_ori_quat.T @ root_ori_quat == 1.0)

            # Joint limits
            opti.subject_to(opti.bounded(self.robot.q_min, q_t[7:], self.robot.q_max))

            # Foot height constraints
            feet_pos_height = self.kinematics.fn_fk_feet(q_t)[2::3]
            opti.subject_to(feet_pos_height >= 0.0)

            if hasattr(self.kinematics, "fn_self_collision"):
                dists_vec = self.kinematics.fn_self_collision(q_t)
                if dists_vec.shape[0] > 0:
                    min_dist = self.cfg.get("self_collision_margin", 0.02)
                    opti.subject_to(dists_vec >= min_dist)

        # Equality constraints: fix first and last frames
        opti.subject_to(q_vars[0] == q_first)
        opti.subject_to(q_vars[-1] == q_last)

        opti.minimize(total_cost)

        # Build residual vector for Gauss-Newton Hessian approximation
        residuals = []
        weights = []

        # Smoothness residuals
        for t in range(T - 1):
            vel = q_vars[t + 1] - q_vars[t]
            residuals.append(vel)
            weights.extend([smoothness_weight] * self.robot.nq)

        # Regularization residuals
        for t in range(T):
            q_orig_t = q_original[t * self.robot.nq : (t + 1) * self.robot.nq]
            diff = q_vars[t] - q_orig_t
            residuals.append(diff)
            weights.extend([reg_weight] * self.robot.nq)

        # Joint regularization residuals
        for t in range(T):
            q_joints_t = q_vars[t][7:]
            q_nominal_joints = self.robot.q0_robot[7:]
            diff = q_joints_t - q_nominal_joints
            residuals.append(diff)
            weights.extend([joint_reg_weight] * (self.robot.nq - 7))

        full_residual = ca.vertcat(*residuals)
        full_W = ca.diag(ca.DM(weights))

        # Extract problem components
        x = opti.x
        p = opti.p
        g = opti.g
        lb = opti.lbg
        ub = opti.ubg
        c = 0.5 * ca.mtimes(ca.mtimes(full_residual.T, full_W), full_residual)

        large_num = 1e20
        lb = ca.fmax(lb, -large_num)
        ub = ca.fmin(ub, large_num)

        cost_jac = ca.gradient(c, x)
        residual_jac = ca.jacobian(full_residual, x)
        constraint_jac = ca.jacobian(g, x)
        cost_hess = ca.triu(
            residual_jac.T @ full_W @ residual_jac
        )  # Gauss-Newton Hessian

        qp_func = ca.Function(
            "stitch_qp_func",
            [x, p],
            [g, cost_jac, cost_hess, constraint_jac, lb, ub],
            ["x", "p"],
            ["constraints", "cost_jac", "cost_hess", "constraint_jac", "lb", "ub"],
        ).expand()

        ls_func = ca.Function(
            "stitch_ls_func",
            [x, p],
            [c, g, lb, ub],
            ["x", "p"],
            ["cost", "constraints", "lb", "ub"],
        ).expand()

        if self.cfg.get("jit_compile", False):
            jit_base_dir = os.path.abspath(f"log/casadi_jit/{self.robot.robot_name}")
            os.makedirs(jit_base_dir, exist_ok=True)

            qp_func = compile_and_load(
                jit_base_dir,
                qp_func,
                "stitch_qp_func",
                recompile=self.cfg.get("jit_recompile", True),
            )
            ls_func = compile_and_load(
                jit_base_dir,
                ls_func,
                "stitch_ls_func",
                recompile=self.cfg.get("jit_recompile", True),
            )

        return qp_func, ls_func
