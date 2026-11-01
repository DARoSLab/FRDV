"""Stateful motion-chunk retargeting with a persistent SQP solver."""

import logging
import time
from dataclasses import replace

import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R

from mocap_retargeting.data_loading.dataset_selection import get_data_type
from mocap_retargeting.kinematic_retargeting.optimization.sqp_solver import SQP

logger = logging.getLogger(__name__)


class RetargetingSolver:
    def __init__(
        self,
        cfg: DictConfig,
        solver: SQP,
        motion_chunk: list,
        q0_robot,
        active_mapping,
        sqp_cfg,
        *,
        context="Motion chunk",
        start_index=0,
    ):
        self.cfg = cfg
        self.solver = solver
        self.motion_chunk = motion_chunk
        self.q0_robot = q0_robot.copy()
        self.active_mapping = active_mapping
        self.sqp_cfg = sqp_cfg
        self.context = context
        self.start_index = start_index
        self.nq = len(q0_robot)
        self.mapping_indices = []
        self.mapping_sites = []

    def _frame_context(self, frame):
        return (
            f"{self.context} | source frame {self.start_index + frame} "
            f"(chunk frame {frame})"
        )

    def _prepare_indices(self, joint_names_ordered):
        self.mapping_indices = []
        self.mapping_sites = []
        for bvh_link, site_name in self.active_mapping.items():
            if bvh_link in joint_names_ordered:
                idx = joint_names_ordered.index(bvh_link)
                self.mapping_indices.append(idx)
            else:
                self.mapping_indices.append(-1)

    def _initialize_orientation(self, joint_names_ordered, first_frame_mocap):
        left_name, right_name = {
            "lafan1": ("LeftUpLeg", "RightUpLeg"),
            "amass": ("L_Hip", "R_Hip"),
        }[get_data_type(self.cfg)]
        left_idx = (
            joint_names_ordered.index(left_name)
            if left_name in self.active_mapping and left_name in joint_names_ordered
            else -1
        )
        right_idx = (
            joint_names_ordered.index(right_name)
            if right_name in self.active_mapping and right_name in joint_names_ordered
            else -1
        )

        if left_idx != -1 and right_idx != -1:
            try:
                l_pos = first_frame_mocap[left_idx]
                r_pos = first_frame_mocap[right_idx]
                delta = l_pos - r_pos

                delta[2] = 0

                norm = np.linalg.norm(delta)
                if norm > 1e-3:
                    delta /= norm

                    target_yaw = np.arctan2(delta[1], delta[0])
                    nominal_yaw = np.pi / 2

                    diff_yaw = target_yaw - nominal_yaw

                    r_rot = R.from_euler("z", diff_yaw)
                    q_root = self.q0_robot[3:7]
                    r_initial = R.from_quat(q_root)

                    r_new = r_rot * r_initial
                    self.q0_robot[3:7] = r_new.as_quat()

                    if self.cfg.verbose:
                        print(
                            f"[INFO] Initialized root yaw to {np.degrees(diff_yaw):.2f} deg based on hip markers."
                        )

            except Exception as e:
                print(f"[WARN] Failed to initialize orientation: {e}")

    def retarget(self, joint_names_ordered):
        self._prepare_indices(joint_names_ordered)

        if self.motion_chunk is not None and len(self.motion_chunk) > 0:
            self._initialize_orientation(joint_names_ordered, self.motion_chunk[0])

        q_trajectory = []
        mocap_trajectory = []

        current_q = self.q0_robot.copy()
        warmup_failures = 0
        frame_failures = 0
        warmup_attempts = 0
        n_warmup = self.cfg.get("warmup_iterations", 20)
        if (
            self.motion_chunk is not None
            and len(self.motion_chunk) > 0
            and n_warmup > 0
        ):
            first_frame = self.motion_chunk[0]
            warmup_sqp_cfg = replace(
                self.sqp_cfg,
                max_sqp_iter=self.sqp_cfg.max_sqp_iter * self.cfg.optimizer.sqp_boost,
            )
            for w in range(n_warmup):
                warmup_attempts += 1
                params = [current_q.copy()]
                for idx in self.mapping_indices:
                    if idx != -1:
                        params.append(first_frame[idx])
                    else:
                        params.append(np.zeros(3))
                params = np.concatenate(params).reshape(-1)
                try:
                    current_q, stats = self.solver.solve(
                        current_q, params, cfg=warmup_sqp_cfg,
                        context=f"{self._frame_context(0)} | warmup iteration {w}",
                    )
                    warmup_failures += int(stats.qp_failed)
                except Exception as e:
                    if w == 0:
                        print(f"[WARN] Warmup solve failed ({e}), using T-pose.")
                        current_q = self.q0_robot.copy()
                    break
            if self.cfg.verbose:
                print(f"[INFO] Warmup: {n_warmup} iterations on first frame complete.")

        q_prev = current_q.copy()

        qp_func_times = []
        line_search_times = []
        qp_solve_times = []
        frame_times = []

        total_start_time = time.perf_counter()

        # self.motion_chunk is now a tensor of shape (N, n_joints, 3)
        num_frames = self.motion_chunk.shape[0]

        for i in range(num_frames):
            frame_start_time = time.perf_counter()

            frame_positions = self.motion_chunk[i]
            mocap_trajectory.append(frame_positions)

            params = [q_prev]

            for idx in self.mapping_indices:
                if idx != -1:
                    params.append(frame_positions[idx])
                else:
                    params.append(np.zeros(3))

            params = np.concatenate(params).reshape(-1)

            # Adaptive SQP iterations: boost for first 20 frames of the chunk
            current_sqp_cfg = self.sqp_cfg
            if i < self.cfg.optimizer.sqp_boost_frames:
                current_sqp_cfg = replace(
                    self.sqp_cfg,
                    max_sqp_iter=self.sqp_cfg.max_sqp_iter
                    * self.cfg.optimizer.sqp_boost,
                )

            try:
                current_q, stats = self.solver.solve(
                    current_q, params, cfg=current_sqp_cfg,
                    context=self._frame_context(i),
                )
                frame_failures += int(stats.qp_failed)
                q_prev = current_q.copy()
                q_trajectory.append(current_q)

                qp_func_times.append(stats.qp_func_time)
                line_search_times.append(stats.line_search_time)
                qp_solve_times.append(stats.qp_solve_time)

            except Exception as e:
                print(f"Solver failed at frame {i}: {e}")
                q_trajectory.append(current_q)

            frame_times.append(time.perf_counter() - frame_start_time)

        total_time = time.perf_counter() - total_start_time

        if warmup_failures or frame_failures:
            logger.warning(
                "QP failure summary for %s: %d/%d frame solves and %d/%d warmup "
                "solves failed; keeping the returned poses.",
                self.context, frame_failures, num_frames,
                warmup_failures, warmup_attempts,
            )

        if qp_func_times and self.cfg.verbose:
            print("\n--- Profiling Stats (Chunk) ---")
            print(
                f"QP Func Time: Avg={np.mean(qp_func_times) * 1000:.4f}ms, Min={np.min(qp_func_times) * 1000:.4f}ms, Max={np.max(qp_func_times) * 1000:.4f}ms"
            )
            print(
                f"Line Search Time: Avg={np.mean(line_search_times) * 1000:.4f}ms, Min={np.min(line_search_times) * 1000:.4f}ms, Max={np.max(line_search_times) * 1000:.4f}ms"
            )
            print(
                f"QP Solve Time: Avg={np.mean(qp_solve_times) * 1000:.4f}ms, Min={np.min(qp_solve_times) * 1000:.4f}ms, Max={np.max(qp_solve_times) * 1000:.4f}ms"
            )
            print(
                f"Frame Processing Time: Avg={np.mean(frame_times) * 1000:.4f}ms, Min={np.min(frame_times) * 1000:.4f}ms, Max={np.max(frame_times) * 1000:.4f}ms"
            )
            print(f"Total Chunk Time: {total_time:.4f}s")
            print("-----------------------\n")

        return q_trajectory, mocap_trajectory
