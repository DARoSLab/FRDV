"""Interactive mocap calibration, alignment helpers, and robot/config export."""

import colorsys
import os
import threading
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import viser
from omegaconf import DictConfig, OmegaConf

from mocap_retargeting import PROJECT_ROOT
from mocap_retargeting.data_loading.amass.loader import AMASSLoader
from mocap_retargeting.data_loading.dataset_selection import (
    get_data_type,
    load_mocap,
    task_config_filename,
)
from mocap_retargeting.data_loading.mocap_transform import (
    resolve_mocap_transform,
    save_mocap_transform,
)
from mocap_retargeting.visualization.mujoco_scene import ViserMujocoScene


def prepare_calibration_positions(
    loader,
    scale,
    rotation_matrix,
    x_offset,
    y_offset,
    z_offset,
    *,
    ground,
    upper_body_offset=None,
    ground_clearance=0.05,
):
    manual_offset = np.array([x_offset, y_offset, z_offset])
    preprocess_kwargs = {}
    if upper_body_offset is not None and loader.supports_upper_body_offset:
        preprocess_kwargs["upper_body_offset"] = list(upper_body_offset)
    if isinstance(loader, AMASSLoader):
        preprocess_kwargs["ground_clearance"] = ground_clearance
    loader.preprocess(
        scale=scale,
        rotation_matrix=rotation_matrix,
        zero_root_xy=True,
        ground=ground,
        manual_offset=manual_offset,
        **preprocess_kwargs,
    )
    if isinstance(loader, AMASSLoader):
        return loader.get_tpose_positions()
    return loader.get_frame_positions(0)


def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.8
        value = 0.9
        colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
    return colors


def process_mocap_sites(
    cfg: DictConfig,
    mj_model,
    mj_data,
    loader,
    scale: np.ndarray,
    rotation_matrix: np.ndarray,
    config_path: str,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
    save: bool = True,
    compute_offsets: bool = True,
    input_xml_path: str = None,
    subject_transform=None,
    upper_body_offset=None,
):
    manual_offset = np.array([x_offset, y_offset, z_offset])
    if upper_body_offset is None:
        upper_body_offset = (
            subject_transform.upper_body_offset
            if subject_transform is not None
            else cfg.get("upper_body_offset")
        )
    positions = prepare_calibration_positions(
        loader,
        scale,
        rotation_matrix,
        x_offset,
        y_offset,
        z_offset,
        ground=cfg.mocap.get("ground", get_data_type(cfg) == "lafan1"),
        upper_body_offset=upper_body_offset,
        ground_clearance=cfg.mocap.get("ground_clearance", 0.05),
    )
    if compute_offsets:
        xml_to_parse = (
            input_xml_path
            if input_xml_path and os.path.exists(input_xml_path)
            else cfg.mujoco_src_xml_path
        )
        print(f"[INFO] Processing XML from: {xml_to_parse}")
        tree = ET.parse(xml_to_parse)
        root = tree.getroot()

        def find_body(element, body_name):
            if element.tag == "body" and element.get("name") == body_name:
                return element
            for child in element:
                res = find_body(child, body_name)
                if res is not None:
                    return res
            return None

        mapping_data = OmegaConf.to_container(cfg.mapping)
        post_mapping_list = []
        if mapping_data:
            for item in mapping_data:
                bvh_joint = item.get("bvh")
                robot_body = item.get("robot")
                color = item.get("color", [1.0, 1.0, 1.0])

                if bvh_joint in positions:
                    bvh_pos = positions[bvh_joint]

                    # Get robot body position and rotation
                    body_id = mujoco.mj_name2id(
                        mj_model, mujoco.mjtObj.mjOBJ_BODY, robot_body
                    )
                    if body_id != -1:
                        robot_pos = mj_data.xpos[body_id]
                        robot_rot = mj_data.xmat[body_id].reshape(3, 3)

                        # Calculate local offset: P_local = R_body^T * (P_bvh - P_body)
                        offset_global = bvh_pos - robot_pos
                        offset_local = robot_rot.T @ offset_global

                        print(
                            f"[INFO] Adding site for {bvh_joint} on {robot_body} at offset {offset_local}"
                        )

                        # Find body element in XML
                        body_element = find_body(root, robot_body)
                        if body_element is not None:
                            # Create site element
                            site_name = f"mocap_{bvh_joint}"

                            # Remove existing site if present to avoid duplicates
                            for child in body_element.findall("site"):
                                if child.get("name") == site_name:
                                    body_element.remove(child)

                            site_elem = ET.Element("site")
                            site_elem.set("name", site_name)
                            site_elem.set(
                                "pos",
                                f"{offset_local[0]:.6f} {offset_local[1]:.6f} {offset_local[2]:.6f}",
                            )
                            site_elem.set("size", "0.01")
                            site_elem.set("rgba", f"{color[0]} {color[1]} {color[2]} 1")

                            body_element.append(site_elem)

                            # Add to post mapping
                            post_mapping_list.append(
                                {
                                    "bvh": bvh_joint,
                                    "robot": site_name,  # Map to the site name
                                    "color": color,
                                }
                            )
                        else:
                            print(
                                f"[WARNING] Body element '{robot_body}' not found in XML."
                            )
                    else:
                        print(
                            f"[WARNING] Robot body '{robot_body}' not found in model."
                        )

        if save:
            print(f"[INFO] Saving modified XML to: {cfg.mujoco_dst_xml_path}")
            ET.indent(tree, space="  ", level=0)
            tree.write(cfg.mujoco_dst_xml_path, encoding="utf-8", xml_declaration=True)

            try:
                saved_path = save_mocap_transform(
                    subject_transform,
                    config_path,
                    scale,
                    rotation_matrix,
                    manual_offset,
                    upper_body_offset=upper_body_offset,
                )
                print(f"[INFO] Transform saved to: {saved_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save config: {e}")

            print("[INFO] Done!")

    return positions, loader


class CalibrationApp:
    """Own application state, GUI callbacks, and the application lifecycle."""

    def __init__(self, cfg, config_path=None):
        self.cfg = cfg
        self.config_path = config_path or os.path.join(
            PROJECT_ROOT, "conf", "task", cfg.robot, task_config_filename(cfg)
        )

    def _rotation_from_controls(self):
        """Apply the GUI yaw adjustment to the full configured rotation."""
        theta = np.radians(self.yaw_slider.value)
        c, s = np.cos(theta), np.sin(theta)
        yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return yaw @ self.transform.rotation

    def _upper_body_offset_from_controls(self):
        if self.upper_body_offset_sliders:
            return np.array([slider.value for slider in self.upper_body_offset_sliders])
        return self.transform.upper_body_offset

    def update_robot_pose(self):
        if self.root_height_slider and self.root_joint_id != -1:
            qpos_adr = self.mj_model.jnt_qposadr[self.root_joint_id]
            self.mj_data.qpos[qpos_adr + 2] = self.root_height_slider.value
        for jnt_name, (jnt_id, slider) in self.joint_sliders.items():
            qpos_adr = self.mj_model.jnt_qposadr[jnt_id]
            self.mj_data.qpos[qpos_adr] = slider.value
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.scene.update_from_mjdata(self.mj_data)

    def update_visualization(self):
        if not self.update_lock.acquire(blocking=False):
            return
        try:
            s_x = self.scale_x_slider.value
            s_y = self.scale_y_slider.value
            s_z = self.scale_z_slider.value
            scale = np.array([s_x, s_y, s_z])
            x_off = self.x_offset_slider.value
            y_off = self.y_offset_slider.value
            z_off = self.z_offset_slider.value
            R = self._rotation_from_controls()
            try:
                positions, _ = process_mocap_sites(
                    self.cfg,
                    self.mj_model,
                    self.mj_data,
                    self.loader,
                    scale,
                    R,
                    self.config_path,
                    x_offset=x_off,
                    y_offset=y_off,
                    z_offset=z_off,
                    save=False,
                    compute_offsets=False,
                    subject_transform=self.transform,
                    upper_body_offset=self._upper_body_offset_from_controls(),
                )
            except Exception as e:
                print(f"Error processing mocap: {e}")
                return
            mocap_points = []
            mocap_colors = []
            for name, pos in positions.items():
                mocap_points.append(pos)
                if name in self.bvh_color_map:
                    mocap_colors.append(self.bvh_color_map[name])
                else:
                    mocap_colors.append([0, 0, 0])
            mocap_points = np.array(mocap_points)
            mocap_colors = np.array(mocap_colors)
            if len(mocap_points) > 0:
                if self.point_cloud_handle is None:
                    self.point_cloud_handle = self.server.scene.add_point_cloud(
                        "/mocap_joints",
                        points=mocap_points,
                        colors=mocap_colors,
                        point_size=0.03,
                        point_shape="circle",
                    )
                else:
                    self.point_cloud_handle.points = mocap_points
                    self.point_cloud_handle.colors = mocap_colors
                bones = self.loader.get_skeleton_structure()
                bone_segments = []
                for parent_name, child_name in bones:
                    if parent_name in positions and child_name in positions:
                        start = positions[parent_name]
                        end = positions[child_name]
                        bone_segments.append([start, end])
                if bone_segments:
                    bone_points = np.array(bone_segments)
                    if self.bones_handle is None:
                        self.bones_handle = self.server.scene.add_line_segments(
                            "/mocap_bones",
                            points=bone_points,
                            colors=np.array([0, 0, 0]),
                            line_width=3.0,
                        )
                    else:
                        self.bones_handle.points = bone_points
            corr_segments = []
            corr_colors = []
            for i, item in enumerate(self.mapping_data):
                bvh = item.get("bvh")
                robot = item.get("robot")
                if bvh in positions:
                    mocap_pos = positions[bvh]
                    bid = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_BODY, robot
                    )
                    if bid != -1:
                        robot_pos = self.mj_data.xpos[bid]
                        corr_segments.append([mocap_pos, robot_pos])
                        c = self.distinct_colors[i]
                        corr_colors.append([c, c])
            if corr_segments:
                self.server.scene.add_line_segments(
                    "/correspondences",
                    points=np.array(corr_segments),
                    colors=np.array(corr_colors),
                    line_width=5.0,
                )
        finally:
            self.update_lock.release()

    def _on_save_all_button(self, _):
        print("Saving All (T-Pose and Mocap Sites)...")
        print("Saving T-Pose...")
        current_qpos = self.mj_data.qpos.copy()
        qpos_str = " ".join(map(str, current_qpos))
        qvel_str = " ".join(["0"] * self.mj_model.nv)
        xml_path = self.cfg.mujoco_dst_xml_path
        if not os.path.exists(xml_path):
            print(
                f"[INFO] Destination XML {xml_path} not found. Starting from source: {self.cfg.mujoco_src_xml_path}"
            )
            source_xml = self.cfg.mujoco_src_xml_path
        else:
            source_xml = xml_path
        try:
            tree = ET.parse(source_xml)
            root = tree.getroot()
            keyframe_section = root.find("keyframe")
            if keyframe_section is None:
                keyframe_section = ET.SubElement(root, "keyframe")
            t_pose_key = None
            for key in keyframe_section.findall("key"):
                if key.get("name") == "t_pose":
                    t_pose_key = key
                    break
            if t_pose_key is None:
                t_pose_key = ET.SubElement(keyframe_section, "key")
                t_pose_key.set("name", "t_pose")
            t_pose_key.set("qpos", qpos_str)
            t_pose_key.set("qvel", qvel_str)
            ET.indent(tree, space="  ", level=0)
            os.makedirs(os.path.dirname(xml_path), exist_ok=True)
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            print(f"T-Pose saved to {xml_path}")
        except Exception as e:
            print(f"Error saving T-Pose: {e}")
        print("Saving mocap sites...")
        s_x = self.scale_x_slider.value
        s_y = self.scale_y_slider.value
        s_z = self.scale_z_slider.value
        scale = np.array([s_x, s_y, s_z])
        x_off = self.x_offset_slider.value
        y_off = self.y_offset_slider.value
        z_off = self.z_offset_slider.value
        R = self._rotation_from_controls()
        process_mocap_sites(
            self.cfg,
            self.mj_model,
            self.mj_data,
            self.loader,
            scale,
            R,
            self.config_path,
            x_offset=x_off,
            y_offset=y_off,
            z_offset=z_off,
            save=True,
            compute_offsets=True,
            input_xml_path=self.cfg.mujoco_dst_xml_path,
            subject_transform=self.transform,
            upper_body_offset=self._upper_body_offset_from_controls(),
        )
        print("Mocap sites saved!")
        print("All saved successfully!")

    def initialize(self):
        self.transform = resolve_mocap_transform(self.cfg)
        print(f"[INFO] Loading robot model from: {self.cfg.mujoco_src_xml_path}")
        self.mj_model = mujoco.MjModel.from_xml_path(self.cfg.mujoco_src_xml_path)
        self.server = viser.ViserServer(port=8080)
        self.scene = ViserMujocoScene.create(self.server, self.mj_model, num_envs=1)
        self.mj_data = self.scene.mj_data
        key_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "t_pose")
        if key_id != -1:
            print(f"[INFO] Found keyframe 't_pose' at index {key_id}")
            q0_robot = self.mj_model.key_qpos[key_id]
            self.mj_data.qpos = q0_robot
            mujoco.mj_forward(self.mj_model, self.mj_data)
        else:
            print("[WARNING] Keyframe 't_pose' not found. Using default pose.")
        print(f"[INFO] Loading mocap data from: {self.cfg.mocap_src_path}")
        self.loader = load_mocap(self.cfg.mocap_src_path, self.cfg)
        print(f"[INFO] Loaded mocap data from: {self.cfg.mocap_src_path}")
        self.scene.update_from_mjdata(self.mj_data)
        self.scene.create_visualization_gui(
            show_environment_gui=False,
            show_contacts_gui=False,
            show_debug_viz_control=False,
        )
        self.server.scene.add_grid(
            "ground_grid", width=10, height=10, cell_size=1.0, plane="xy"
        )
        self.server.scene.add_frame(
            "world_axes", show_axes=True, axes_length=0.5, axes_radius=0.01
        )
        with self.server.gui.add_folder("Mocap Transform"):
            # Scale axes refer to the source frame, before the configured rotation.
            self.scale_x_slider = self.server.gui.add_slider(
                "Scale X",
                min=min(0.001, self.transform.scale[0]),
                max=max(1.0, self.transform.scale[0]),
                step=1e-05,
                initial_value=self.transform.scale[0],
            )
            self.scale_y_slider = self.server.gui.add_slider(
                "Scale Y",
                min=min(0.001, self.transform.scale[1]),
                max=max(1.0, self.transform.scale[1]),
                step=1e-05,
                initial_value=self.transform.scale[1],
            )
            self.scale_z_slider = self.server.gui.add_slider(
                "Scale Z",
                min=min(0.001, self.transform.scale[2]),
                max=max(1.0, self.transform.scale[2]),
                step=1e-05,
                initial_value=self.transform.scale[2],
            )
            self.yaw_slider = self.server.gui.add_slider(
                "Yaw Adjustment (deg)", min=-180, max=180, step=1, initial_value=0
            )
            self.x_offset_slider = self.server.gui.add_slider(
                "X Offset",
                min=min(-2.0, self.transform.offset[0]),
                max=max(2.0, self.transform.offset[0]),
                step=0.01,
                initial_value=self.transform.offset[0],
            )
            self.y_offset_slider = self.server.gui.add_slider(
                "Y Offset",
                min=min(-2.0, self.transform.offset[1]),
                max=max(2.0, self.transform.offset[1]),
                step=0.01,
                initial_value=self.transform.offset[1],
            )
            self.z_offset_slider = self.server.gui.add_slider(
                "Z Offset",
                min=min(-2.0, self.transform.offset[2]),
                max=max(2.0, self.transform.offset[2]),
                step=0.01,
                initial_value=self.transform.offset[2],
            )
            self.upper_body_offset_sliders = []
            if get_data_type(self.cfg) == "amass":
                upper_offset = self.transform.upper_body_offset
                if upper_offset is None:
                    upper_offset = np.zeros(3)
                for axis, value in zip("XYZ", upper_offset):
                    self.upper_body_offset_sliders.append(
                        self.server.gui.add_slider(
                            f"Upper Body Offset {axis}",
                            min=min(-2.0, value),
                            max=max(2.0, value),
                            step=0.001,
                            initial_value=float(value),
                        )
                    )
        with self.server.gui.add_folder("Robot T-Pose Adjustment"):
            self.root_joint_id = -1
            for i in range(self.mj_model.njnt):
                if self.mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    self.root_joint_id = i
                    break
            self.root_height_slider = None
            if self.root_joint_id != -1:
                qpos_adr = self.mj_model.jnt_qposadr[self.root_joint_id]
                current_root_z = self.mj_data.qpos[qpos_adr + 2]
                self.root_height_slider = self.server.gui.add_slider(
                    "Root Height (Z)",
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    initial_value=float(current_root_z),
                )
            self.joint_sliders = {}
            for i in range(self.mj_model.njnt):
                jnt_type = self.mj_model.jnt_type[i]
                jnt_name = mujoco.mj_id2name(
                    self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i
                )
                if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
                    qpos_adr = self.mj_model.jnt_qposadr[i]
                    current_val = self.mj_data.qpos[qpos_adr]
                    if self.mj_model.jnt_limited[i]:
                        min_val, max_val = self.mj_model.jnt_range[i]
                    else:
                        min_val, max_val = (-3.14, 3.14)
                    clamped_val = max(min_val, min(max_val, float(current_val)))
                    slider = self.server.gui.add_slider(
                        f"{jnt_name}",
                        min=min_val,
                        max=max_val,
                        step=0.01,
                        initial_value=clamped_val,
                    )
                    self.joint_sliders[jnt_name] = (i, slider)
        if self.root_height_slider:
            self.root_height_slider.on_update(lambda _: self.update_robot_pose())
        for _, (_, slider) in self.joint_sliders.items():
            slider.on_update(lambda _: self.update_robot_pose())
        self.mapping_data = OmegaConf.to_container(self.cfg.mapping)
        self.bvh_color_map = {}
        robot_body_map = []
        if self.mapping_data:
            num_mappings = len(self.mapping_data)
            self.distinct_colors = generate_distinct_colors(num_mappings)
            for i, item in enumerate(self.mapping_data):
                bvh = item.get("bvh")
                robot = item.get("robot")
                color = self.distinct_colors[i]
                rgb_255 = np.array(color)
                if bvh:
                    self.bvh_color_map[bvh] = rgb_255
                if robot:
                    robot_body_map.append((robot, rgb_255))
        robot_points = []
        robot_colors = []
        for body_name, color in robot_body_map:
            bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid != -1:
                robot_points.append(self.mj_data.xpos[bid])
                robot_colors.append(color)
        self.point_cloud_handle = None
        self.bones_handle = None
        self.update_lock = threading.Lock()
        self.scale_x_slider.on_update(lambda _: self.update_visualization())
        self.scale_y_slider.on_update(lambda _: self.update_visualization())
        self.scale_z_slider.on_update(lambda _: self.update_visualization())
        self.yaw_slider.on_update(lambda _: self.update_visualization())
        self.x_offset_slider.on_update(lambda _: self.update_visualization())
        self.y_offset_slider.on_update(lambda _: self.update_visualization())
        self.z_offset_slider.on_update(lambda _: self.update_visualization())
        for slider in self.upper_body_offset_sliders:
            slider.on_update(lambda _: self.update_visualization())
        save_all_button = self.server.gui.add_button("Save All")
        save_all_button.on_click(self._on_save_all_button)
        with self.server.gui.add_folder("Geometry Groups", expand_by_default=False):
            tabs = self.server.gui.add_tab_group()
            self.scene.create_geom_groups_gui(tabs)
        self.update_visualization()

    def run(self):
        self.initialize()
        print("Visualization started. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Exiting...")
