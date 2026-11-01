"""Playback application state, controls, trajectory loading, and frame updates."""

import logging
import os
import random
import threading
import time

import mujoco
import numpy as np
import trimesh
import viser

from mocap_retargeting.mujoco_integration.mujoco_interface import pin_to_mj
from mocap_retargeting.retargeted_motion_io.retargeted_motion_io import load_trajectory

from .mesh_conversion import rotation_quat_from_vectors
from .mujoco_scene import ViserMujocoScene

logger = logging.getLogger(__name__)


def create_playback_controls(app):
    with app.server.gui.add_folder("Playback Controls"):
        app.shuffle_btn = app.server.gui.add_button("Shuffle", color="purple")
        app.record_btn = app.server.gui.add_button("Record", color="red")
        app.play_btn = app.server.gui.add_button("Play", color="green")
        app.pause_btn = app.server.gui.add_button("Pause", color="yellow")
        app.stop_btn = app.server.gui.add_button("Stop", color="red")
        app.frame_slider = app.server.gui.add_slider(
            "Frame", min=0, max=100, step=1, initial_value=0
        )
        app.speed_slider = app.server.gui.add_slider(
            "Speed", min=0.1, max=20.0, step=0.1, initial_value=1.0
        )
        app.loop_checkbox = app.server.gui.add_checkbox("Loop", initial_value=False)
        app.mocap_checkbox = app.server.gui.add_checkbox(
            "Show Mocap", initial_value=False
        )
        app.sites_checkbox = app.server.gui.add_checkbox(
            "Show Sites", initial_value=True
        )


class PlaybackApp:
    """Own application state, GUI callbacks, and the application lifecycle."""

    def __init__(self, args, xml_path):
        self.args = args
        self.xml_path = xml_path

    def _on_mocap_checkbox(self, _):
        self.state["show_mocap"] = self.mocap_checkbox.value
        if not self.state["show_mocap"]:
            for motion in self.state["motions"]:
                if "point_cloud_handle" in motion:
                    motion["point_cloud_handle"].remove()
                    del motion["point_cloud_handle"]
                if "bones_handle" in motion:
                    motion["bones_handle"].remove()
                    del motion["bones_handle"]

    def _on_sites_checkbox(self, _):
        self.state["show_sites"] = self.sites_checkbox.value
        if not self.state["show_sites"]:
            for motion in self.state["motions"]:
                if "sites_handle" in motion:
                    motion["sites_handle"].remove()
                    del motion["sites_handle"]

    def scan_files(self):
        if os.path.isdir(self.args.path):
            files = [f for f in os.listdir(self.args.path) if f.endswith(".npy")]
            files.sort()
            return [os.path.join(self.args.path, f) for f in files]
        elif os.path.isfile(self.args.path) and self.args.path.endswith(".npy"):
            return [self.args.path]
        else:
            print(f"Invalid path: {self.args.path}")
            return []

    def cleanup_visualization(self):
        if self.state["mj_scene"]:
            for motion in self.state["motions"]:
                if "point_cloud_handle" in motion:
                    motion["point_cloud_handle"].remove()
                if "bones_handle" in motion:
                    motion["bones_handle"].remove()

    def load_motions(self, paths):
        if not paths:
            return
        with self.state["lock"]:
            self.cleanup_visualization()
            self.state["motions"] = []
            motions = []
        max_frames = 0
        for i, filepath in enumerate(paths):
            print(f"Loading {filepath}...")
            try:
                data = load_trajectory(filepath)
                filename = os.path.basename(filepath)
                import math

                spacing = 2.0
                cols = math.ceil(math.sqrt(len(paths)))
                rows = math.ceil(len(paths) / cols)
                row = i // cols
                col = i % cols
                x_offset = (col - (cols - 1) / 2) * spacing
                y_offset = -(row - (rows - 1) / 2) * spacing
                offset = np.array([x_offset, y_offset, 0])
                motions.append({"data": data, "offset": offset, "name": filename})
                num_frames = len(data["q"])
                if num_frames > max_frames:
                    max_frames = num_frames
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")
        if not motions:
            print("No valid motions loaded.")
            return
        fps_values = sorted({motion["data"].get("fps", 30) for motion in motions})
        if len(fps_values) > 1:
            logger.warning(
                "Loaded motions have different FPS values (%s). Playback uses the "
                "first motion's rate (%g FPS) and the same frame index for every "
                "motion; motions with other rates will play at a different speed.",
                ", ".join(f"{fps:g}" for fps in fps_values),
                motions[0]["data"].get("fps", 30),
            )
        self.state["motions"] = motions
        self.state["total_frames"] = max_frames
        self.frame_slider.max = max_frames - 1
        print(f"Loading robot model from {self.xml_path}")
        self.state["mj_model"] = mujoco.MjModel.from_xml_path(self.xml_path)
        self.state["mj_data"] = mujoco.MjData(self.state["mj_model"])
        mocap_site_ids = []
        for sid in range(self.state["mj_model"].nsite):
            name = mujoco.mj_id2name(
                self.state["mj_model"], mujoco.mjtObj.mjOBJ_SITE, sid
            )
            if name and name.startswith("mocap_"):
                mocap_site_ids.append(sid)
        self.state["mocap_site_ids"] = mocap_site_ids
        self.state["mj_scene"] = ViserMujocoScene.create(
            self.server, self.state["mj_model"], num_envs=len(motions)
        )
        self.state["mj_scene"].create_visualization_gui(
            show_debug_viz_control=True,
            show_environment_gui=True,
            show_contacts_gui=True,
        )
        self.state["mj_scene"].fixed_bodies_frame.visible = False
        self.state["mj_scene"].show_contact_points = False
        self.state["mj_scene"].show_contact_forces = False
        self.update_visualization(0)

    def _on_shuffle_btn(self, _):
        if not self.state["all_available_files"]:
            print("No files to shuffle.")
            return
        n = (
            self.args.num
            if self.args.num is not None
            else len(self.state["all_available_files"])
        )
        n = min(n, len(self.state["all_available_files"]))
        selected_files = random.sample(self.state["all_available_files"], n)
        self.load_motions(selected_files)

    def _on_record_btn(self, _):
        if not self.state["recording"]:
            print("Starting recording...")
            self.state["recording"] = True
            self.record_btn.label = "Stop Recording"
            self.record_btn.color = "gray"
            self.state["serializer"] = self.server.get_scene_serializer()
        else:
            print("Stopping recording...")
            self.state["recording"] = False
            self.record_btn.label = "Record"
            self.record_btn.color = "red"
            if self.state["serializer"]:
                with self.state["lock"]:
                    try:
                        os.makedirs("recordings/private", exist_ok=True)
                        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                        filename = f"recordings/private/playback_{timestamp}.viser"
                        serializer = self.state["serializer"]
                        self.state["serializer"] = None
                        recording_data = serializer.serialize()
                        with open(filename, "wb") as f:
                            f.write(recording_data)
                        print(f"Saved recording to {filename}")
                    except Exception as e:
                        print(f"Error saving recording: {e}")
                        self.state["serializer"] = None

    def update_visualization(self, frame_idx):
        if not self.state["motions"] or self.state["mj_scene"] is None:
            return
        if self.state["lock"].locked():
            return
        num_envs = len(self.state["motions"])
        nbody = self.state["mj_model"].nbody
        nmocap = self.state["mj_model"].nmocap
        batch_xpos = np.zeros((num_envs, nbody, 3))
        batch_xmat = np.zeros((num_envs, nbody, 3, 3))
        batch_mocap_pos = np.zeros((num_envs, nmocap, 3))
        batch_mocap_quat = np.zeros((num_envs, nmocap, 4))
        scene_offset = np.zeros(3)
        for i, motion in enumerate(self.state["motions"]):
            data = motion["data"]
            offset = motion["offset"]
            num_frames = len(data["q"])
            current_frame = min(frame_idx, num_frames - 1)
            q = data["q"][current_frame]
            if self.args.quat_format == "xyzw":
                q_mujoco = pin_to_mj(q)
            else:
                q_mujoco = q.copy()
            q_mujoco[:3] += offset
            self.state["mj_data"].qpos = q_mujoco
            mujoco.mj_forward(self.state["mj_model"], self.state["mj_data"])
            batch_xpos[i] = self.state["mj_data"].xpos
            batch_xmat[i] = self.state["mj_data"].xmat.reshape(-1, 3, 3)
            batch_mocap_pos[i] = self.state["mj_data"].mocap_pos
            batch_mocap_quat[i] = self.state["mj_data"].mocap_quat
            if self.state["show_sites"] and self.state.get("mocap_site_ids"):
                site_positions = self.state["mj_data"].site_xpos[
                    self.state["mocap_site_ids"]
                ]
                points_array = np.array(site_positions, dtype=np.float32)
                colors_array = np.full(
                    (len(points_array), 3), [1.0, 0.0, 0.0], dtype=np.float32
                )
                if "sites_handle" not in motion:
                    motion["sites_handle"] = self.server.scene.add_point_cloud(
                        f"/mocap_sites_{motion['name']}",
                        points=points_array,
                        colors=colors_array,
                        point_size=0.03,
                        point_shape="circle",
                    )
                else:
                    motion["sites_handle"].points = points_array
            if self.state["show_mocap"] and "mocap" in data:
                mocap_data = data["mocap"][current_frame]
                if isinstance(mocap_data, np.ndarray):
                    if "joint_names" in data and data["joint_names"] is not None:
                        joint_names = data["joint_names"]
                        mocap_dict = {}
                        for idx, name in enumerate(joint_names):
                            if idx < len(mocap_data):
                                mocap_dict[name] = mocap_data[idx]
                        mocap_data = mocap_dict
                    else:
                        pass
                points = []
                colors = []
                if isinstance(mocap_data, dict):
                    for name, pos in mocap_data.items():
                        points.append(pos + offset)
                        colors.append([0.0, 0.0, 0.0])
                    points_array = np.array(points, dtype=np.float32)
                    colors_array = np.array(colors, dtype=np.float32)
                    if "point_cloud_handle" not in motion:
                        motion["point_cloud_handle"] = (
                            self.server.scene.add_point_cloud(
                                f"/mocap_joints_{motion['name']}",
                                points=points_array,
                                colors=colors_array,
                                point_size=0.02,
                                point_shape="circle",
                            )
                        )
                    else:
                        motion["point_cloud_handle"].points = points_array
            if self.state["show_mocap"] and "skeleton" in data:
                bones = data["skeleton"]
                bone_segments = []
                for parent_name, child_name in bones:
                    if parent_name in mocap_data and child_name in mocap_data:
                        start = mocap_data[parent_name] + offset
                        end = mocap_data[child_name] + offset
                        bone_segments.append([start, end])
                if bone_segments:
                    starts = np.array([s for s, e in bone_segments], dtype=np.float32)
                    ends = np.array([e for s, e in bone_segments], dtype=np.float32)
                    num_bones = len(starts)
                    batched_positions = np.zeros((num_bones, 3), dtype=np.float32)
                    batched_wxyzs = np.zeros((num_bones, 4), dtype=np.float32)
                    batched_scales = np.zeros((num_bones, 3), dtype=np.float32)
                    batched_colors = np.zeros((num_bones, 3), dtype=np.uint8)
                    batched_colors[:] = [0, 0, 0]
                    z_axis = np.array([0, 0, 1.0])
                    directions = ends - starts
                    lengths = np.linalg.norm(directions, axis=1)
                    valid_indices = lengths > 1e-06
                    if np.any(valid_indices):
                        directions[valid_indices] /= lengths[valid_indices][:, None]
                        for b_idx in range(num_bones):
                            if not valid_indices[b_idx]:
                                continue
                            batched_positions[b_idx] = starts[b_idx]
                            batched_scales[b_idx] = [0.015, 0.015, lengths[b_idx]]
                            batched_wxyzs[b_idx] = rotation_quat_from_vectors(
                                z_axis, directions[b_idx]
                            )
                        if (
                            "bones_handle" not in motion
                            or motion.get("num_bones") != num_bones
                        ):
                            if "bones_handle" in motion:
                                motion["bones_handle"].remove()
                            motion["bones_handle"] = (
                                self.server.scene.add_batched_meshes_simple(
                                    f"/mocap_bones_mesh_{motion['name']}",
                                    vertices=self.state["bone_mesh"].vertices,
                                    faces=self.state["bone_mesh"].faces,
                                    batched_positions=batched_positions,
                                    batched_wxyzs=batched_wxyzs,
                                    batched_scales=batched_scales,
                                    batched_colors=batched_colors,
                                )
                            )
                            motion["num_bones"] = num_bones
                        else:
                            motion["bones_handle"].batched_positions = batched_positions
                            motion["bones_handle"].batched_wxyzs = batched_wxyzs
                            motion["bones_handle"].batched_scales = batched_scales
        contacts = None
        if (
            self.state["mj_scene"].show_contact_points
            or self.state["mj_scene"].show_contact_forces
        ):
            contacts = self.state["mj_scene"]._extract_contacts_from_mjdata(
                self.state["mj_data"]
            )
        self.state["mj_scene"]._update_visualization(
            batch_xpos,
            batch_xmat,
            batch_mocap_pos,
            batch_mocap_quat,
            env_idx=0,
            scene_offset=scene_offset,
            contacts=contacts,
        )

    def playback_loop(self):
        while True:
            if self.state["playing"] and (not self.state["paused"]):
                if self.state["current_frame"] < self.state["total_frames"] - 1:
                    self.state["current_frame"] += 1
                    self.frame_slider.value = self.state["current_frame"]
                    self.update_visualization(self.state["current_frame"])
                    fps = 30
                    if self.state["motions"]:
                        fps = self.state["motions"][0]["data"].get("fps", 30)
                    time.sleep(1.0 / (fps * self.state["speed"]))
                    if self.state["recording"] and self.state["serializer"]:
                        self.server.flush()
                        self.state["serializer"].insert_sleep(
                            1.0 / (fps * self.state["speed"])
                        )
                elif self.state["loop"]:
                    self.state["current_frame"] = 0
                else:
                    self.state["playing"] = False
                    self.play_btn.label = "Play"
            else:
                time.sleep(0.1)
            if self.state["stop_playback"]:
                break

    def _on_play_btn(self, _):
        self.state["playing"] = True
        self.state["paused"] = False
        self.play_btn.label = "Playing..."

    def _on_pause_btn(self, _):
        self.state["paused"] = not self.state["paused"]
        self.pause_btn.label = "Resume" if self.state["paused"] else "Pause"

    def _on_stop_btn(self, _):
        self.state["playing"] = False
        self.state["paused"] = False
        self.state["current_frame"] = 0
        self.frame_slider.value = 0
        self.update_visualization(0)
        self.play_btn.label = "Play"
        self.pause_btn.label = "Pause"

    def _on_frame_slider(self, _):
        if not self.state["playing"]:
            self.state["current_frame"] = int(self.frame_slider.value)
            self.update_visualization(self.state["current_frame"])

    def _on_speed_slider(self, _):
        self.state["speed"] = self.speed_slider.value

    def _on_loop_checkbox(self, _):
        self.state["loop"] = self.loop_checkbox.value

    def initialize(self):
        self.server = viser.ViserServer(port=8085)
        self.server.gui.configure_theme(dark_mode=False)
        self.server.scene.add_grid("ground", width=20, height=20, cell_size=0.5)
        self.server.scene.add_frame("world_frame", show_axes=True, axes_length=0.2)
        self.state = {
            "playing": False,
            "paused": False,
            "current_frame": 0,
            "total_frames": 0,
            "motions": [],
            "mj_model": None,
            "mj_data": None,
            "mj_scene": None,
            "stop_playback": False,
            "speed": 1.0,
            "loop": False,
            "bone_mesh": trimesh.creation.cylinder(radius=1.0, height=1.0),
        }
        self.state["bone_mesh"].apply_translation([0, 0, 0.5])
        self.state["all_available_files"] = []
        self.state["lock"] = threading.Lock()
        self.state["recording"] = False
        self.state["serializer"] = None
        create_playback_controls(self)
        self.state["show_mocap"] = False
        self.state["show_sites"] = False
        self.mocap_checkbox.on_update(self._on_mocap_checkbox)
        self.sites_checkbox.on_update(self._on_sites_checkbox)
        self.shuffle_btn.on_click(self._on_shuffle_btn)
        self.record_btn.on_click(self._on_record_btn)
        self.play_btn.on_click(self._on_play_btn)
        self.pause_btn.on_click(self._on_pause_btn)
        self.stop_btn.on_click(self._on_stop_btn)
        self.frame_slider.on_update(self._on_frame_slider)
        self.speed_slider.on_update(self._on_speed_slider)
        self.loop_checkbox.on_update(self._on_loop_checkbox)
        self.state["all_available_files"] = self.scan_files()
        initial_n = (
            self.args.num
            if self.args.num is not None
            else len(self.state["all_available_files"])
        )
        initial_files = self.state["all_available_files"][:initial_n]
        self.load_motions(initial_files)

    def run(self):
        self.initialize()
        threading.Thread(target=self.playback_loop, daemon=True).start()
        print("Playback script running. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.state["stop_playback"] = True
            print("Exiting...")
