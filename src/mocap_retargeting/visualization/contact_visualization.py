"""Contact visualization."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import trimesh
import viser.transforms as vtf

from .mesh_conversion import rotation_matrix_from_vectors


@dataclass
class _Contact:
    """Contact data from MuJoCo."""

    pos: np.ndarray
    frame: np.ndarray  # 3x3 rotation matrix.
    force: np.ndarray  # Force in contact frame.
    dist: float
    included: bool


@dataclass
class _ContactPointVisual:
    """Visual representation data for a contact point."""

    position: np.ndarray
    orientation: np.ndarray  # Quaternion (wxyz).
    scale: np.ndarray  # [width, width, height].


@dataclass
class _ContactForceVisual:
    """Visual representation data for a contact force arrow."""

    shaft_position: np.ndarray
    shaft_orientation: np.ndarray  # Quaternion (wxyz).
    shaft_scale: np.ndarray  # [width, width, length].
    head_position: np.ndarray
    head_orientation: np.ndarray  # Quaternion (wxyz).
    head_scale: np.ndarray  # [width, width, width].


def _extract_contacts_from_mjdata(scene, mj_data: mujoco.MjData) -> list[_Contact]:
    """Extract contact data from given MuJoCo data."""

    def make_contact(i: int) -> _Contact:
        con, force = mj_data.contact[i], np.zeros(6)
        mujoco.mj_contactForce(scene.mj_model, mj_data, i, force)
        return _Contact(
            pos=con.pos.copy(),
            frame=con.frame.copy().reshape(3, 3),
            force=force[:3].copy(),
            dist=con.dist,
            included=con.efc_address >= 0,
        )

    return [make_contact(i) for i in range(mj_data.ncon)]


def _update_contact_visualization(
    scene, contacts: list[_Contact], scene_offset: np.ndarray
) -> None:
    """Update contact point and force visualization."""
    contact_points: list[_ContactPointVisual] = []
    contact_forces: list[_ContactForceVisual] = []

    meansize = scene.meansize_override or scene.mj_model.stat.meansize

    for contact in contacts:
        if not contact.included:
            continue

        # Transform force from contact frame to world frame.
        force_world = contact.frame.T @ contact.force
        force_mag = np.linalg.norm(force_world)

        # Contact point visualization (cylinder).
        if scene.show_contact_points:
            contact_points.append(
                _ContactPointVisual(
                    position=contact.pos + scene_offset,
                    orientation=vtf.SO3.from_matrix(
                        rotation_matrix_from_vectors(
                            np.array([0, 0, 1]), contact.frame[0, :]
                        )
                    ).wxyz,
                    scale=np.array(
                        [
                            scene.mj_model.vis.scale.contactwidth * meansize,
                            scene.mj_model.vis.scale.contactwidth * meansize,
                            scene.mj_model.vis.scale.contactheight * meansize,
                        ]
                    ),
                )
            )

        # Contact force visualization (arrow shaft + head).
        if scene.show_contact_forces and force_mag > 1e-6:
            force_dir = force_world / force_mag
            arrow_length = (
                force_mag
                * (scene.mj_model.vis.map.force / scene.mj_model.stat.meanmass)
                if scene.mj_model.stat.meanmass > 0
                else force_mag
            )
            arrow_width = scene.mj_model.vis.scale.forcewidth * meansize
            force_quat = vtf.SO3.from_matrix(
                rotation_matrix_from_vectors(np.array([0, 0, 1]), force_dir)
            ).wxyz

            contact_forces.append(
                _ContactForceVisual(
                    shaft_position=contact.pos + scene_offset,
                    shaft_orientation=force_quat,
                    shaft_scale=np.array([arrow_width, arrow_width, arrow_length]),
                    head_position=contact.pos + scene_offset + force_dir * arrow_length,
                    head_orientation=force_quat,
                    head_scale=np.array([arrow_width, arrow_width, arrow_width]),
                )
            )

    # Update or create contact point handle.
    if contact_points:
        positions = np.array([p.position for p in contact_points], dtype=np.float32)
        orientations = np.array(
            [p.orientation for p in contact_points], dtype=np.float32
        )
        scales = np.array([p.scale for p in contact_points], dtype=np.float32)
        if scene.contact_point_handle is None:
            mesh = trimesh.creation.cylinder(radius=1.0, height=1.0)
            scene.contact_point_handle = scene.server.scene.add_batched_meshes_simple(
                "/contacts/points",
                mesh.vertices,
                mesh.faces,
                batched_wxyzs=orientations,
                batched_positions=positions,
                batched_scales=scales,
                batched_colors=np.array(scene.contact_point_color, dtype=np.uint8),
                opacity=0.8,
                lod="off",
                cast_shadow=False,
                receive_shadow=False,
            )
        scene.contact_point_handle.batched_positions = positions
        scene.contact_point_handle.batched_wxyzs = orientations
        scene.contact_point_handle.batched_scales = scales
        scene.contact_point_handle.visible = True
    elif scene.contact_point_handle is not None:
        scene.contact_point_handle.visible = False

    # Update or create contact force handles (shaft and head separately).
    if contact_forces:
        shaft_positions = np.array(
            [f.shaft_position for f in contact_forces], dtype=np.float32
        )
        shaft_orientations = np.array(
            [f.shaft_orientation for f in contact_forces], dtype=np.float32
        )
        shaft_scales = np.array(
            [f.shaft_scale for f in contact_forces], dtype=np.float32
        )
        head_positions = np.array(
            [f.head_position for f in contact_forces], dtype=np.float32
        )
        head_orientations = np.array(
            [f.head_orientation for f in contact_forces], dtype=np.float32
        )
        head_scales = np.array([f.head_scale for f in contact_forces], dtype=np.float32)
        if scene.contact_force_shaft_handle is None:
            shaft_mesh = trimesh.creation.cylinder(radius=0.4, height=1.0)
            shaft_mesh.apply_translation([0, 0, 0.5])
            scene.contact_force_shaft_handle = (
                scene.server.scene.add_batched_meshes_simple(
                    "/contacts/forces/shaft",
                    shaft_mesh.vertices,
                    shaft_mesh.faces,
                    batched_wxyzs=shaft_orientations,
                    batched_positions=shaft_positions,
                    batched_scales=shaft_scales,
                    batched_colors=np.array(scene.contact_force_color, dtype=np.uint8),
                    opacity=0.8,
                    lod="off",
                    cast_shadow=False,
                    receive_shadow=False,
                )
            )
            head_mesh = trimesh.creation.cone(radius=1.0, height=1.0, sections=8)
            scene.contact_force_head_handle = (
                scene.server.scene.add_batched_meshes_simple(
                    "/contacts/forces/head",
                    head_mesh.vertices,
                    head_mesh.faces,
                    batched_wxyzs=head_orientations,
                    batched_positions=head_positions,
                    batched_scales=head_scales,
                    batched_colors=np.array(scene.contact_force_color, dtype=np.uint8),
                    opacity=0.8,
                    lod="off",
                    cast_shadow=False,
                    receive_shadow=False,
                )
            )
        assert scene.contact_force_shaft_handle is not None
        assert scene.contact_force_head_handle is not None
        scene.contact_force_shaft_handle.batched_positions = shaft_positions
        scene.contact_force_shaft_handle.batched_wxyzs = shaft_orientations
        scene.contact_force_shaft_handle.batched_scales = shaft_scales
        scene.contact_force_shaft_handle.visible = True
        scene.contact_force_head_handle.batched_positions = head_positions
        scene.contact_force_head_handle.batched_wxyzs = head_orientations
        scene.contact_force_head_handle.batched_scales = head_scales
        scene.contact_force_head_handle.visible = True
    elif (
        scene.contact_force_shaft_handle is not None
        and scene.contact_force_head_handle is not None
    ):
        scene.contact_force_shaft_handle.visible = (
            scene.contact_force_head_handle.visible
        ) = False
