"""Debug visualization."""

from __future__ import annotations

import mujoco
import numpy as np
import trimesh
import viser.transforms as vtf
from mujoco import mjtGeom

from .mesh_conversion import (
    create_primitive_mesh,
    get_body_name,
    mujoco_mesh_to_trimesh,
    rotation_quat_from_vectors,
)


def add_arrow(
    scene,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[float, float, float, float],
    width: float = 0.015,
    label: str | None = None,
) -> None:
    """Queue an arrow for batched rendering.

    Arrows are not rendered immediately but queued and rendered together
    in the next update() call for efficiency.
    """
    if not scene.debug_visualization_enabled:
        return

    del label  # Unused.

    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-6:
        return

    # Queue the arrow for batched rendering (without scene offset - applied during sync)
    scene._queued_arrows.append((start, end, color, width))


def add_ghost_mesh(
    scene,
    qpos: np.ndarray,
    model: mujoco.MjModel,
    alpha: float = 0.5,
    label: str | None = None,
) -> None:
    """Add a ghost mesh by rendering the robot at a different pose.

    For Viser, we create meshes once and update their poses for efficiency.

    Args:
      qpos: Joint positions for the ghost pose
      model: MuJoCo model with pre-configured appearance (geom_rgba for colors)
      alpha: Transparency override
      label: Optional label for this ghost
    """
    if not scene.debug_visualization_enabled:
        return

    # Use model hash to support models with same structure but different colors
    model_hash = hash((model.ngeom, model.nbody, model.nq))

    scene._viz_data.qpos[:] = qpos
    mujoco.mj_forward(model, scene._viz_data)

    # Use current scene offset
    scene_offset = scene._scene_offset

    # Group geoms by body
    body_geoms: dict[int, list[int]] = {}
    for i in range(model.ngeom):
        body_id = model.geom_bodyid[i]
        is_collision = model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0
        if is_collision:
            continue

        if model.body_dofnum[body_id] == 0 and model.body_parentid[body_id] == 0:
            continue

        if body_id not in body_geoms:
            body_geoms[body_id] = []
        body_geoms[body_id].append(i)

    # Update or create mesh for each body
    for body_id, geom_indices in body_geoms.items():
        body_pos = scene._viz_data.xpos[body_id] + scene_offset
        body_quat = scene._mat_to_quat(scene._viz_data.xmat[body_id].reshape(3, 3))

        # Check if we already have a handle for this body
        if body_id in scene._ghost_handles:
            handle = scene._ghost_handles[body_id]
            handle.wxyz = body_quat
            handle.position = body_pos
        else:
            # Create mesh if not cached
            if model_hash not in scene._ghost_meshes:
                scene._ghost_meshes[model_hash] = {}

            if body_id not in scene._ghost_meshes[model_hash]:
                meshes = []
                for geom_id in geom_indices:
                    mesh = scene._create_geom_mesh_from_model(model, geom_id)
                    if mesh is not None:
                        geom_pos = model.geom_pos[geom_id]
                        geom_quat = model.geom_quat[geom_id]
                        transform = np.eye(4)
                        transform[:3, :3] = vtf.SO3(geom_quat).as_matrix()
                        transform[:3, 3] = geom_pos
                        mesh.apply_transform(transform)
                        meshes.append(mesh)

                if not meshes:
                    continue

                combined_mesh = (
                    meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
                )

                scene._ghost_meshes[model_hash][body_id] = combined_mesh
            else:
                combined_mesh = scene._ghost_meshes[model_hash][body_id]

            body_name = get_body_name(model, body_id)
            handle_name = f"/debug/env_{scene.env_idx}/ghost/body_{body_name}"

            # Extract color from geom (convert RGBA 0-1 to RGB 0-255)
            rgba = model.geom_rgba[geom_indices[0]].copy()
            color_uint8 = (rgba[:3] * 255).astype(np.uint8)

            handle = scene.server.scene.add_mesh_simple(
                handle_name,
                combined_mesh.vertices,
                combined_mesh.faces,
                color=tuple(color_uint8),
                opacity=alpha,
                wxyz=body_quat,
                position=body_pos,
                cast_shadow=False,
                receive_shadow=False,
            )
            scene._ghost_handles[body_id] = handle


def add_frame(
    scene,
    position: np.ndarray,
    rotation_matrix: np.ndarray,
    scale: float = 0.3,
    label: str | None = None,
    axis_radius: float = 0.01,
    alpha: float = 1.0,
    axis_colors: tuple[tuple[float, float, float], ...] | None = None,
) -> None:
    """Add a coordinate frame visualization with RGB-colored axes.

    This implementation reuses add_arrow to draw the three axis arrows.

    Args:
      position: Position of the frame origin (3D vector)
      rotation_matrix: Rotation matrix (3x3)
      scale: Scale/length of the axis arrows
      label: Optional label for this frame.
      axis_radius: Radius of the axis arrows.
      alpha: Opacity for all axes (0=transparent, 1=opaque). Note: This implementation
        does not support per-arrow transparency. All arrows in the scene will share
        the same alpha value.
      axis_colors: Optional tuple of 3 RGB colors for X, Y, Z axes. If None, uses
        default RGB coloring (X=red, Y=green, Z=blue).
    """
    if not scene.debug_visualization_enabled:
        return

    del label  # Unused.

    default_colors = [(0.9, 0, 0), (0, 0.9, 0.0), (0.0, 0.0, 0.9)]
    colors = axis_colors if axis_colors is not None else default_colors

    for axis_idx in range(3):
        axis_direction = rotation_matrix[:, axis_idx]
        end_position = position + axis_direction * scale
        rgb = colors[axis_idx]
        color_rgba = (rgb[0], rgb[1], rgb[2], alpha)
        scene.add_arrow(
            start=position,
            end=end_position,
            color=color_rgba,
            width=axis_radius,
        )


def clear_debug_all(scene) -> None:
    """Clear all debug visualizations including ghosts.

    Called when switching to a different environment or disabling debug visualization.
    """
    scene.clear()

    # Remove arrow meshes
    if scene._arrow_shaft_handle is not None:
        scene._arrow_shaft_handle.remove()
        scene._arrow_shaft_handle = None
    if scene._arrow_head_handle is not None:
        scene._arrow_head_handle.remove()
        scene._arrow_head_handle = None

    # Remove ghost meshes
    for handle in scene._ghost_handles.values():
        handle.remove()
    scene._ghost_handles.clear()


def _create_geom_mesh_from_model(
    scene, mj_model: mujoco.MjModel, geom_id: int
) -> trimesh.Trimesh | None:
    """Create a trimesh from a MuJoCo geom using the specified model.

    Args:
      mj_model: MuJoCo model containing geom definition
      geom_id: Index of the geom to create mesh for

    Returns:
      Trimesh representation of the geom, or None if unsupported type
    """
    geom_type = mj_model.geom_type[geom_id]

    if geom_type == mjtGeom.mjGEOM_MESH:
        return mujoco_mesh_to_trimesh(mj_model, geom_id, verbose=False)
    else:
        return create_primitive_mesh(mj_model, geom_id)


def _sync_arrows(scene) -> None:
    """Render all queued arrows using batched meshes.

    This should be called after all debug visualizations have been queued
    for the current frame.
    """
    if not scene.debug_visualization_enabled:
        return

    if not scene._queued_arrows:
        # Remove arrow meshes if no arrows to render
        if scene._arrow_shaft_handle is not None:
            scene._arrow_shaft_handle.remove()
            scene._arrow_shaft_handle = None
        if scene._arrow_head_handle is not None:
            scene._arrow_head_handle.remove()
            scene._arrow_head_handle = None
        return

    # Create arrow mesh components if needed (unit-sized base meshes)
    if scene._arrow_shaft_mesh is None:
        # Unit cylinder: radius=1.0, height=1.0
        scene._arrow_shaft_mesh = trimesh.creation.cylinder(radius=1.0, height=1.0)
        scene._arrow_shaft_mesh.apply_translation(
            np.array([0, 0, 0.5])
        )  # Center at z=0.5

    if scene._arrow_head_mesh is None:
        # Unit cone: radius=2.0, height=1.0 (base at z=0, tip at z=1.0 by default)
        head_width = 2.0
        scene._arrow_head_mesh = trimesh.creation.cone(radius=head_width, height=1.0)
        # No translation needed - cone already has base at z=0

    # Prepare batched data
    num_arrows = len(scene._queued_arrows)
    shaft_positions = np.zeros((num_arrows, 3), dtype=np.float32)
    shaft_wxyzs = np.zeros((num_arrows, 4), dtype=np.float32)
    shaft_scales = np.zeros((num_arrows, 3), dtype=np.float32)
    shaft_colors = np.zeros((num_arrows, 3), dtype=np.uint8)

    head_positions = np.zeros((num_arrows, 3), dtype=np.float32)
    head_wxyzs = np.zeros((num_arrows, 4), dtype=np.float32)
    head_scales = np.zeros((num_arrows, 3), dtype=np.float32)
    head_colors = np.zeros((num_arrows, 3), dtype=np.uint8)

    z_axis = np.array([0, 0, 1])
    shaft_length_ratio = 0.8
    head_length_ratio = 0.2

    # Apply scene offset to all arrows
    for i, (start, end, color, width) in enumerate(scene._queued_arrows):
        # Apply scene offset
        start_offset = start + scene._scene_offset
        end_offset = end + scene._scene_offset

        direction = end_offset - start_offset
        length = np.linalg.norm(direction)
        direction = direction / length

        rotation_quat = rotation_quat_from_vectors(z_axis, direction)

        # Shaft: scale width in XY, length in Z
        shaft_length = shaft_length_ratio * length
        shaft_positions[i] = start_offset
        shaft_wxyzs[i] = rotation_quat
        shaft_scales[i] = [width, width, shaft_length]  # Per-axis scale
        shaft_colors[i] = (np.array(color[:3]) * 255).astype(np.uint8)

        # Head: position at end of shaft
        # The cone has its base at z=0, so after scaling by head_length,
        # the base is still at z=0 in local coords
        # We want the base at the end of the shaft (at shaft_length)
        head_length = head_length_ratio * length
        head_position = start_offset + direction * shaft_length
        head_positions[i] = head_position
        head_wxyzs[i] = rotation_quat
        head_scales[i] = [width, width, head_length]  # Per-axis scale
        head_colors[i] = (np.array(color[:3]) * 255).astype(np.uint8)

    # Check if we need to recreate handles (number of arrows changed)
    needs_recreation = (
        scene._arrow_shaft_handle is None
        or scene._arrow_head_handle is None
        or len(shaft_positions) != len(scene._arrow_shaft_handle.batched_positions)
    )

    if needs_recreation:
        # Remove old handles
        if scene._arrow_shaft_handle is not None:
            scene._arrow_shaft_handle.remove()
        if scene._arrow_head_handle is not None:
            scene._arrow_head_handle.remove()

        # Create new batched meshes
        scene._arrow_shaft_handle = scene.server.scene.add_batched_meshes_simple(
            f"/debug/env_{scene.env_idx}/arrow_shafts",
            scene._arrow_shaft_mesh.vertices,
            scene._arrow_shaft_mesh.faces,
            batched_wxyzs=shaft_wxyzs,
            batched_positions=shaft_positions,
            batched_scales=shaft_scales,
            batched_colors=shaft_colors,
            cast_shadow=False,
            receive_shadow=False,
        )

        scene._arrow_head_handle = scene.server.scene.add_batched_meshes_simple(
            f"/debug/env_{scene.env_idx}/arrow_heads",
            scene._arrow_head_mesh.vertices,
            scene._arrow_head_mesh.faces,
            batched_wxyzs=head_wxyzs,
            batched_positions=head_positions,
            batched_scales=head_scales,
            batched_colors=head_colors,
            cast_shadow=False,
            receive_shadow=False,
        )
    else:
        # Update existing handles (guaranteed to exist by needs_recreation check)
        assert scene._arrow_shaft_handle is not None
        assert scene._arrow_head_handle is not None

        scene._arrow_shaft_handle.batched_positions = shaft_positions
        scene._arrow_shaft_handle.batched_wxyzs = shaft_wxyzs
        scene._arrow_shaft_handle.batched_scales = shaft_scales
        scene._arrow_shaft_handle.batched_colors = shaft_colors

        scene._arrow_head_handle.batched_positions = head_positions
        scene._arrow_head_handle.batched_wxyzs = head_wxyzs
        scene._arrow_head_handle.batched_scales = head_scales
        scene._arrow_head_handle.batched_colors = head_colors

    # Clear queue after rendering
    scene._queued_arrows.clear()
