"""Scene controls."""

from __future__ import annotations

import numpy as np
import viser

_DEFAULT_FOV_DEGREES = 60
_DEFAULT_FOV_MIN = 20
_DEFAULT_FOV_MAX = 150


def create_visualization_gui(
    scene,
    camera_distance: float = 3.0,
    camera_azimuth: float = 45.0,
    camera_elevation: float = 30.0,
    show_debug_viz_control: bool = True,
    show_environment_gui: bool = True,
    show_contacts_gui: bool = True,
) -> None:
    """Add standard GUI controls that automatically update this scene's settings.

    Args:
      camera_distance: Default camera distance from tracked body when tracking is enabled.
      camera_azimuth: Default camera azimuth angle in degrees.
      camera_elevation: Default camera elevation angle in degrees.
      show_debug_viz_control: Whether to show the debug visualization checkbox.
      show_environment_gui: Whether to show the Environment folder and its controls.
      show_contacts_gui: Whether to show the Contacts folder and its controls.
    """
    with scene.server.gui.add_folder("Visualization"):
        slider_fov = scene.server.gui.add_slider(
            "FOV (°)",
            min=_DEFAULT_FOV_MIN,
            max=_DEFAULT_FOV_MAX,
            step=1,
            initial_value=_DEFAULT_FOV_DEGREES,
            hint="Vertical FOV of viewer camera, in degrees.",
        )

        @slider_fov.on_update
        def _(_) -> None:
            for client in scene.server.get_clients().values():
                client.camera.fov = np.radians(slider_fov.value)

        @scene.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.fov = np.radians(slider_fov.value)

    # Environment selection (only if multiple environments).
    if show_environment_gui:
        with scene.server.gui.add_folder("Environment"):
            # Environment selection slider (if multiple envs).
            if scene.num_envs > 1:
                env_slider = scene.server.gui.add_slider(
                    "Select",
                    min=0,
                    max=scene.num_envs - 1,
                    step=1,
                    initial_value=scene.env_idx,
                    hint=f"Select environment (0-{scene.num_envs - 1})",
                )

                @env_slider.on_update
                def _(_) -> None:
                    scene.env_idx = int(env_slider.value)
                    scene._request_update()

                show_only_cb = scene.server.gui.add_checkbox(
                    "Hide others",
                    initial_value=scene.show_only_selected,
                    hint="Show only the selected environment.",
                )

                @show_only_cb.on_update
                def _(_) -> None:
                    scene.show_only_selected = show_only_cb.value
                    scene._request_update()

            # Camera tracking controls.
            cb_camera_tracking = scene.server.gui.add_checkbox(
                "Track camera",
                initial_value=scene.camera_tracking_enabled,
                hint="Keep tracked body centered. Use Viser camera controls to adjust view.",
            )

            @cb_camera_tracking.on_update
            def _(_) -> None:
                scene.camera_tracking_enabled = cb_camera_tracking.value
                # Snap camera to default view when enabling tracking.
                if scene.camera_tracking_enabled:
                    # Convert to radians and calculate camera position.
                    azimuth_rad = np.deg2rad(camera_azimuth)
                    elevation_rad = np.deg2rad(camera_elevation)

                    # Calculate forward vector from spherical coordinates.
                    forward = np.array(
                        [
                            np.cos(elevation_rad) * np.cos(azimuth_rad),
                            np.cos(elevation_rad) * np.sin(azimuth_rad),
                            np.sin(elevation_rad),
                        ]
                    )

                    # Camera position is origin - forward * distance.
                    camera_pos = -forward * camera_distance

                    # Snap all connected clients to this view.
                    for client in scene.server.get_clients().values():
                        client.camera.position = camera_pos
                        client.camera.look_at = np.zeros(3)

                scene._request_update()

    # Debug visualization controls (only show if requested).
    if show_debug_viz_control:
        cb_debug_vis = scene.server.gui.add_checkbox(
            "Debug visualization",
            initial_value=scene.debug_visualization_enabled,
            hint="Show debug arrows and ghost meshes.",
        )

        @cb_debug_vis.on_update
        def _(_) -> None:
            scene.debug_visualization_enabled = cb_debug_vis.value
            # Clear visualizer if hiding.
            if not scene.debug_visualization_enabled:
                scene.clear_debug_all()
            scene._request_update()

    # Contact visualization settings.
    if show_contacts_gui:
        with scene.server.gui.add_folder("Contacts"):
            cb_contact_points = scene.server.gui.add_checkbox(
                "Points",
                initial_value=False,
                hint="Toggle contact point visualization.",
            )
            contact_point_color = scene.server.gui.add_rgb(
                "Points Color", initial_value=scene.contact_point_color
            )
            cb_contact_forces = scene.server.gui.add_checkbox(
                "Forces",
                initial_value=False,
                hint="Toggle contact force visualization.",
            )
            contact_force_color = scene.server.gui.add_rgb(
                "Forces Color", initial_value=scene.contact_force_color
            )
            meansize_input = scene.server.gui.add_number(
                "Scale",
                step=scene.mj_model.stat.meansize * 0.01,
                initial_value=scene.mj_model.stat.meansize,
            )

            @cb_contact_points.on_update
            def _(_) -> None:
                scene.show_contact_points = cb_contact_points.value
                scene._sync_visibilities()
                scene._request_update()

            @contact_point_color.on_update
            def _(_) -> None:
                scene.contact_point_color = contact_point_color.value
                if scene.contact_point_handle is not None:
                    scene.contact_point_handle.remove()
                    scene.contact_point_handle = None
                scene._request_update()

            @cb_contact_forces.on_update
            def _(_) -> None:
                scene.show_contact_forces = cb_contact_forces.value
                scene._sync_visibilities()
                scene._request_update()

            @contact_force_color.on_update
            def _(_) -> None:
                scene.contact_force_color = contact_force_color.value
                if scene.contact_force_shaft_handle is not None:
                    scene.contact_force_shaft_handle.remove()
                    scene.contact_force_shaft_handle = None
                if scene.contact_force_head_handle is not None:
                    scene.contact_force_head_handle.remove()
                    scene.contact_force_head_handle = None
                scene._request_update()

            @meansize_input.on_update
            def _(_) -> None:
                scene.meansize_override = meansize_input.value
                scene._request_update()


def create_geom_groups_gui(scene, tabs) -> None:
    """Add geom groups tab to the given tab group.

    Args:
      tabs: The viser tab group to add the geom groups tab to.
    """
    with tabs.add_tab("Geoms", icon=viser.Icon.EYE):
        for i in range(6):
            cb = scene.server.gui.add_checkbox(
                f"Group {i}",
                initial_value=scene.geom_groups_visible[i],
                hint=f"Show/hide geoms in group {i}",
            )

            @cb.on_update
            def _(event, group_idx=i) -> None:
                scene.geom_groups_visible[group_idx] = event.target.value
                scene._sync_visibilities()
                scene._request_update()
