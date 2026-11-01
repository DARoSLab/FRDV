"""
Standalone contact detection script.

Loads a retargeted motion file (.npy/.npz), runs each frame through MuJoCo,
detects which configured links are in contact with the ground, and writes
the result back to the file (or a new output path).

Usage:
    pixi run python tools/dynamics_processing/contact_detect.py \
        --config-name contact_detect \
        motion_path=<path>.npy \
        mujoco_xml_path=<robot>.xml \
        "contact_links=[L_foot_link,R_foot_link]" \
        plot=true
"""
import os
import numpy as np
import mujoco
import hydra
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from omegaconf import DictConfig
from scipy.ndimage import median_filter
from tqdm import tqdm

from mocap_retargeting.mujoco_integration.mujoco_interface import pin_to_mj
from mocap_retargeting.mujoco_integration.mujoco_interface import get_ground_geoms
from mocap_retargeting import PROJECT_ROOT


def get_subtree_geoms(mj_model: mujoco.MjModel, root_body_id: int) -> list:
    """Return all geom IDs belonging to root_body_id and descendants."""
    # Collect all bodies in the subtree via BFS
    subtree: list = [root_body_id]
    queue = [root_body_id]
    while queue:
        parent = queue.pop()
        for bid in range(mj_model.nbody):
            if mj_model.body_parentid[bid] == parent and bid != parent:
                subtree.append(bid)
                queue.append(bid)

    geom_ids = []
    for geom_id in range(mj_model.ngeom):
        if (
            mj_model.geom_bodyid[geom_id] in subtree
            and mj_model.geom_contype[geom_id] != 0
        ):
            geom_ids.append(geom_id)
    return geom_ids


# Suppress multi-threading conflicts from MKL/OpenMP
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def detect_link_contacts(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    q_traj: np.ndarray,
    contact_links: list,
    contact_filter_size: int = 10,
) -> np.ndarray:
    """Detect ground contacts for each link over a trajectory.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        q_traj: Joint positions (T, nq), Pinocchio convention
            (root quaternion xyzw at indices 3:7).
        contact_links: Body names to check for ground contact.
        contact_filter_size: Median-filter kernel size for debouncing.
            Set to 1 to disable.

    Returns:
        Boolean array of shape (T, len(contact_links)).
    """
    T = len(q_traj)
    num_links = len(contact_links)

    link_geoms: dict = {}
    for body_name in contact_links:
        bid = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if bid == -1:
            print(
                f"[WARNING] Link '{body_name}' not found in model — skipping."
            )
            continue
        geom_ids = get_subtree_geoms(mj_model, bid)
        if not geom_ids:
            print(
                f"[WARNING] Link '{body_name}' has no geoms — skipping."
            )
            continue
        link_geoms[body_name] = geom_ids

    if not link_geoms:
        raise ValueError("[ERROR] No valid contact links found in the model.")

    all_link_geoms: set = set()
    for gids in link_geoms.values():
        all_link_geoms.update(gids)

    ground_geoms = set(get_ground_geoms(mj_model))
    if not ground_geoms and mj_model.ngeom > 0:
        print("[WARNING] No ground geoms found — falling back to geom 0.")
        ground_geoms.add(0)

    contacts = np.zeros((T, num_links), dtype=bool)

    for t in tqdm(range(T), desc="Contact Detection"):
        q_mj = pin_to_mj(q_traj[t])
        mj_data.qpos[:] = q_mj
        mj_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        for c_idx in range(mj_data.ncon):
            con = mj_data.contact[c_idx]
            g1, g2 = int(con.geom1), int(con.geom2)

            is_g1_link = g1 in all_link_geoms
            is_g2_link = g2 in all_link_geoms
            is_g1_ground = g1 in ground_geoms
            is_g2_ground = g2 in ground_geoms

            hit_geom = -1
            if is_g1_link and is_g2_ground:
                hit_geom = g1
            elif is_g2_link and is_g1_ground:
                hit_geom = g2

            if hit_geom == -1:
                continue

            for idx, lname in enumerate(contact_links):
                if lname in link_geoms and hit_geom in link_geoms[lname]:
                    contacts[t, idx] = True
                    break

    if contact_filter_size > 1:
        print(
            f"[INFO] Applying median filter "
            f"(kernel={contact_filter_size})"
        )
        contacts = median_filter(
            contacts.astype(float), size=(contact_filter_size, 1)
        ).astype(bool)

    return contacts


def detect_mocap_contacts(
    mocap: np.ndarray,
    joint_names: list,
    contact_joints: list,
    threshold_margin: float = 0.03,
    contact_filter_size: int = 10,
) -> np.ndarray:
    """Detect ground contact from raw mocap marker heights.

    Uses a per-joint adaptive threshold: min_z + threshold_margin,
    so it works regardless of marker height offsets (heel vs toe).

    Args:
        mocap: shape (T, J, 3), z is up.
        joint_names: list of joint name strings, length J.
        contact_joints: joint names to check.
        threshold_margin: margin above each joint's min z for threshold.
        contact_filter_size: median filter kernel for debouncing.

    Returns:
        Boolean array (T, len(contact_joints)).
    """
    name_to_idx = {str(n): i for i, n in enumerate(joint_names)}
    T = len(mocap)
    contacts = np.zeros((T, len(contact_joints)), dtype=bool)

    for col, jname in enumerate(contact_joints):
        idx = name_to_idx.get(jname, -1)
        if idx == -1:
            print(f"[WARNING] Mocap joint '{jname}' not found — skipping.")
            continue
        z = mocap[:, idx, 2]
        thresh = z.min() + threshold_margin
        contacts[:, col] = z < thresh
        print(
            f"  {jname}: z_min={z.min():.4f}, "
            f"threshold={thresh:.4f}"
        )

    if contact_filter_size > 1:
        contacts = median_filter(
            contacts.astype(float), size=(contact_filter_size, 1)
        ).astype(bool)

    return contacts


def _format_link_label(name: str) -> str:
    """Normalise a link/joint name to a short LaTeX label."""
    # Strip trailing _link / _contact
    for suffix in ("_link", "_contact"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    # Normalise BVH-style names: Right/Left prefix, Foot -> heel
    name = name.replace("Right", "R_").replace("Left", "L_")
    name = name.replace("Foot", "heel").replace("Toe", "toe")
    # Collapse any double underscores introduced above
    while "__" in name:
        name = name.replace("__", "_")
    name = name.strip("_")
    return name.replace("_", r"\_")


def _draw_contact_panel(
    ax: Axes,
    contacts: np.ndarray,
    labels: list,
    t: np.ndarray,
    title: str,
    colors: list,
    show_ylabel: bool = True,
) -> None:
    for i, label in enumerate(labels):
        ax.fill_between(
            t,
            i,
            i + contacts[:, i].astype(float) * 0.8,
            alpha=0.75,
            color=colors[i],
            label=label,
            step="post",
            linewidth=0,
        )
    ax.set_yticks([i + 0.4 for i in range(len(labels))])
    ax.set_yticklabels(labels if show_ylabel else [])
    ax.set_ylim(-0.1, len(labels))
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)


def plot_contacts(
    contacts: np.ndarray,
    contact_links: list,
    fps: float,
    output_path: str,
    mocap_contacts: np.ndarray | None = None,
    mocap_joints: list | None = None,
) -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 11,
    })
    robot_labels = [_format_link_label(n) for n in contact_links]
    t = np.arange(len(contacts)) / fps
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [colors[i % len(colors)] for i in range(len(robot_labels))]

    if mocap_contacts is not None and mocap_joints is not None:
        mocap_labels = [_format_link_label(n) for n in mocap_joints]
        fig, (ax_mocap, ax_robot) = plt.subplots(
            1, 2, figsize=(14, 3), sharey=True
        )
        _draw_contact_panel(
            ax_mocap, mocap_contacts, mocap_labels, t,
            r"(a) Raw Mocap", colors
        )
        _draw_contact_panel(
            ax_robot, contacts, robot_labels, t,
            r"(b) Robot (Retargeted)", colors, show_ylabel=False
        )
        handles, labels = ax_mocap.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower right",
            ncol=len(robot_labels),
            fontsize=9,
            bbox_to_anchor=(1.0, -0.08),
        )
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(7, 3))
        _draw_contact_panel(ax, contacts, robot_labels, t, "Robot", colors)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower right",
            ncol=len(robot_labels),
            fontsize=9,
        )
        fig.tight_layout()

    plot_path = os.path.splitext(output_path)[0] + "_contacts.pdf"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    print(f"[INFO] Contact plot saved to: {plot_path}")


@hydra.main(
    version_base=None,
    config_path=os.path.join(PROJECT_ROOT, "conf"),
    config_name="contact_detect",
)
def main(cfg: DictConfig) -> None:
    motion_path: str = cfg.get("motion_path", None)
    xml_path: str = cfg.get(
        "mujoco_xml_path", cfg.get("mujoco_dst_xml_path", None)
    )
    contact_links: list = list(cfg.get("contact_links", []))
    contact_filter_size: int = int(cfg.get("contact_filter_size", 10))
    do_plot: bool = bool(cfg.get("plot", False))
    root_z_offset: float = float(cfg.get("root_z_offset", 0.0))
    start_index: int = int(cfg.get("start_index", 0))
    end_index: int | None = cfg.get("end_index", None)
    mocap_joints: list = list(cfg.get("mocap_joints", []))
    mocap_threshold: float = float(cfg.get("mocap_threshold", 0.05))

    if not motion_path or not os.path.exists(motion_path):
        raise FileNotFoundError(
            f"[ERROR] motion_path not found: {motion_path}"
        )
    if not xml_path or not os.path.exists(xml_path):
        raise FileNotFoundError(
            f"[ERROR] mujoco_xml_path not found: {xml_path}"
        )
    if not contact_links:
        raise ValueError(
            "[ERROR] contact_links must be a non-empty list of body names."
        )
    if not mocap_joints:
        raise ValueError(
            "[ERROR] mocap_joints must be set — "
            "it replaces the foot contact schedule in the saved file."
        )
    if len(mocap_joints) != len(contact_links):
        raise ValueError(
            f"[ERROR] mocap_joints ({len(mocap_joints)}) must have the same "
            f"length as contact_links ({len(contact_links)})."
        )

    print(f"[INFO] Loading motion from: {motion_path}")
    raw = np.load(motion_path, allow_pickle=True)
    if isinstance(raw, np.lib.npyio.NpzFile):
        data: dict = dict(raw)
    else:
        data = raw.item()

    q_traj: np.ndarray = data["q"].copy()
    fps: float = float(data.get("fps", 30))

    end_idx = int(end_index) if end_index is not None else len(q_traj)
    q_traj = q_traj[start_index:end_idx]
    T = len(q_traj)

    # Slice all time-series in data to match — compute n before the loop
    n_full = len(data["q"])
    for key in list(data.keys()):
        arr = data[key]
        if (
            isinstance(arr, np.ndarray)
            and arr.ndim >= 1
            and len(arr) == n_full
        ):
            data[key] = arr[start_index:end_idx]
    print(
        f"[INFO] Trajectory: T={T}, fps={fps}, "
        f"frames [{start_index}:{end_idx}] = {T/fps:.1f}s"
    )

    if root_z_offset != 0.0:
        print(f"[INFO] Applying root z offset: {root_z_offset:+.4f} m")
        q_traj[:, 2] += root_z_offset

    print(f"[INFO] Loading MuJoCo model from: {xml_path}")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    if q_traj.shape[1] != mj_model.nq:
        raise ValueError(
            f"[ERROR] q dim {q_traj.shape[1]} != model nq {mj_model.nq}. "
            "Ensure motion_path and mujoco_xml_path match."
        )

    # --- Robot contact detection (foot links only, for comparison) ---
    print(f"[INFO] Detecting robot contacts for: {contact_links}")
    robot_contacts = detect_link_contacts(
        mj_model, mj_data, q_traj, contact_links, contact_filter_size
    )
    for i, lname in enumerate(contact_links):
        frac = robot_contacts[:, i].mean() * 100
        print(f"  {lname}: in contact {frac:.1f}% of frames")

    # --- Mocap contact detection (foot links, replaces robot) ---
    raw_mocap: np.ndarray = data["mocap"][start_index:end_idx]
    joint_names: list = list(data.get("joint_names", []))
    print(f"[INFO] Detecting mocap contacts for: {mocap_joints}")
    mocap_contacts = detect_mocap_contacts(
        raw_mocap, joint_names, mocap_joints,
        threshold_margin=mocap_threshold,
        contact_filter_size=contact_filter_size,
    )
    for i, jname in enumerate(mocap_joints):
        frac = mocap_contacts[:, i].mean() * 100
        print(f"  {jname}: in contact {frac:.1f}% of frames")

    # --- Build full contacts array: all model bodies, zeros except foot links ---
    all_body_names = [
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(mj_model.nbody)
    ]
    all_contacts = np.zeros((T, len(all_body_names)), dtype=bool)
    foot_body_idx = {
        name: all_body_names.index(name)
        for name in contact_links
        if name in all_body_names
    }
    for col, lname in enumerate(contact_links):
        if lname in foot_body_idx:
            all_contacts[:, foot_body_idx[lname]] = mocap_contacts[:, col]

    data["contacts"] = all_contacts
    data["contact_link_names"] = np.array(all_body_names, dtype=object)

    _out = cfg.get("output_path", None)
    if _out is None:
        base, ext = os.path.splitext(motion_path)
        output_path: str = (
            f"{base}_contact_{start_index}_{end_idx}{ext}"
        )
    else:
        output_path = str(_out)

    print(f"[INFO] Saving to: {output_path}")
    print(
        f"[INFO] contacts shape: {all_contacts.shape} "
        f"({len(all_body_names)} bodies, "
        f"{int(all_contacts.any(axis=1).sum())} active frames)"
    )
    if output_path.endswith(".npz"):
        np.savez(output_path, **data)
    else:
        np.save(output_path, data)  # type: ignore[arg-type]

    if do_plot:
        plot_contacts(
            robot_contacts, contact_links, fps, output_path,
            mocap_contacts=mocap_contacts,
            mocap_joints=mocap_joints if mocap_joints else None,
        )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
