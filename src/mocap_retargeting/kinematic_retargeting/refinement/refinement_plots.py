"""Save refinement diagnostics without opening a display."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_analysis_plot(
    output_path,
    vel_norm,
    acc_norm,
    vel_spikes,
    acc_spikes,
    bins,
    resolved_groups,
    fps,
):
    total_frames = len(vel_norm)

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    boundary_frames = [b[0] for b in bins[1:]]
    for ax in axes:
        for i, b_frame in enumerate(boundary_frames):
            ax.axvline(
                x=b_frame,
                color="purple",
                linestyle="-",
                alpha=0.5,
                label="Chunk Boundary" if i == 0 else "",
            )

    for ax in axes:
        for gi, group in enumerate(resolved_groups):
            for bin_idx in group:
                bstart, bend = bins[bin_idx]
                ax.axvspan(
                    bstart,
                    bend,
                    color="red",
                    alpha=0.12,
                    label="Re-solved Bin" if (gi == 0 and bin_idx == group[0]) else "",
                )

    axes[0].plot(vel_norm, label="Joint Angular Vel (rad/s)", alpha=0.7)
    if len(vel_spikes) > 0:
        axes[0].scatter(
            vel_spikes,
            vel_norm[vel_spikes],
            color="red",
            s=15,
            zorder=5,
            label=f"Vel Spikes ({len(vel_spikes)})",
        )
    axes[0].set_title("Velocity Norms")
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.7)
    axes[0].legend()

    axes[1].plot(acc_norm, label="Joint Angular Acc (rad/s^2)", alpha=0.7)
    if len(acc_spikes) > 0:
        axes[1].scatter(
            acc_spikes,
            acc_norm[acc_spikes],
            color="red",
            s=15,
            zorder=5,
            label=f"Acc Spikes ({len(acc_spikes)})",
        )
    axes[1].set_title("Acceleration Norms")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_xlabel("Frame")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.7)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Analysis plot saved to: {output_path}")
