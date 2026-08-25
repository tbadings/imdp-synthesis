from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np


# 2D plots of trajectories
def plot_rl_trajectories(base_model, eval_env, trajectories, dims, output_dir, max_trajectories=100):
    if len(dims) != 2:
        raise ValueError("This runner currently supports plotting exactly 2 dimensions.")

    d0, d1 = dims
    fig, ax = plt.subplots(figsize=(8, 8))
    legend_handles = []

    # Plot regions (critical, goal, charging station)
    regions = [
        (getattr(eval_env, "critical", None), "red", 0.25, "Critical"),
        (getattr(eval_env, "goal", None), "green", 0.25, "Goal"),
        (getattr(eval_env, "charging_station", None), "blue", 0.25, "Charging station"),
    ]
    for boxes, color, alpha, label in regions:
        if boxes is not None and boxes.size > 0:
            rects = [
                mpatches.Rectangle(
                    (b[0, d0], b[0, d1]),
                    b[1, d0] - b[0, d0],
                    b[1, d1] - b[0, d1],
                )
                for b in boxes
            ]
            ax.add_collection(PatchCollection(rects, facecolor=color, edgecolor="none", alpha=alpha, rasterized=True))
            legend_handles.append(mpatches.Patch(color=color, alpha=alpha, label=label))

    # Plot trajectory traces
    selected_traces = [t[:, dims] for t in (trajectories or [])[:max_trajectories] if len(t) > 0]
    if selected_traces:
        nan_sep = np.full((1, 2), np.nan, dtype=np.float32)
        combined = np.concatenate([np.vstack([t, nan_sep]) for t in selected_traces])
        ax.plot(
            combined[:, 0],
            combined[:, 1],
            linewidth=1.0,
            alpha=0.9,
            color="black",
            marker=".",
            markersize=3.0,
            markeredgecolor="red",
            markerfacecolor="red",
            rasterized=True,
        )

    ax.set(
        xlim=(eval_env.obs_low[d0], eval_env.obs_high[d0]),
        ylim=(eval_env.obs_low[d1], eval_env.obs_high[d1]),
        xlabel=base_model.state_variables[d0],
        ylabel=base_model.state_variables[d1],
        title=f"PPO trajectories ({base_model.__class__.__name__})",
    )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    fig.tight_layout()

    # Save plots
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"rl_trajectories.{ext}", format=ext, bbox_inches="tight", dpi=200)
    plt.close(fig)
