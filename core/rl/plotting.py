from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from core.plotting.utils import (
    col, style_axes, save_fig, set_plot_ticks,
    GOAL_COLOR, CRITICAL_COLOR, GOAL_HATCH, CRITICAL_HATCH, START_COLOR, END_COLOR
)
from core.plotting.traces import _format_state_label_math

def plot_rl_trajectories(base_model, eval_env, trajectories, dims, output_dir, max_trajectories=100, algo_name=None):
    # Plot 2D RL rollouts
    if len(dims) != 2:
        raise ValueError("Requires 2 dimensions.")
    d0, d1 = dims
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    legend_handles = []

    # Goal & unsafe regions
    regions = [(getattr(eval_env, "critical", None), CRITICAL_COLOR, 'darkred', 'Critical', CRITICAL_HATCH),
               (getattr(eval_env, "goal", None), GOAL_COLOR, 'darkgreen', 'Goal', GOAL_HATCH)]
    for boxes, c, ec, lbl, hatch in regions:
        if boxes is not None and boxes.size > 0:
            rects = [mpatches.Rectangle((b[0, d0], b[0, d1]), b[1, d0] - b[0, d0], b[1, d1] - b[0, d1]) for b in boxes]
            ax.add_collection(PatchCollection(rects, facecolor=col(c), edgecolor=col(ec), lw=0, hatch=hatch, alpha=0.4, rasterized=True))
            legend_handles.append(mpatches.Patch(facecolor=col(c), edgecolor=col(ec), lw=0, hatch=hatch, alpha=0.4, label=lbl))
    for s in getattr(base_model, 'charging_station', []):
        ax.add_patch(mpatches.Rectangle(s[0][[d0, d1]], *(s[1] - s[0])[[d0, d1]], facecolor=col('limegreen'), alpha=0.4, lw=1))
        ax.text(*(s[0] + s[1])[[d0, d1]] / 2, '⚡', fontsize=40, ha='center', va='center', color=col('darkgreen'))
        legend_handles.append(mpatches.Patch(facecolor=col('limegreen'), alpha=0.4, label='Charging'))

    # RL trajectories
    for trace in (trajectories or [])[:max_trajectories]:
        if len(trace) > 1:
            t = trace[:, dims]
            ax.plot(t[:, 0], t[:, 1], '-o', color=col('black'), lw=1, markersize=1.5, alpha=0.4, markeredgewidth=0, rasterized=True)
            ax.plot(t[0, 0], t[0, 1], 's', color=col(START_COLOR), markersize=2, markeredgewidth=0, zorder=7)
            ax.plot(t[-1, 0], t[-1, 1], 'o', color=col(END_COLOR), markersize=2, alpha=1, markeredgewidth=0, zorder=6)

    # Style axes
    style_axes(ax)
    set_plot_ticks(ax)
    ax.set_xlim(eval_env.obs_low[d0], eval_env.obs_high[d0])
    ax.set_ylim(eval_env.obs_low[d1], eval_env.obs_high[d1])
    ax.set_box_aspect(1)
    ax.set_xlabel(_format_state_label_math(base_model.state_variables[d0]), fontsize=18, labelpad=10)
    ax.set_ylabel(_format_state_label_math(base_model.state_variables[d1]), fontsize=18, labelpad=10)
    ax.set_title(f"{str(algo_name or 'RL').upper()} Trajectories ({base_model.__class__.__name__})", fontsize=18, pad=12)

    # Legend & layout
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", frameon=True, facecolor='white', framealpha=1, edgecolor=col('lightgray'), fontsize=18)
    fig.tight_layout()
    save_fig(fig, Path(output_dir) / 'rl_trajectories')
