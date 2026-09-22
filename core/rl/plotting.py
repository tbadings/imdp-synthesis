from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import numpy as np
from core.plotting.utils import (
    col, style_axes, save_fig, set_plot_ticks,
    GOAL_COLOR, CRITICAL_COLOR, GOAL_HATCH, CRITICAL_HATCH, START_COLOR, END_COLOR
)
from core.plotting.traces import _format_state_label_math


ACTIVE_COLOR = "cornflowerblue"
VISITED_COLOR = "royalblue"


def _project_cells(cells, dims):
    """Return the unique 2-D grid cells visible in a state-space projection."""
    cells_array = np.asarray(list(cells) if isinstance(cells, set) else cells, dtype=int)
    if cells_array.size == 0:
        return np.empty((0, 2), dtype=int)
    cells_array = np.atleast_2d(cells_array)
    if cells_array.shape[1] <= max(dims):
        raise ValueError("State cells do not contain the requested plot dimensions.")
    return np.unique(cells_array[:, dims], axis=0)


def _cell_collection(cells, eval_env, dims, **kwargs):
    """Create rectangles for grid-indexed cells projected onto two dimensions."""
    d0, d1 = dims
    projected = _project_cells(cells, dims)
    rects = [
        mpatches.Rectangle(
            (
                eval_env.obs_low[d0] + cell[0] * eval_env.bin_widths[d0],
                eval_env.obs_low[d1] + cell[1] * eval_env.bin_widths[d1],
            ),
            eval_env.bin_widths[d0],
            eval_env.bin_widths[d1],
        )
        for cell in projected
    ]
    return PatchCollection(rects, **kwargs)


def _trajectory_cells(trajectories, eval_env, max_trajectories):
    """Return cells visited by the trajectories that will actually be plotted."""
    visited = []
    for trace in (trajectories or [])[:max_trajectories]:
        trace = np.asarray(trace)
        if len(trace) < 2:
            continue
        in_bounds = np.all(
            (trace >= eval_env.obs_low) & (trace <= eval_env.obs_high),
            axis=1,
        )
        bounded_trace = trace[in_bounds]
        if len(bounded_trace) == 0:
            continue
        cells = np.floor(
            (bounded_trace - eval_env.obs_low) / eval_env.bin_widths
        ).astype(int)
        visited.append(np.clip(cells, 0, eval_env.number_per_dim - 1))

    if not visited:
        return np.empty((0, len(eval_env.obs_low)), dtype=int)
    return np.unique(np.vstack(visited), axis=0)


def _plot_partition_grid(ax, eval_env, dims):
    """Draw every cell boundary in the selected 2-D partition projection."""
    d0, d1 = dims
    x_edges = np.linspace(
        eval_env.obs_low[d0], eval_env.obs_high[d0], int(eval_env.number_per_dim[d0]) + 1
    )
    y_edges = np.linspace(
        eval_env.obs_low[d1], eval_env.obs_high[d1], int(eval_env.number_per_dim[d1]) + 1
    )
    segments = [
        ((x, eval_env.obs_low[d1]), (x, eval_env.obs_high[d1])) for x in x_edges
    ] + [
        ((eval_env.obs_low[d0], y), (eval_env.obs_high[d0], y)) for y in y_edges
    ]
    ax.add_collection(
        LineCollection(
            segments,
            colors=col("lightgray"),
            linewidths=0.35,
            linestyles=":",
            alpha=0.8,
            zorder=2,
        )
    )


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


def plot_rl_trajectories_with_active_states(
    base_model,
    eval_env,
    trajectories,
    active_states,
    dims,
    output_dir,
    max_trajectories=100,
    algo_name=None,
):
    """Plot RL rollouts over the complete partition and the resulting active tube.

    In state spaces with more than two dimensions, a cell is highlighted when its
    projection onto ``dims`` is active. Directly visited cells are drawn more
    strongly than cells added later by tube construction.
    """
    if len(dims) != 2:
        raise ValueError("Requires 2 dimensions.")

    d0, d1 = dims
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    legend_handles = []

    # Keep the active tube subtle so the partition and trajectories stay legible.
    ax.add_collection(
        _cell_collection(
            active_states,
            eval_env,
            dims,
            facecolor=col(ACTIVE_COLOR),
            edgecolor="none",
            alpha=0.23,
            rasterized=True,
            zorder=1,
        )
    )
    legend_handles.append(
        mpatches.Patch(facecolor=col(ACTIVE_COLOR), edgecolor="none", alpha=0.23, label="Active states")
    )

    # Derive this layer from the same trajectory subset drawn below. The tube is
    # based on all evaluation episodes, but only visible paths should get this fill.
    plotted_visited_states = _trajectory_cells(trajectories, eval_env, max_trajectories)
    ax.add_collection(
        _cell_collection(
            plotted_visited_states,
            eval_env,
            dims,
            facecolor=col(VISITED_COLOR),
            edgecolor="none",
            alpha=0.48,
            rasterized=True,
            zorder=1.5,
        )
    )
    legend_handles.append(
        mpatches.Patch(
            facecolor=col(VISITED_COLOR),
            edgecolor="none",
            alpha=0.48,
            label="Visited states",
        )
    )

    # Draw all remaining partition cells as a light grid.
    _plot_partition_grid(ax, eval_env, dims)

    # Goal, unsafe, and optional charging regions.
    regions = [
        (getattr(eval_env, "critical", None), CRITICAL_COLOR, "darkred", "Critical", CRITICAL_HATCH),
        (getattr(eval_env, "goal", None), GOAL_COLOR, "darkgreen", "Goal", GOAL_HATCH),
    ]
    for boxes, color, edgecolor, label, hatch in regions:
        if boxes is not None and boxes.size > 0:
            rects = [
                mpatches.Rectangle(
                    (box[0, d0], box[0, d1]),
                    box[1, d0] - box[0, d0],
                    box[1, d1] - box[0, d1],
                )
                for box in boxes
            ]
            ax.add_collection(
                PatchCollection(
                    rects,
                    facecolor=col(color),
                    edgecolor=col(edgecolor),
                    lw=0,
                    hatch=hatch,
                    alpha=0.4,
                    rasterized=True,
                    zorder=3,
                )
            )
            legend_handles.append(
                mpatches.Patch(
                    facecolor=col(color),
                    edgecolor=col(edgecolor),
                    lw=0,
                    hatch=hatch,
                    alpha=0.4,
                    label=label,
                )
            )

    for station in getattr(base_model, "charging_station", []):
        ax.add_patch(
            mpatches.Rectangle(
                station[0][[d0, d1]],
                *(station[1] - station[0])[[d0, d1]],
                facecolor=col("limegreen"),
                alpha=0.4,
                lw=1,
                zorder=3,
            )
        )
        ax.text(
            *(station[0] + station[1])[[d0, d1]] / 2,
            "⚡",
            fontsize=40,
            ha="center",
            va="center",
            color=col("darkgreen"),
            zorder=4,
        )
        legend_handles.append(
            mpatches.Patch(facecolor=col("limegreen"), alpha=0.4, label="Charging")
        )

    # Render all paths as one transparent artist. If each trajectory were a
    # separate artist, their alpha values would compound wherever paths overlap.
    projected_trajectories = [
        np.asarray(trace)[:, dims]
        for trace in (trajectories or [])[:max_trajectories]
        if len(trace) > 1
    ]
    if projected_trajectories:
        separator = np.full((1, 2), np.nan)
        combined = np.concatenate(
            [np.vstack((trace, separator)) for trace in projected_trajectories]
        )
        ax.plot(
            combined[:, 0], combined[:, 1], "-o", color=col("black"), lw=0.45,
            markersize=1.5, alpha=0.3, markeredgewidth=0, rasterized=True, zorder=5,
        )
        starts = np.asarray([trace[0] for trace in projected_trajectories])
        ends = np.asarray([trace[-1] for trace in projected_trajectories])
        ax.scatter(
            starts[:, 0], starts[:, 1], marker="s", color=col(START_COLOR),
            s=4, alpha=0.45, edgecolors="none", zorder=7,
        )
        ax.scatter(
            ends[:, 0], ends[:, 1], marker="o", color=col(END_COLOR),
            s=4, alpha=0.45, edgecolors="none", zorder=6,
        )

    style_axes(ax)
    set_plot_ticks(ax)
    ax.tick_params(labelsize=14)
    ax.set_xlim(eval_env.obs_low[d0], eval_env.obs_high[d0])
    ax.set_ylim(eval_env.obs_low[d1], eval_env.obs_high[d1])
    ax.set_box_aspect(1)
    ax.set_xlabel(_format_state_label_math(base_model.state_variables[d0]), fontsize=16, labelpad=7)
    ax.set_ylabel(_format_state_label_math(base_model.state_variables[d1]), fontsize=16, labelpad=7)
    legend = ax.legend(
        handles=legend_handles[:2],
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=1,
        edgecolor=col("lightgray"),
        fontsize=13,
        ncol=1,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.4,
        handletextpad=0.5,
        columnspacing=0.8,
    )
    legend.set_zorder(10)
    fig.tight_layout()
    save_fig(fig, Path(output_dir) / "rl_trajectories_active_states")
