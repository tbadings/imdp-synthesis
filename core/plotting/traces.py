from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import CubicSpline

from core.plotting.utils import (
    col, style_axes, save_fig, set_plot_ticks, set_plot_lims, plot_boxes, plot_grid, plot_missing_cells,
    GOAL_COLOR, CRITICAL_COLOR, CRITICAL_HATCH, UNACTIVE_COLOR, START_COLOR, END_COLOR
)

def _format_state_label_math(var_name):
    # LaTeX format state name
    if not var_name:
        return ""
    labels = {'position': 'Position', 'velocity': 'Velocity', 'angle': 'Angle (rad)', 'angular_velocity': 'Angular Velocity (rad/s)'}
    if var_name in labels:
        return labels[var_name]
    if '_' in var_name:
        h, t = var_name.split('_', 1)
        return f'{h.capitalize()} ({t})' if len(h) > 1 else f'${h}_{{{t}}}$'
    return f'${var_name}$' if len(var_name) <= 2 else var_name.capitalize()

def _cuboid_faces(low, high):
    # Vertices of 6 cuboid faces
    x0, y0, z0 = low
    x1, y1, z1 = high
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]

def _plot_cuboid(ax, low, high, facecolor, alpha=0.4):
    # 3D cuboid patch
    ax.add_collection3d(Poly3DCollection(_cuboid_faces(low, high), facecolors=col(facecolor), edgecolors='none', lw=0, alpha=alpha))

def plot_traces_3d(args, stamp, idx_show, partition, model, traces, num_traces=100, filename="traces_3d", show_ticks=None, camera_angles=None):
    # 3D state trajectories
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    i1, i2, i3 = np.array(idx_show, dtype=int)
    state_lb = np.array(partition.boundary_lb)[[i1, i2, i3]]
    state_ub = np.array(partition.boundary_ub)[[i1, i2, i3]]

    # Target & unsafe regions
    for s in model.goal:
        _plot_cuboid(ax, np.array(s[0])[[i1, i2, i3]], np.array(s[1])[[i1, i2, i3]], GOAL_COLOR, alpha=0.4)
    for s in model.critical:
        _plot_cuboid(ax, np.array(s[0])[[i1, i2, i3]], np.array(s[1])[[i1, i2, i3]], CRITICAL_COLOR, alpha=0.4)

    # Simulation traces
    for trace in traces.values():
        t = np.array(trace['x'])[:, [i1, i2, i3]]
        if len(t) > 1:
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color=col('black'), alpha=0.4, lw=1)
            ax.scatter(t[:, 0], t[:, 1], t[:, 2], color=col('black'), s=2, alpha=0.4, depthshade=False)
            ax.scatter(t[0, 0], t[0, 1], t[0, 2], marker='s', color=col(START_COLOR), s=6, edgecolors='none', lw=0, alpha=1)
            ax.scatter(t[-1, 0], t[-1, 1], t[-1, 2], marker='o', color=col(END_COLOR), s=6, edgecolors='none', lw=0, alpha=1)

    # Limits and aspect
    ax.set_xlim(state_lb[0], state_ub[0])
    ax.set_ylim(state_lb[1], state_ub[1])
    ax.set_zlim(state_lb[2], state_ub[2])
    try:
        ax.set_box_aspect((state_ub - state_lb).tolist())
    except Exception:
        pass

    # 3D grid and ticks
    widths, lowers, uppers = np.array(partition.cell_width)[[i1, i2, i3]], state_lb, state_ub
    fmt = FuncFormatter(lambda v, p: f'{round(v, 1):g}')
    for axis, (l, u, w) in zip((ax.xaxis, ax.yaxis, ax.zaxis), zip(lowers, uppers, widths)):
        axis.set_ticks(np.arange(l, u + w, w))
        axis.set_major_formatter(fmt)
    ax.tick_params(labelsize=0, direction='in', length=3)
    ax.grid(True, ls=':', lw=0.5, alpha=0.4)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(col('lightgray'))

    # Views and saving
    if args.plot_title:
        ax.set_title(f"Simulation for {args.model}", fontsize=18, pad=12)
    output_dir = Path(getattr(args, 'output_dir', 'output'))
    for vi, (elev, azim) in enumerate(camera_angles or [(60, 15)]):
        ax.view_init(elev=elev, azim=azim)
        fig.tight_layout()
        save_fig(fig, output_dir / f'{filename}_{stamp}_view{vi}_{stamp}')

def plot_traces(args, stamp, idx_show, partition, model, traces, line=True, num_traces=100, add_unsafe_box=True, filename="traces", show_ticks=None):
    # 2D state trajectories
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    i1, i2 = np.array(idx_show, dtype=int)
    ax.set_xlabel(_format_state_label_math(model.state_variables[i1]), fontsize=18, labelpad=10)
    ax.set_ylabel(_format_state_label_math(model.state_variables[i2]), fontsize=18, labelpad=10)

    # Ticks & limits
    expand = 1 if add_unsafe_box else 0
    _show_ticks = args.plot_ticks if show_ticks is None else show_ticks
    if _show_ticks:
        set_plot_ticks(ax)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    set_plot_lims(ax, np.array(partition.boundary_lb)[[i1, i2]] - expand, np.array(partition.boundary_ub)[[i1, i2]] + expand)
    ax.set_aspect('equal', adjustable='box')

    if args.plot_grid:
        plot_grid(ax, np.array(partition.boundary_lb)[[i1, i2]], np.array(partition.boundary_ub)[[i1, i2]])

    # Target & unsafe boxes
    plot_boxes(ax, model, plot_dimensions=[i1, i2])

    # Inactive missing cells
    idxs = np.asarray(partition.region_idx_inv, dtype=int)
    existing = {tuple(v) for v in idxs[:, [i1, i2]]}
    w1, w2 = float(partition.cell_width[i1]), float(partition.cell_width[i2])
    x0, y0 = float(partition.boundary_lb[i1]), float(partition.boundary_lb[i2])
    mc = [(ix, iy) for ix in range(partition.number_per_dim[i1]) for iy in range(partition.number_per_dim[i2]) if (ix, iy) not in existing]
    if mc:
        mc = np.array(mc)
        plot_missing_cells(ax, x0 + mc[:, 0] * w1, y0 + mc[:, 1] * w2, w1, w2, zorder=4)

    # Unsafe boundary box
    if add_unsafe_box:
        lb, ub = np.array(partition.boundary_lb), np.array(partition.boundary_ub)
        lows = [np.array([lb[i1] - expand, lb[i2] - expand]), np.array([lb[i1], lb[i2] - expand]),
                np.array([lb[i1], ub[i2]]), np.array([ub[i1], lb[i2] - expand])]
        highs = [np.array([lb[i1], ub[i2] + expand]), np.array([ub[i1], lb[i2]]),
                 np.array([ub[i1], ub[i2] + expand]), np.array([ub[i1] + expand, ub[i2] + expand])]
        for low, high in zip(lows, highs):
            ax.add_patch(Rectangle(low, *(high - low), facecolor=col(CRITICAL_COLOR), alpha=0.4, edgecolor=col('darkred'), lw=0, hatch=CRITICAL_HATCH))

    # Style axes
    style_axes(ax)

    # Trajectories
    for i, trace in enumerate(traces.values()):
        if i >= num_traces:
            break
        t = np.array(trace['x'])[:, [i1, i2]]
        if len(t) < 2:
            continue
        if line:
            dist = np.insert(np.cumsum(np.hypot(*np.diff(t, axis=0).T)), 0, 0)
            pts = CubicSpline(dist / dist[-1], t, bc_type='natural')(np.linspace(0, 1, 75))
            ax.plot(*pts.T, '-', color=col('black'), lw=1, alpha=0.4, zorder=4)
            ax.plot(t[:, 0], t[:, 1], 'o', color=col('black'), markersize=1.5, alpha=0.4, markeredgewidth=0, zorder=5)
        else:
            ax.plot(t[:, 0], t[:, 1], '-o', color=col('black'), lw=1, markersize=1.5, alpha=0.4, markeredgewidth=0, rasterized=True, zorder=4)
        ax.plot(t[0, 0], t[0, 1], 's', color=col(START_COLOR), markersize=2, markeredgewidth=0, zorder=7)
        ax.plot(t[-1, 0], t[-1, 1], 'o', color=col(END_COLOR), markersize=2, alpha=1, markeredgewidth=0, zorder=6)

    # Title & save
    if args.plot_title:
        ax.set_title(f"Simulation for {args.model}", fontsize=18, pad=12)
    fig.tight_layout()
    save_fig(fig, Path(getattr(args, 'output_dir', 'output')) / f'{filename}_{stamp}')
