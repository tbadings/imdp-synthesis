#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from scipy.interpolate import CubicSpline

from core.utils import cm2inch, remove_consecutive_duplicates
from core.plotting.utils import plot_boxes, plot_grid, set_plot_lims, set_plot_ticks

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def _format_state_label_math(var_name):
    if '_' in var_name:
        head, tail = var_name.split('_', 1)
        return f'${head}_{{{tail}}}$'
    return f'${var_name}$'


def _cuboid_faces(low, high):
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


def _plot_cuboid(ax, low, high, facecolor, edgecolor, alpha):
    faces = _cuboid_faces(low, high)
    collection = Poly3DCollection(
        faces,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=0.4,
        alpha=alpha,
    )
    ax.add_collection3d(collection)


def plot_traces_3d(args, stamp, idx_show, partition, model, traces, num_traces=10, filename="traces_3d", show_ticks=None, camera_angles=None):
    """Plot 3D traces.

    New params:
    - grid_linewidth: float or None -> line width for grid (very thin if None defaults)
    - camera_angles: list of (elev, azim) tuples. If provided, saves one file per angle.
    """
    fig = plt.figure(figsize=cm2inch(7.5, 6.5), dpi=1000)
    ax = fig.add_subplot(111, projection='3d')

    font = {'size': 10}
    mpl.rc('font', **font)

    i1, i2, i3 = np.array(idx_show, dtype=int)

    # ax.set_xlabel(_format_state_label_math(model.state_variables[i1]), labelpad=4)
    # ax.set_ylabel(_format_state_label_math(model.state_variables[i2]), labelpad=4)
    # ax.set_zlabel(_format_state_label_math(model.state_variables[i3]), labelpad=4)

    state_lb = np.array(partition.boundary_lb)[[i1, i2, i3]]
    state_ub = np.array(partition.boundary_ub)[[i1, i2, i3]]

    for set_ in model.goal:
        low = np.array(set_[0])[[i1, i2, i3]]
        high = np.array(set_[1])[[i1, i2, i3]]
        _plot_cuboid(ax, low, high, facecolor='green', edgecolor='none', alpha=0.48)

    for set_ in model.critical:
        low = np.array(set_[0])[[i1, i2, i3]]
        high = np.array(set_[1])[[i1, i2, i3]]
        _plot_cuboid(ax, low, high, facecolor='red', edgecolor='none', alpha=0.44)

    for trace in traces.values():
        state_trace = np.array(trace['x'])[:, [i1, i2, i3]]
        ax.scatter(state_trace[:, 0], state_trace[:, 1], state_trace[:, 2], s=1, color='black', edgecolors='none', alpha=0.7)

    ax.set_xlim(state_lb[0], state_ub[0])
    ax.set_ylim(state_lb[1], state_ub[1])
    ax.set_zlim(state_lb[2], state_ub[2])
    try:
        ax.set_box_aspect((state_ub - state_lb).tolist())
    except Exception:
        pass

    widths = np.array(partition.cell_width)[[i1, i2, i3]]
    lowers = np.array(partition.boundary_lb)[[i1, i2, i3]]
    uppers = np.array(partition.boundary_ub)[[i1, i2, i3]]
    counts = np.array(partition.number_per_dim)[[i1, i2, i3]]

    tick_sets = []
    for lower, upper, width in zip(lowers, uppers, widths):
        ticks = np.arange(lower, upper + width, width)
        tick_sets.append(ticks)

    formatter = FuncFormatter(lambda value, pos: f'{value:.3g}')
    ax.set_xticks(tick_sets[0])
    ax.set_yticks(tick_sets[1])
    ax.set_zticks(tick_sets[2])
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)
    ax.tick_params(axis='both', which='major', labelsize=0, direction='in', length=3)

    # Use a very thin default grid line width unless overridden
    ax.grid(True, linestyle='--', linewidth=1, opacity=0.5)

    # Default to one sensible view if no camera angles provided
    if camera_angles is None:
        camera_angles = [(60, 15)]
    if args.plot_title:
        ax.set_title(f"Simulation for {args.model}")

    output_dir = Path(getattr(args, 'output_dir', 'output'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save one file per camera angle
    for vi, (elev, azim) in enumerate(camera_angles):
        ax.view_init(elev=elev, azim=azim)
        fig.tight_layout()

        pdf_name = output_dir / f'{filename}_{stamp}_view{vi}_{stamp}.pdf'
        png_name = output_dir / f'{filename}_{stamp}_view{vi}_{stamp}.png'
        plt.savefig(pdf_name, format='pdf', bbox_inches='tight')
        plt.savefig(png_name, format='png', bbox_inches='tight')

    plt.close(fig)


def plot_traces(args, stamp, idx_show, partition, model, traces, line=True, num_traces=10, add_unsafe_box=True, filename="traces", show_ticks=None):
    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5), dpi=300)

    font = {'size': 10}
    mpl.rc('font', **font)

    i1, i2 = np.array(idx_show, dtype=int)

    plt.xlabel(_format_state_label_math(model.state_variables[i1]), labelpad=2)
    plt.ylabel(_format_state_label_math(model.state_variables[i2]), labelpad=2)

    if add_unsafe_box:
        expand = 1
    else:
        expand = 0

    # show_ticks overrides args.plot_ticks when explicitly set
    _show_ticks = args.plot_ticks if show_ticks is None else show_ticks

    if _show_ticks == 'nice' or show_ticks is True:
        # Keep a small number of readable ticks and avoid scientific-offset labels.
        formatter = FuncFormatter(lambda x, pos: f'{x:.3g}')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis='both', which='major', labelsize=6, direction='in', length=3, top=False, right=False)
    elif _show_ticks:
        set_plot_ticks(ax,
                  state_min=np.array(partition.boundary_lb)[[i1, i2]] - expand,
                  state_max=np.array(partition.boundary_ub)[[i1, i2]] + expand,
                  width=np.array(partition.cell_width)[[i1, i2]])
        formatter = FuncFormatter(lambda x, pos: f'{x:.3g}')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis='both', which='major', labelsize=6, direction='in', length=3, top=False, right=False)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    
    set_plot_lims(ax,
                  state_min=np.array(partition.boundary_lb)[[i1, i2]] - expand,
                  state_max=np.array(partition.boundary_ub)[[i1, i2]] + expand)

    # Plot grid
    if args.plot_grid:
        plot_grid(ax,
                state_min=np.array(partition.boundary_lb)[[i1, i2]],
                state_max=np.array(partition.boundary_ub)[[i1, i2]],
                size=[1, 1])

    # Plot goal/unsafe regions
    plot_boxes(ax, model, plot_dimensions=[i1, i2])

    # Mark missing projected states (from sparse partitions) with a thin red X.
    idxs = np.asarray(partition.region_idx_inv, dtype=int)
    existing_proj = {tuple(v.tolist()) for v in idxs[:, [i1, i2]]}
    w1 = float(partition.cell_width[i1])
    w2 = float(partition.cell_width[i2])
    x0 = float(partition.boundary_lb[i1])
    y0 = float(partition.boundary_lb[i2])

    for ix in range(partition.number_per_dim[i1]):
        for iy in range(partition.number_per_dim[i2]):
            if (ix, iy) in existing_proj:
                continue

            x_low = x0 + ix * w1
            x_high = x_low + w1
            y_low = y0 + iy * w2
            y_high = y_low + w2

            ax.plot([x_low, x_high], [y_low, y_high], color='red', linewidth=0.4, alpha=1.0, zorder=4)
            ax.plot([x_low, x_high], [y_high, y_low], color='red', linewidth=0.4, alpha=1.0, zorder=4)

    # Plot boundary of unsafe regions if requested
    if add_unsafe_box:
        state_lb = np.array(partition.boundary_lb)
        state_ub = np.array(partition.boundary_ub)

        LOWS = [np.array([state_lb[i1] - expand, state_lb[i2] - expand]),
                np.array([state_lb[i1], state_lb[i2] - expand]),
                np.array([state_lb[i1], state_ub[i2]]),
                np.array([state_ub[i1], state_lb[i2] - expand])
                ]
        HIGHS = [np.array([state_lb[i1], state_ub[i2] + expand]),
                 np.array([state_ub[i1], state_lb[i2]]),
                 np.array([state_ub[i1], state_ub[i2] + expand]),
                 np.array([state_ub[i1] + expand, state_ub[i2] + expand]),
                 ]

        for low, high in zip(LOWS, HIGHS):
            width, height = (high - low)
            ax.add_patch(Rectangle(low, width, height, facecolor='red', alpha=0.3, edgecolor='red'))

    # Add traces
    i = 0
    for trace in traces.values():
        state_trace = np.array(trace['x'])[:, [i1, i2]]

        # Only show trace if there are at least two points
        if len(state_trace) < 2:
            continue
        else:
            i += 1

        # Stop at desired number of traces
        if i > num_traces:
            break

        # state_trace = remove_consecutive_duplicates(state_trace)

        # Plot precise points
        plt.plot(*state_trace.T, 'o', markersize=1, color="black");

        if line:
            # Linear length along the line:
            distance = np.cumsum(np.sqrt(np.sum(np.diff(state_trace, axis=0) ** 2,
                                                axis=1)))
            distance = np.insert(distance, 0, 0) / distance[-1]

            # Interpolation for different methods:
            alpha = np.linspace(0, 1, 75)

            if len(state_trace) == 2:
                kind = 'linear'
            else:
                kind = 'quadratic'

            cs = CubicSpline(distance, state_trace, bc_type='natural')
            interpolated_points = cs(alpha)

            # Plot trace
            plt.plot(*interpolated_points.T, '-', color="blue", linewidth=1);

    # plt.gca().set_aspect('equal')

    # Set tight layout
    fig.tight_layout()

    if args.plot_title:
        ax.set_title(f"Simulation for {args.model}")

    # Save figure
    output_dir = Path(getattr(args, 'output_dir', 'output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{filename}_{stamp}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(output_dir / f'{filename}_{stamp}.png', format='png', bbox_inches='tight')

    plt.close(fig)
