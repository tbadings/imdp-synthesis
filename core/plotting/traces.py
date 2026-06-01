#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
from matplotlib.patches import Rectangle
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
    plt.savefig(f'output/{filename}_{stamp}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'output/{filename}_{stamp}.png', format='png', bbox_inches='tight')

    plt.show()
