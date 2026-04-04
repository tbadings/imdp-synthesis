#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.interpolate import CubicSpline

from core.utils import cm2inch, remove_consecutive_duplicates
from core.plotting.utils import plot_boxes, plot_grid, set_plot_lims, set_plot_ticks

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def plot_traces(args, stamp, idx_show, partition, model, traces, line=True, num_traces=10, add_unsafe_box=True, filename="traces"):
    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5), dpi=300)

    font = {'size': 10}
    mpl.rc('font', **font)

    i1, i2 = np.array(idx_show, dtype=int)

    plt.xlabel(f'${model.state_variables[i1]}$', labelpad=2)
    plt.ylabel(f'${model.state_variables[i2]}$', labelpad=2)

    if add_unsafe_box:
        expand = 1
    else:
        expand = 0

    if args.plot_ticks:
        set_plot_ticks(ax,
                  state_min=np.array(partition.boundary_lb)[[i1, i2]] - expand,
                  state_max=np.array(partition.boundary_ub)[[i1, i2]] + expand,
                  width=np.array(partition.cell_width))
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
