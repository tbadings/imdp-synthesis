import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_boxes(ax, model, plot_dimensions=[0, 1], labels=False, latex=False, size=12):
    ''' Plot the target, initial, and unsafe state sets '''

    lsize = size + 4

    # Plot target set
    for set in model.goal:
        low = set[0]
        high = set[1]

        width, height = (high - low)[plot_dimensions]
        ax.add_patch(Rectangle(low[plot_dimensions], width, height, facecolor='green', alpha=0.3, edgecolor='green'))

        MID = (high + low)[plot_dimensions] / 2
        if labels:
            if latex:
                text = '$\mathcal{X}_T$'
            else:
                text = 'X_T'
            ax.annotate(text, MID, color='green', fontsize=lsize, ha='center', va='center')

    # Plot unsafe set
    for set in model.critical:
        low = set[0]
        high = set[1]

        width, height = (high - low)[plot_dimensions]
        ax.add_patch(Rectangle(low[plot_dimensions], width, height, facecolor='red', alpha=0.3, edgecolor='red'))

        MID = (high + low)[plot_dimensions] / 2
        if labels:
            if latex:
                text = '$\mathcal{X}_U$'
            else:
                text = 'X_U'
            ax.annotate(text, MID, color='red', fontsize=lsize, ha='center', va='center')

    return


def plot_grid(ax, state_min, state_max, size=[1,1]):

    min_floor = state_min // np.array(size)
    max_ceil = state_max // np.array(size) + 1

    X = np.arange(min_floor[0], max_ceil[0], size[0])
    Y = np.arange(min_floor[1], max_ceil[1], size[1])

    # Vertical lines
    ax.hlines(Y, xmin=state_min[0] - 1, xmax=state_max[0] + 1, color='gray', linewidth=0.5)
    ax.vlines(X, ymin=state_min[1] - 1, ymax=state_max[1] + 1, color='gray', linewidth=0.5)


def set_plot_ticks(ax, state_min, state_max, width):

    regions = np.round((state_max - state_min) / width)

    # Plot at most x ticks per axis, but at least skip every 4 regions
    skip_x = max(1, regions[0] // 5, 4)
    skip_y = max(1, regions[1] // 10, 4)

    major_ticks_x = np.arange(state_min[0] + 0.5 * width[0], state_max[0] - 0.5 * width[0], skip_x * width[0])
    major_ticks_y = np.arange(state_min[1] + 0.5 * width[1], state_max[1] - 0.5 * width[1], skip_y * width[1])

    # minor_ticks_x = np.arange(state_min[0] + 0.5 * width[0], state_max[0] - 0.5 * width[0], width[0])
    # minor_ticks_y = np.arange(state_min[1] + 0.5 * width[1], state_max[1] - 0.5 * width[1], width[1])

    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    # ax.set_xticks(minor_ticks_x, minor=True)
    # ax.set_yticks(minor_ticks_y, minor=True)

    for axi in (ax.xaxis, ax.yaxis):
        for tic in axi.get_minor_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)

    # plt.grid(which='minor', color='#CCCCCC', linewidth=0.3)

def set_plot_lims(ax, state_min, state_max):

    # Goal x-y limits
    ax.set_xlim(state_min[0], state_max[0])
    ax.set_ylim(state_min[1], state_max[1])