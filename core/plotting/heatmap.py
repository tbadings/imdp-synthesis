import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl

from core.plotting.utils import set_plot_ticks

def heatmap(args, stamp, idx_show, slice_values, partition, results, filename="heatmap"):
    '''
    Create heat map for the satisfaction probability from any initial state.

    Parameters
    ----------

    Returns
    -------
    None.

    '''

    font = {'size': 10}
    mpl.rc('font', **font)

    sns.set_style("whitegrid", {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

    i1, i2 = np.array(idx_show, dtype=int)

    lb = np.array(partition.boundary_lb)
    ub = np.array(partition.boundary_ub)

    values = np.zeros((partition.number_per_dim[i2], partition.number_per_dim[i1]))
    slice_idx = np.array(((slice_values - lb) / (ub - lb) * np.array(partition.number_per_dim)) // 1, dtype=int)

    # Fill values in matrix to plot in heatmap
    for x in range(partition.number_per_dim[i1]):
        for y in range(partition.number_per_dim[i2]):
            slice_at = slice_idx
            slice_at[i1] = x
            slice_at[i2] = y

            # Retrieve state ID
            state_idx = partition.region_idx_array[tuple(slice_at)]
            
            # Fill heatmap value
            values[y, x] = results[state_idx]

    X = partition.regions_per_dim['centers'][i1]
    Y = partition.regions_per_dim['centers'][i2]

    DF = pd.DataFrame(values[::-1, :], index=Y[::-1], columns=X)

    ax = sns.heatmap(DF)

    if args.plot_ticks:
        set_plot_ticks(ax,
                  state_min=np.array(partition.boundary_lb)[[i1, i2]] - expand,
                  state_max=np.array(partition.boundary_ub)[[i1, i2]] + expand,
                  width=np.array(partition.cell_width))
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if args.plot_title:
        ax.set_title(f"Heatmap for {args.model} ({filename})")

    # Save figure
    plt.savefig(f'output/{filename}_{stamp}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'output/{filename}_{stamp}.png', format='png', bbox_inches='tight')

    plt.show()
