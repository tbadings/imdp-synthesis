import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl



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

    sns.set_style("white", {
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
            state_idx, exists = partition.grid_idx2state(slice_at)

            # Fill heatmap value
            values[y, x] = results[state_idx] if exists else np.nan

    X = partition.regions_per_dim['centers'][i1]
    Y = partition.regions_per_dim['centers'][i2]

    DF = pd.DataFrame(values[::-1, :], index=Y[::-1], columns=X)

    show_ticks = args.plot_ticks
    ax = sns.heatmap(DF,
                     xticklabels=False,
                     yticklabels=False)

    # Mark missing states (NaN cells from sparse partitions) with a thin X.
    missing_mask = np.isnan(DF.to_numpy())
    missing_rows, missing_cols = np.where(missing_mask)
    for row, col in zip(missing_rows, missing_cols):
        ax.plot([col, col + 1], [row, row + 1], color='red', linewidth=0.4, alpha=1.0, zorder=5)
        ax.plot([col, col + 1], [row + 1, row], color='red', linewidth=0.4, alpha=1.0, zorder=5)

    if show_ticks:
        fmt = lambda v: f'{v:.4g}'
        n_ticks_x = min(6, len(X))
        n_ticks_y = min(6, len(Y))

        x_idx = np.unique(np.linspace(0, len(X) - 1, n_ticks_x, dtype=int))
        y_idx = np.unique(np.linspace(0, len(Y) - 1, n_ticks_y, dtype=int))

        # Heatmap cells are centered at integer + 0.5 positions.
        ax.set_xticks(x_idx + 0.5)
        ax.set_yticks(y_idx + 0.5)
        ax.set_xticklabels([fmt(X[i]) for i in x_idx])
        ax.set_yticklabels([fmt(Y[::-1][i]) for i in y_idx])

        ax.tick_params(axis='x', labelsize=8, rotation=0)
        ax.tick_params(axis='y', labelsize=8, rotation=0)

    if args.plot_title:
        ax.set_title(f"Heatmap for {args.model} ({filename})")

    # Save figure
    plt.savefig(f'output/{filename}_{stamp}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'output/{filename}_{stamp}.png', format='png', bbox_inches='tight')

    plt.close()
