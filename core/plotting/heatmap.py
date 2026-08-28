import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from pathlib import Path



def heatmap(args, stamp, idx_show, slice_values=None, partition=None, results=None, filename="heatmap", **kwargs):
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

    nx = partition.number_per_dim[i1]
    ny = partition.number_per_dim[i2]

    idxs = np.asarray(partition.region_idx_inv, dtype=int)
    num_states = len(idxs)
    res = np.asarray(results)[:num_states]

    sum_vals = np.zeros((ny, nx), dtype=float)
    counts = np.zeros((ny, nx), dtype=int)

    # Accumulate sums and counts for states (averaging over missing dimensions)
    valid = ~np.isnan(res)
    x_coords = idxs[valid, i1]
    y_coords = idxs[valid, i2]
    valid_res = res[valid].astype(float)

    np.add.at(sum_vals, (y_coords, x_coords), valid_res)
    np.add.at(counts, (y_coords, x_coords), 1)

    values = np.full((ny, nx), fill_value=np.nan, dtype=float)
    mask = counts > 0
    values[mask] = sum_vals[mask] / counts[mask]

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
    output_dir = Path(getattr(args, 'output_dir', 'output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{filename}_{stamp}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(output_dir / f'{filename}_{stamp}.png', format='png', bbox_inches='tight')

    plt.close()
