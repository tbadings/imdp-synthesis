from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from core.plotting.traces import _format_state_label_math
from core.plotting.utils import col, save_fig, plot_missing_cells

def heatmap(args, stamp, idx_show, slice_values=None, partition=None, results=None, filename="heatmap", **kwargs):
    # Compute 2D satisfaction probabilities
    i1, i2 = np.array(idx_show, dtype=int)
    nx, ny = partition.number_per_dim[i1], partition.number_per_dim[i2]
    idxs = np.asarray(partition.region_idx_inv, dtype=int)
    res = np.asarray(results)[:len(idxs)]

    sum_vals, counts = np.zeros((ny, nx)), np.zeros((ny, nx))
    valid = ~np.isnan(res)
    np.add.at(sum_vals, (idxs[valid, i2], idxs[valid, i1]), res[valid])
    np.add.at(counts, (idxs[valid, i2], idxs[valid, i1]), 1)

    values = np.full((ny, nx), np.nan)
    mask = counts > 0
    values[mask] = sum_vals[mask] / counts[mask]

    # Plot heatmap grid
    DF = pd.DataFrame(values[::-1, :], index=partition.regions_per_dim['centers'][i2][::-1], columns=partition.regions_per_dim['centers'][i1])
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    ax = sns.heatmap(DF, cmap='rocket', vmin=0, vmax=1, xticklabels=False, yticklabels=False, square=False,
                     cbar_kws={'label': 'Satisfaction Probability', 'shrink': 0.8, 'aspect': 25, 'pad': 0.03}, ax=ax)
    ax.set_box_aspect(1)

    # Frame styling
    for s in ('top', 'bottom', 'left', 'right'):
        ax.spines[s].set_visible(False)
        ax.spines[s].set_color(col('gray'))

    # Colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18, length=5, width=1)
    cbar.set_label('Satisfaction Probability', fontsize=18, labelpad=12)
    cbar.outline.set_linewidth(1)

    # Missing inactive cells
    rows, cols = np.where(np.isnan(DF.to_numpy()))
    plot_missing_cells(ax, cols, rows, 1, 1, zorder=5)

    # Axis ticks and labels
    if args.plot_ticks:
        x_lb, x_ub = partition.boundary_lb[i1], partition.boundary_ub[i1]
        y_lb, y_ub = partition.boundary_lb[i2], partition.boundary_ub[i2]
        loc = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
        x_v = [v for v in loc.tick_values(x_lb, x_ub) if x_lb <= v <= x_ub]
        y_v = [v for v in loc.tick_values(y_lb, y_ub) if y_lb <= v <= y_ub]
        ax.set_xticks([(v - x_lb) / (x_ub - x_lb) * nx for v in x_v])
        ax.set_xticklabels([f'{round(v, 1):g}' for v in x_v])
        ax.set_yticks([(y_ub - v) / (y_ub - y_lb) * ny for v in y_v])
        ax.set_yticklabels([f'{round(v, 1):g}' for v in y_v])
        ax.tick_params(labelsize=18, length=5, width=1, color=col('dimgray'), rotation=0)

    model = kwargs.get('model')
    if model and hasattr(model, 'state_variables'):
        ax.set_xlabel(_format_state_label_math(model.state_variables[i1]), fontsize=18, labelpad=10)
        ax.set_ylabel(_format_state_label_math(model.state_variables[i2]), fontsize=18, labelpad=10)
    if args.plot_title:
        ax.set_title(f"Heatmap for {args.model} ({filename})", fontsize=18, pad=12)

    # Save figure
    fig.tight_layout()
    save_fig(fig, Path(getattr(args, 'output_dir', 'output')) / f'{filename}_{stamp}')
