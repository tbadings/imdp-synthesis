from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.collections import LineCollection

mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm', 'hatch.linewidth': 0.3})

# Color constants
GOAL_COLOR = 'forestgreen'
CRITICAL_COLOR = 'crimson'
UNACTIVE_COLOR = 'darkred'
START_COLOR = 'black'
END_COLOR = 'black'
GOAL_HATCH = '//////'
CRITICAL_HATCH = 'xxxxxx'

def col(c):
    # Resolve color name or LaTeX tint
    if isinstance(c, str) and '!' in c:
        name, pct = c.split('!')
        rgb = mcolors.to_rgb(name)
        p = float(pct) / 100
        return tuple(p * x + (1 - p) for x in rgb)
    return c

def plot_boxes(ax, model, plot_dimensions=[0, 1], labels=False, latex=False, size=12):
    # Target and unsafe regions
    d = plot_dimensions
    sets = [(model.goal, GOAL_COLOR, 'darkgreen', GOAL_HATCH, r'$\mathcal{X}_T$' if latex else 'X_T'),
            (model.critical, CRITICAL_COLOR, 'darkred', CRITICAL_HATCH, r'$\mathcal{X}_U$' if latex else 'X_U')]
    for boxes, fc, ec, hatch, lbl in sets:
        for s in boxes:
            w, h = (s[1] - s[0])[d]
            ax.add_patch(Rectangle(s[0][d], w, h, facecolor=col(fc), alpha=0.4, edgecolor=col(ec), lw=0, hatch=hatch))
            if labels:
                ax.annotate(lbl, (s[0] + s[1])[d] / 2, color=col(ec), fontsize=size + 2, ha='center', va='center')

def plot_grid(ax, state_min, state_max, size=[1, 1]):
    # Background grid lines
    X = np.arange(state_min[0] // size[0], state_max[0] // size[0] + 1) * size[0]
    Y = np.arange(state_min[1] // size[1], state_max[1] // size[1] + 1) * size[1]
    ax.hlines(Y, state_min[0] - 1, state_max[0] + 1, color=col('lightgray'), lw=0.5, ls=':', alpha=0.8)
    ax.vlines(X, state_min[1] - 1, state_max[1] + 1, color=col('lightgray'), lw=0.5, ls=':', alpha=0.8)

def set_plot_ticks(ax, state_min=None, state_max=None, width=None):
    # Ticks and numeric format
    fmt = FuncFormatter(lambda v, p: f'{round(v, 1):g}')
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        axis.set_major_formatter(fmt)
    ax.tick_params(labelsize=18, direction='out', length=5, width=1, color=col('dimgray'))

def set_plot_lims(ax, state_min, state_max):
    # State space limits
    ax.set_xlim(state_min[0], state_max[0])
    ax.set_ylim(state_min[1], state_max[1])

def style_axes(ax):
    # Despine and subtle grid
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(col('gray'))
    ax.grid(True, ls=':', color=col('lightgray'), lw=0.5, alpha=0.8)

def save_fig(fig, path_without_ext):
    # Save PDF and PNG
    p = Path(path_without_ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{p}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_missing_cells(ax, x, y, w=1, h=1, zorder=4):
    # Cross pattern for inactive states
    if len(x) == 0:
        return
    segs = np.empty((len(x) * 2, 2, 2))
    segs[0::2, 0, :] = np.column_stack([x, y])
    segs[0::2, 1, :] = np.column_stack([x + w, y + h])
    segs[1::2, 0, :] = np.column_stack([x, y + h])
    segs[1::2, 1, :] = np.column_stack([x + w, y])
    ax.add_collection(LineCollection(segs, colors=col(UNACTIVE_COLOR), lw=0.4, alpha=0.7, zorder=zorder))