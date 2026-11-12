"""
Fixed-arguments launcher for experimentation.

The RL settings (PPO training, reward function, and the tube around the rollouts) live in
each benchmark's `rl_config` in `benchmarks/`; only what is not benchmark-specific is passed
here. Any RL option can still be overridden per run by adding e.g. "--total_timesteps",
"1000" below, which takes precedence over the benchmark's `rl_config`.
"""

import ast
import csv
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent

def run_fixed_SVMDP(args: list[str]) -> None:
    runfile = ROOT / "Main_SVMDP.py"

    cmd = [sys.executable, str(runfile), *args]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        # Negative code = killed by a signal (-9 is typical for an OS out-of-memory kill)
        print(f"!! Run failed (exit code {result.returncode}): {' '.join(args)}", file=sys.stderr)


# ---------------------------------------------------------------------------------------------
# TikZ/pgfplots figures from the runs in experiment_logs_Q4
# ---------------------------------------------------------------------------------------------

# Style per method. Keyed by method (not by plot order), so a method has the same color in every
# plot. Markers differ too, so the series stay distinguishable in grayscale print.
_METHOD_STYLE = {
    "ppo":   {"label": "PPO",            "rgb": "42,120,214", "mark": "*"},
    "sac":   {"label": "SAC",            "rgb": "235,104,52", "mark": "square*"},
    "dense": {"label": "Dense", "rgb": "0,0,0",      "mark": "triangle*"},
}
_FALLBACK_STYLE = {"rgb": "27,175,122", "mark": "diamond*"}

# The y labels are LaTeX: write them as raw strings (r"..."), so a backslash reaches LaTeX as is
# (in a normal string, e.g. "\t" in "\theta" would become a tab). Macros used here must be defined
# in the paper; for the standalone preview.tex, give them a definition in _PREVIEW_MACROS.
# A label taller than the axis box can be split over two lines with \\.
# "legend": put a legend inside that plot, at this pgfplots `legend pos`.
# "data": plot this metric's data instead of the key's own (to show one metric in two ways).
# "break": (b, f) gives a broken y axis on [0, 1]: values 0..b take the bottom fraction f of the
#          plot height and b..1 the rest, with a zigzag line across the plot at b.
# "invlog": plot 1 - p instead of p. With "ymode=log, y dir=reverse" in "axis", this gives an inverse
#           log axis: each "nine" (0-0.9, 0.9-0.99, ...) takes the same height and p = 1 is at the top.
#           The ticks are then given in 1 - p, with the labels in p.
_METRICS = {
    "bound":   {"ylabel": r"Reach-avoid prob.", "axis": "ymin=0.9, ymax=1", "dense": True,
                "legend": "south west", "note": "Bounds below ymin lie outside the axis range and are clipped."},
    "bound_broken": {"data": "bound", "ylabel": r"Reach-avoid prob.", "break": (0.9, 0.5), "dense": True,
                     "axis": "ymin=0, ymax=1, ytick={0,0.5,0.9,0.95,1}, yticklabels={0,0.5,0.9,0.95,1}",
                     "legend": "south west",
                     "note": "Broken y axis: 0-0.9 fills the bottom half of the plot, 0.9-1 the top half."},
    "bound_invlog": {"data": "bound", "ylabel": r"Reach-avoid prob.", "invlog": True, "dense": True,
                     # Labeled ticks at p = 0, 0.5, 0.9, 0.95, 0.99 and, as usual on a log scale, minor
                     # ticks at 2, 3, ..., 9 times each power of 10 (in 1 - p)
                     "axis": "ymode=log, y dir=reverse, ymin=0.005, ymax=1,\n"
                             "    ytick={1,0.5,0.1,0.05,0.01}, yticklabels={0,0.5,0.9,0.95,0.99},\n"
                             "    minor ytick={0.2,0.3,...,0.9, 0.02,0.03,...,0.09, 0.006,0.007,...,0.009}",
                     "legend": "south west",
                     "note": "Inverse log y axis: the y coordinates are 1 - p on a reversed log axis; the tick labels show p. "
                             "Bounds above 0.995 lie outside the axis range and are clipped."},
    # Ticks of 1000 s or more are shown as multiples of 10^3 (as "1, 2, ..." with "x10^3" on top),
    # since labels like "4,000" are too wide for the left label margin
    "runtime": {"ylabel": r"Total runtime [s]", "axis": "ymin=0, scale ticks above exponent=2", "dense": True},
    "states":  {"ylabel": r"Abstract states ($|S|$)", "axis": "ymin=0", "dense": True},
    "states_log": {"data": "states", "ylabel": r"Abstract states ($|S|$)", "axis": "ymode=log", "dense": True},
}
# Per benchmark, the legend position (pgfplots `legend pos`) that replaces the metrics' "legend" above
_MODEL_LEGEND = {
    "Dubins4D_x0set": "north east",
}
# Per benchmark, axis options that replace a metric's "axis" above (e.g. a fixed y range that only
# suits that benchmark's data). Without an entry, the metric's "axis" is used, and with only ymin
# set, pgfplots picks the upper limit and ticks from the data.
_MODEL_AXIS = {
    "Drone4D_x0set": {"runtime": "ymin=0, ymax=150, ytick={0,50,100,150}",
                      # The data spans less than a decade, for which pgfplots would label 10^5.5
                      "states_log": "ymode=log, ymin=5e4, ymax=1e6"},
}
_PREVIEW_MACROS = [r"\providecommand{\satprob}{\mathrm{Pr}}",  # as in the paper's ancillary/include.tex
                   r"\providecommand{\X}{\mathcal{X}}"]

# x-axis: the span (width) of the initial state set X0 in the x and y position coordinates, rounded
# up to whole grid cells like the abstraction does (it uses every cell that overlaps the logged
# x0_box; see box2grid_idxs in core/abstraction/partition.py).
_XLABEL = r"Width of $\X_0$ in $x,y$"
# Per benchmark, the x and y state dimensions as (dimension, grid lower bound, grid upper bound,
# number of cells), copied from partition['boundary'] and ['number_per_dim'] in benchmarks/<model>.py.
# Keep these in sync with the benchmark files.
_XY_DIMS = {
    "Drone4D_x0set":  [(0, -20.0, 20.0, 80), (2, -20.0, 20.0, 80)],  # state [x, v_x, y, v_y]
    "Dubins4D_x0set": [(0, -10.0, 10.0, 100), (1, 0.0, 10.0, 50)],   # state [x, y, theta, v]
    "CartPole_x0set": [(0, -2.4, 2.4, 200), (2, -0.25, 0.25, 200)],  # same X0 scaling as Drone4D_x0set
}
# Per benchmark, the number of cells per dimension of the dense grid (partition['number_per_dim'] in
# benchmarks/<model>.py; keep in sync). The dense baseline has one state per cell, whatever X0 is, so
# the states plot shows this number for every X0 span without a dense run (e.g. one that ran out of memory).
_DENSE_GRID = {
    "Dubins4D_x0set": (100, 50, 100, 100),  # 5.0*10^7 states
}


def _x0_cell_span(lb: float, ub: float, grid_lb: float, grid_ub: float, n: int) -> float:
    """Width of the grid cells overlapping [lb, ub] in one dimension (as box2grid_idxs selects them)."""
    cell = (grid_ub - grid_lb) / n
    t_lb, t_ub = (lb - grid_lb) / cell, (ub - grid_lb) / cell
    # Snap box edges that lie on grid lines (float round-off), with the same tolerance as box2grid_idxs
    t_lb, t_ub = (round(t) if abs(t - round(t)) <= 1e-4 else t for t in (t_lb, t_ub))
    lo = min(max(math.floor(t_lb), 0), n - 1)
    hi = min(max(lo, math.ceil(t_ub) - 1), n - 1)
    return round((hi - lo + 1) * cell, 9)

# Every picture is exactly `width` wide: the axis box gets the width minus these margins, which hold
# the y tick labels and y label (left) and the overhang of the last x tick label (right). All plots
# use the same margins, so their axes line up when placed side by side.
_LABEL_MARGIN_LEFT = "25pt"
_LABEL_MARGIN_RIGHT = "4pt"


def _read_summary(path: Path) -> dict[str, str]:
    with open(path, newline="") as f:
        return {row[0]: row[1] for row in csv.reader(f) if len(row) >= 2}


def load_Q4_results(log_dir: Path) -> dict:
    """Collect the run summaries in `log_dir`.

    Returns results[model][method][x0_span][seed] = {"bound", "runtime", "states"}, where x0_span
    is the width of the initial state set X0 in the x and y coordinates (see _XY_DIMS), rounded up
    to whole grid cells, and method
    is "dense" for the dense baseline and the RL algorithm (e.g. "ppo") otherwise. The
    runtime is the total of RL training, abstraction, and synthesis (Monte Carlo simulations are
    excluded). Runs without a summary.csv (e.g., killed for running out of memory) are skipped.
    If a configuration was run more than once, the most recent run is used.
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # Folder names start with a timestamp, so sorting them puts the runs in chronological order
    for run_dir in sorted(p for p in Path(log_dir).iterdir() if p.is_dir()):
        summary_file = run_dir / "summary.csv"
        if not summary_file.exists():
            print(f"Skipping {run_dir.name} (no summary.csv)", file=sys.stderr)
            continue
        s = _read_summary(summary_file)
        method = "dense" if s["dense"] == "True" else (s.get("rl_algo") or "rl")
        runtime = sum(float(s[k]) for k in ("time_RL", "time_abstraction", "time_synthesis") if s.get(k))
        lower, upper = ast.literal_eval(s["x0_box"])
        dx, dy = (_x0_cell_span(lower[d], upper[d], *grid) for d, *grid in _XY_DIMS[s["model"]])
        if abs(dx - dy) > 1e-9:
            print(f"{run_dir.name}: X0 spans {dx:g} in x but {dy:g} in y; plotting the span in x",
                  file=sys.stderr)
        results[s["model"]][method][dx][int(s["seed"])] = {
            "bound": float(s["optimal_value"]),
            "runtime": runtime,
            "states": int(s["abstraction_states"]),
        }
    return results


def _coords(points) -> str:
    return " ".join(f"({x:g},{y:.6g})" for x, y in points)


def _nice_ticks(lo: float, hi: float, max_ticks: int = 5) -> list[float]:
    """Round tick values (multiples of 1, 2, 5, 10, ...) in [lo, hi], at most max_ticks of them."""
    step = 10 ** math.floor(math.log10(max(hi - lo, 1e-9) / max_ticks))
    while True:
        for factor in (1, 2, 5):
            ticks = [k * step * factor for k in range(math.ceil(lo / (step * factor)),
                                                       math.floor(hi / (step * factor)) + 1)]
            if len(ticks) <= max_ticks:
                return ticks
        step *= 10


def _ordered(methods) -> list[str]:
    """RL methods in a fixed order (PPO, SAC, others), with the dense baseline last."""
    order = list(_METHOD_STYLE)
    return sorted(methods, key=lambda m: (m == "dense", order.index(m) if m in order else len(order), m))


def _style(m: str) -> dict:
    return _METHOD_STYLE.get(m, {"label": m.upper(), **_FALLBACK_STYLE})


def _line_options(m: str) -> str:
    opts = f"q4{m}, semithick, mark={_style(m)['mark']}, mark size=0.85pt"
    return opts + (", densely dotted, mark options={solid}" if m == "dense" else "")


def _color_defs(methods) -> list[str]:
    return [f"\\definecolor{{q4{m}}}{{RGB}}{{{_style(m)['rgb']}}}" for m in methods]


def _tikz_plot(model: str, per_method: dict, metric: str, width: str, height: str) -> str:
    """One pgfplots axis: per RL method a min/max band over seeds with the median as a line."""
    spec = _METRICS[metric]
    data = spec.get("data", metric)
    # The dense baseline's number of states, at the X0 spans without a dense run (see _DENSE_GRID)
    filled = []
    if data == "states" and spec["dense"] and model in _DENSE_GRID:
        dense = dict(per_method.get("dense", {}))
        filled = sorted({x for m, runs in per_method.items() if m != "dense" for x in runs} - set(dense))
        dense.update({x: {0: {"states": math.prod(_DENSE_GRID[model])}} for x in filled})
        per_method = {**per_method, "dense": dense}
    methods = _ordered(m for m in per_method if m != "dense" or spec["dense"])
    spans = sorted({x for m in methods for x in per_method[m]})
    if len(spans) <= 5:
        # A tick at every X0 span. Widen the x range by half the spacing between the spans, so that the
        # first and last tick labels clear the y tick labels and the right edge of the picture
        xticks = spans
        pad = 0.5 * min((b - a for a, b in zip(spans, spans[1:])), default=0.5)
        xmin, xmax = spans[0] - pad, spans[-1] + pad
    else:
        xticks = _nice_ticks(spans[0], spans[-1])  # at most 5 round ticks, so the labels fit
        xmin, xmax = spans[0], spans[-1]
    # The y coordinate of a value: 1 - p on an inverse log axis, else the value itself
    to_y = (lambda p: 1 - p) if spec.get("invlog") else (lambda v: v)
    axis = _MODEL_AXIS.get(model, {}).get(metric, spec["axis"])

    out = [
        f"% Generated by ICLR_Q4.py from experiment_logs_Q4 -- {model}: {spec['ylabel']}",
        "% Shaded band = min/max over seeds; line = median over seeds.",
        f"% Picture width is exactly {width}. Requires \\usepackage{{pgfplots}}.",
    ]
    if "note" in spec:
        out.append(f"% {spec['note']}")
    # Baseline at the bottom of the axis box: in a row of \begin{minipage}[b]s, the axes then line up
    out += ["\\begin{tikzpicture}[baseline=(current axis.south)]", *_color_defs(methods)]
    out += [
        "\\begin{axis}[",
        f"    scale only axis, width={width}-{_LABEL_MARGIN_LEFT}-{_LABEL_MARGIN_RIGHT}, height={height},",
        f"    xlabel={{{_XLABEL}}},",
        f"    ylabel={{{spec['ylabel']}}}, ylabel style={{align=center}},",
        f"    xmin={xmin:g}, xmax={xmax:g}, xtick={{{','.join(f'{t:g}' for t in xticks)}}},",
        f"    {axis},",
        "    grid=major, grid style={gray!20}, clip marker paths=true,",
        "    label style={font=\\footnotesize}, tick label style={font=\\fontsize{8}{9}\\selectfont, inner sep=2pt},",
        "    xlabel shift=-2pt, ylabel shift=-3pt,  % labels close to the tick labels",
    ]
    if "break" in spec:
        # Piecewise-linear y scale: [0, b] -> [0, f] and [b, 1] -> [f, 1]
        b, f = spec["break"]
        lo, hi = f / b, (1 - f) / (1 - b)
        out += [
            f"    y coord trafo/.code={{\\pgfmathparse{{#1 < {b:g} ? #1*{lo:g} : {f:g} + (#1 - {b:g})*{hi:g}}}}},",
            f"    y coord inv trafo/.code={{\\pgfmathparse{{#1 < {f:g} ? #1/{lo:g} : {b:g} + (#1 - {f:g})/{hi:g}}}}},",
        ]
    if "legend" in spec:
        out += [
            f"    legend pos={_MODEL_LEGEND.get(model, spec['legend'])}, legend cell align=left,",
            "    legend style={xshift=0pt, yshift=0pt, font=\\fontsize{8}{9}\\selectfont, draw=none, fill=white, fill opacity=0.8, text opacity=1,",
            "        inner sep=0.5pt, row sep=-3.5pt},",
            "    % Short line samples (0.36cm instead of 0.6cm) with the marker at full size in the middle",
            "    legend image code/.code={\\draw[mark repeat=2, mark phase=2, #1] plot coordinates {(0cm,0cm) (0.15cm,0cm) (0.30cm,0cm)};},",
        ]
    out += [
        "]",
    ]
    if "break" in spec:
        b, f = spec["break"]
        # Subtle zigzag across the plot (drawn below the data), as plain segments so that no TikZ
        # decoration library is needed: 24 teeth, 0.8pt above/below the break
        n = 48
        zigzag = " -- ".join(f"([yshift={0.8 if i % 2 else -0.8}pt]rel axis cs:{i / n:.4g},{f:g})" for i in range(n + 1))
        out += [
            f"% Zigzag line across the plot at {b:g}, where the y scale changes (drawn below the data)",
            f"\\draw[gray!70, thin, line join=round] {zigzag};",
        ]
    for m in methods:
        per_span = {v: [to_y(run[data]) for run in seeds.values()] for v, seeds in sorted(per_method[m].items())}
        medians = _coords((v, median(vals)) for v, vals in per_span.items())
        if m == "dense":
            out.append(f"% {_style(m)['label']} (single deterministic run per X0 span)")
            if filled:
                out.append(f"% No dense run for X0 span(s) {', '.join(f'{x:g}' for x in filled)}: there, the number "
                           f"of cells of the dense grid ({'x'.join(map(str, _DENSE_GRID[model]))}) is plotted")
        else:
            lower = [(v, min(vals)) for v, vals in per_span.items()]
            upper = [(v, max(vals)) for v, vals in per_span.items()]
            n_seeds = sorted({len(vals) for vals in per_span.values()})
            out.append(f"% {_style(m)['label']}: {'/'.join(map(str, n_seeds))} seeds per X0 span")
            out.append(f"\\addplot[draw=none, fill=q4{m}, fill opacity=0.2, forget plot] coordinates {{"
                       f"{_coords(lower + upper[::-1])}}} -- cycle;")
        out.append(f"\\addplot[{_line_options(m)}] coordinates {{{medians}}};")
        if "legend" in spec:
            out.append(f"\\addlegendentry{{{_style(m)['label']}}}")
    out.append("\\end{axis}")
    out += [
        f"% Fix the bounding box to exactly {width} wide: the axis box plus the label margins",
        f"\\path (current axis.south west |- current bounding box.south) ++(-{_LABEL_MARGIN_LEFT},0) coordinate (q4sw)",
        f"      (current axis.north east |- current bounding box.north) ++({_LABEL_MARGIN_RIGHT},0) coordinate (q4ne);",
        "\\pgfresetboundingbox",
        "\\useasboundingbox (q4sw) rectangle (q4ne);",
        # The trailing % keeps the file's final newline from adding a space after the picture
        "\\end{tikzpicture}%",
        "",
    ]
    return "\n".join(out)


def _tikz_legend(methods) -> str:
    """A legend-only picture shared by all plots (a legend does not fit inside such small plots)."""
    out = [
        "% Generated by ICLR_Q4.py -- shared legend for the Q4 plots. Requires \\usepackage{pgfplots}.",
        "\\begin{tikzpicture}",
        *_color_defs(methods),
        "\\begin{axis}[",
        "    hide axis, scale only axis, width=1pt, height=1pt, xmin=0, xmax=1, ymin=0, ymax=1,",
        "    legend columns=-1, legend style={draw=none, font=\\fontsize{8}{9}\\selectfont, at={(0.5,0.5)}, anchor=center,",
        "        /tikz/every even column/.append style={column sep=1em}},",
        "]",
    ]
    for m in methods:
        out.append(f"\\addlegendimage{{{_line_options(m)}}}")
        out.append(f"\\addlegendentry{{{_style(m)['label']}}}")
    out += [
        "\\addlegendimage{area legend, draw=gray!30, fill=gray!30}",
        "\\addlegendentry{Min--max over seeds}",
        "\\end{axis}",
        "\\end{tikzpicture}%",
        "",
    ]
    return "\n".join(out)


def generate_Q4_plots(log_dir: Path = ROOT / "experiment_logs_Q4", out_dir: Path = ROOT / "figures_Q4",
                      width: str = "\\linewidth", height: str = "2.4cm") -> list[Path]:
                      
    """Write TikZ/pgfplots figures for every benchmark in `log_dir` to `out_dir`.

    Per benchmark, three figures as a function of the span of X0 in x and y, each showing the min/max
    over seeds (band) and the median (line) per RL algorithm:
      <model>_bound.tex    bound on the reach-avoid probability, plus the dense baseline
      <model>_bound_broken.tex  the same, on a broken y axis (0-0.9 and 0.9-1, half the height each)
      <model>_bound_invlog.tex  the same, on an inverse log y axis (equal height per "nine": 0.9, 0.99, ...)
      <model>_runtime.tex  total runtime (RL + abstraction + synthesis)
      <model>_states.tex   number of states of the abstraction (for the dense baseline, see _DENSE_GRID)
      <model>_states_log.tex  the same, on a log y axis
    plus one shared `legend.tex`. Each file holds one tikzpicture to `\\input` into a paper.
    `width` is the width of the whole picture (labels included), `height` that of the axis box.
    By default a plot fills the width of its container, so put each one in a
    `\\begin{minipage}[b]{0.24\\textwidth}`
    ([b] lines up the axes of plots side by side). Do not size the plots with `\\textwidth` directly: inside a
    minipage, LaTeX sets `\\textwidth` to the minipage width, so it no longer is the page's.
    `preview.tex` is a standalone document with the ICLR text width (5.5in) and font that inputs
    all of them like that, for checking the figures with pdflatex.
    """
    results = load_Q4_results(log_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    all_methods = {m for per_method in results.values() for m in per_method}
    if any(model in _DENSE_GRID for model in results):
        all_methods.add("dense")  # its number of states is plotted even without dense runs
    all_methods = _ordered(all_methods)
    legend = out_dir / "legend.tex"
    legend.write_text(_tikz_legend(all_methods))
    written.append(legend)
    preview = [
        "\\documentclass{article}",
        "\\usepackage{times}",
        "\\usepackage[letterpaper, textwidth=5.5in]{geometry}  % ICLR text width",
        "\\usepackage{pgfplots}",
        "\\pgfplotsset{compat=1.17}",
        *_PREVIEW_MACROS,
        "\\begin{document}",
        "\\begin{center}\\input{legend.tex}\\end{center}",
    ]
    for model, per_method in sorted(results.items()):
        preview.append(f"\\section*{{{model.replace('_', chr(92) + '_')}}}")
        preview.append("\\noindent")
        for metric in _METRICS:
            path = out_dir / f"{model}_{metric}.tex"
            path.write_text(_tikz_plot(model, per_method, metric, width, height))
            written.append(path)
            preview.append(f"\\begin{{minipage}}[b]{{0.24\\textwidth}}\\input{{{path.name}}}\\end{{minipage}}\\hfill")
        preview.append("")
    preview.append("\\end{document}")
    (out_dir / "preview.tex").write_text("\n".join(preview) + "\n")

    print(f"Wrote {len(written)} files (+ preview.tex) to {out_dir}")
    return written


# To run a particular benchmark, simply change the argument in the function call below.
# Run `python ICLR_Q4.py --plot` to only generate the figures from experiment_logs_Q4.
if __name__ == "__main__":
    if "--plot" in sys.argv[1:]:
        generate_Q4_plots()
        sys.exit(0)

    # Runs for RL rollout plots
    # Drone4D (w=8.5)
    run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--output_root", "experiment_logs_Q4/rollout_plots", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo", "--model_version","8", "--paper_figures", "--no-mc_simulations"])
    # Dubins4D (w=1.8)
    run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--output_root", "experiment_logs_Q4/rollout_plots", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","sac", "--model_version","4", "--dp_terminate_window", "5", "--dp_terminate_improvement", "0.001", "--dp_terminate_min_iterations", "10", "--paper_figures", "--no-mc_simulations"])

    # Drone4D_x0set
    drone_dp_stop = ["--dp_terminate_window", "5", "--dp_terminate_improvement", "0.001", "--dp_terminate_min_iterations", "25"]
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        v = str(v)
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--dense", "--model_version", v, *drone_dp_stop])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo", "--model_version", v, *drone_dp_stop])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo", "--model_version", v, *drone_dp_stop])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo", "--model_version", v, *drone_dp_stop])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","ppo", "--model_version", v, *drone_dp_stop])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","ppo", "--model_version", v, *drone_dp_stop])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","sac", "--model_version", v, *drone_dp_stop])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","sac", "--model_version", v, *drone_dp_stop])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","sac", "--model_version", v, *drone_dp_stop])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","sac", "--model_version", v, *drone_dp_stop])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","sac", "--model_version", v, *drone_dp_stop])

    # Dubins4D_x0set
    # Stop dynamic programming if min V[s0] improves by less than 0.001 within 5 iterations (but do at least 10 iterations)
    dubins_dp_stop = ["--dp_terminate_window", "5", "--dp_terminate_improvement", "0.001", "--dp_terminate_min_iterations", "10"]
    for v in [1, 2, 3, 4, 5, 6]:
        v = str(v)
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--dense", "--model_version", v, *dubins_dp_stop])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo", "--model_version", v, *dubins_dp_stop])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo", "--model_version",v, *dubins_dp_stop])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo", "--model_version",v, *dubins_dp_stop])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","ppo", "--model_version",v, *dubins_dp_stop])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","ppo", "--model_version",v, *dubins_dp_stop])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","sac", "--model_version",v, *dubins_dp_stop])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","sac", "--model_version",v, *dubins_dp_stop])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","sac", "--model_version",v, *dubins_dp_stop])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","sac", "--model_version",v, *dubins_dp_stop])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--output_root", "experiment_logs_Q4", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","sac", "--model_version",v, *dubins_dp_stop])