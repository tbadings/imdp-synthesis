#!/usr/bin/env python3
"""Generate only a LaTeX tabular in fixed_budget_table.tex from adjacent CSVs and logs.

Usage (from any working directory):
    python3 /path/to/output/Fixed-budget/create_table.py

Uses only the Python standard library. Run order follows
Run_SVMDP_fixed-budget.py: default actions, then fixed actions for each
benchmark; the first Drone4D pair has no damping, the second has damping.
The logs do not record damping or the action-grid override, so these labels
rely on that launcher order. Unexpected run counts are rejected rather than
silently assigning variants. Both runs are retained even when identical.

Sizes use the template's scientific notation; percentages use one decimal.
Exact values and source directories are preserved in LaTeX comments.
"""

import csv
import math
import re
from pathlib import Path


# Model, index of its first run, LaTeX benchmark label, state/action dimensions.
BENCHMARKS = [
    ("MountainCar", 0, r"\textbf{MountainCar}", 2, 1),
    ("Pendulum", 0, r"\textbf{Pendulum}", 2, 1),
    ("Dubins3D", 0, r"\textbf{Dubins3D}", 3, 2),
    ("CartPole", 0, r"\textbf{CartPole}", 4, 1),
    ("Dubins4D", 0, r"\textbf{Dubins4D}", 4, 2),
    ("Drone4D", 0, r"\textbf{Drone4D}", 4, 2),
    ("Drone4D", 2, r"\shortstack[l]{\textbf{Drone4D}\\(+damping)}", 4, 2),
    ("Drone4D_battery", 0, r"\shortstack[l]{\textbf{Drone4D}\\(+bat+damp)}", 5, 2),
    ("Drone6D", 0, r"\textbf{Drone6D}", 6, 3),
]

HEADER = r"""\begin{tabular}{llll rrr rr}
\toprule
\multirow{2}{*}{\textbf{Benchmark}} & \multirow{2}{*}{$d$} & \multirow{2}{*}{$m$} & \multirow{2}{*}{\textbf{Method}} & \multicolumn{3}{c}{\textbf{Abstraction size}} & \multicolumn{2}{c}{\textbf{Reach-avoid probability}} \\
\cmidrule(lr){5-7} \cmidrule(lr){8-9}
& & & & States & Actions per state & State-actions & Optimal value $\satprob^\star_{s_0}$ & Empirical satprob \\
\midrule
"""


def read_run(summary):
    with summary.open(newline="") as stream:
        reader = csv.reader(stream)
        if next(reader) != ["metric", "value"]:
            raise ValueError(f"Unexpected CSV header: {summary}")
        metrics = dict(reader)
    logs = list(summary.parent.glob("run_*.log"))
    if len(logs) != 1:
        raise ValueError(f"Expected one run log in {summary.parent}")
    log = logs[0].read_text()

    def match(pattern):
        found = re.search(pattern, log)
        if found is None:
            raise ValueError(f"Missing log field {pattern!r}: {logs[0]}")
        return found[1]

    model = match(r"model=(\w+)")
    if "Using DensePartition" not in log:
        raise ValueError(f"Expected a dense abstraction: {summary.parent}")
    states = int(metrics["abstraction_states"])
    actions = int(metrics["abstraction_actions"])
    choices = int(metrics["abstraction_state-actions"])
    optimal = float(metrics["optimal_value"])
    empirical = float(metrics["empirical_satprob"])
    checks = [
        states > 0 and actions > 0,
        states * actions == choices,
        states == int(match(r"- Number of states: (\d+)")),
        actions == int(match(r"Max number of actions per state: (\d+)")),
        choices == int(match(r"Total number of choices: (\d+)")),
        0 <= optimal <= 1 and 0 <= empirical <= 1,
        math.isclose(optimal, float(match(r"Value in initial state .*?: ([\d.eE+-]+)")),
                     rel_tol=0, abs_tol=1e-6),
        empirical == float(match(r"Empirical satisfaction probability: ([\d.eE+-]+)")),
    ]
    if not all(checks):
        raise ValueError(f"CSV/log mismatch or invalid metrics: {summary}")
    return model, (summary.parent.name, states, actions, choices, optimal, empirical)


def scientific(value):
    exponent = int(math.log10(value))
    return f"${value / 10**exponent:.1f} \\times 10^{{{exponent}}}$"


def percentage(value):
    return f"${value * 100:.1f}\\%$"


def create_table(root):
    runs = {}
    for summary in sorted(root.glob("*/summary.csv")):
        model, row = read_run(summary)
        runs.setdefault(model, []).append(row)
    expected = {model: 4 if model == "Drone4D" else 2
                for model, *_ in BENCHMARKS}
    actual = {model: len(rows) for model, rows in runs.items()}
    if actual != expected:
        raise ValueError(f"Expected run counts {expected}; found {actual}")

    parts = [HEADER]
    for index, (model, offset, label, d, m) in enumerate(BENCHMARKS):
        parts.append(
            f"\\multirow{{2}}{{*}}{{{label}}} & \\multirow{{2}}{{*}}{{{d}}}"
            f" & \\multirow{{2}}{{*}}{{{m}}}\n"
        )
        for variant, row in enumerate(runs[model][offset:offset + 2]):
            source, states, actions, choices, optimal, empirical = row
            parts.append(
                f"% Source: {source}; exact states={states}, actions={actions}, "
                f"state-actions={choices}, optimal={optimal}, empirical={empirical}\n"
            )
            method = "default" if variant == 0 else "fixed"
            prefix = " & " if variant == 0 else " & & & "
            parts.append(
                prefix + f"Dense ({method} actions) & {scientific(states)} & ${actions}$"
                f" & {scientific(choices)} & {percentage(optimal)} & {percentage(empirical)}"
                + r" \\" + "\n"
            )
        if index < len(BENCHMARKS) - 1:
            parts.append("\\midrule\n")
    parts.append("\\bottomrule\n\\end{tabular}%\n")
    output = root / "fixed_budget_table.tex"
    output.write_text("".join(parts))
    return output


if __name__ == "__main__":
    try:
        output = create_table(Path(__file__).resolve().parent)
    except (ValueError, KeyError, OSError, StopIteration) as error:
        raise SystemExit(f"Cannot generate table: {error}") from error
    print(f"Created {output} (18 runs validated against logs).")
