"""Grow a dynamics-based tree, then inflate noisy replays of its selected inputs."""
import json
import logging
from itertools import product
from math import prod
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from core.abstraction.partition import EPS, box2grid_idxs

# Side-experiment settings, kept out of the shared command-line parser and benchmarks.
RRT_SETTINGS = dict(iterations=2000, step_horizon=64, control_samples=16, parent_candidates=8, goal_bias=0.1,
                    time_weight=0.5, tube_trajectories=100, tube_radius=(2,), clearance=(2,))
# Radii are in grid-cell units; planning clearance is 2 cells in each dimension.
BENCHMARK_SETTINGS = {
    name: dict(tube_radius=radius, clearance=(2,) * len(radius))
    for name, radius in {
        'Dubins3D': (5, 5, 5),
        'Dubins4D': (5, 5, 5, 5),
        'Drone4D': (3, 3, 3, 3),
        'Drone4D_battery': (4, 2, 4, 2, 4),
        'Drone6D': (3, 2, 3, 2, 3, 2),
        'Drone6D_small': (2, 1, 2, 1, 2, 1),
        'Drone6D_battery': (2, 1, 2, 1, 2, 1, 1),
        'Pendulum': (5, 5),
        'MountainCar': (50, 50), 
        'CartPole': (7, 7, 7, 7),
        'DoubleIntegrator': (2, 1),
        'Test1D': (1,),
    }.items()
}


def find_active(model, args):
    config = {**RRT_SETTINGS, **BENCHMARK_SETTINGS.get(args.model, {})}
    started, rng = perf_counter(), np.random.default_rng(args.seed)
    low, high = np.asarray(model.partition['boundary'], dtype=float)
    counts = np.asarray(model.partition['number_per_dim'], dtype=int)
    span, wrap = high - low, np.asarray(model.wrap)
    width = span / counts
    radius = np.broadcast_to(np.asarray(config['tube_radius']), (model.n,))
    clearance = np.broadcast_to(np.asarray(config['clearance'] or radius / 2), (model.n,))
    if not np.all(np.isfinite([radius, clearance])) or np.any(radius <= 0) or np.any(clearance <= 0):
        raise ValueError('Tube radii and clearance must be finite and positive.')
    margin = clearance * width
    lower, upper = low + np.where(wrap, 0, margin), high - np.where(wrap, 0, margin)
    if np.any(lower >= upper):
        raise ValueError('No planning space remains. Reduce clearance in core/rrt.py.')
    obstacles = np.asarray(model.critical, dtype=float).reshape(-1, 2, model.n).copy()
    obstacles[:, 0] -= margin
    obstacles[:, 1] += margin
    goals = np.asarray(model.goal, dtype=float).reshape(-1, 2, model.n)
    first = np.maximum(0, np.ceil((goals[:, 0] - EPS / 2 - low) / width)).astype(int)
    last = np.minimum(counts - 1, np.floor((goals[:, 1] + EPS / 2 - low) / width) - 1).astype(int)
    first, last = first[np.all(first <= last, axis=1)], last[np.all(first <= last, axis=1)]
    goal_low, goal_high = np.maximum(low + first * width, lower), np.minimum(low + (last + 1) * width, upper)
    usable = np.all(goal_low < goal_high, axis=1)
    goal_low, goal_high = goal_low[usable], goal_high[usable]
    if not len(goal_low):
        raise ValueError('No abstract goal inside the planning bounds. Reduce clearance or refine the grid.')

    def difference(a, b):
        delta = a - b
        return np.where(wrap, (delta + span / 2) % span - span / 2, delta)

    def safe(x):
        x = np.atleast_2d(x).astype(float)
        x = np.where(wrap, low + (x - low) % span, x)
        valid = np.all(np.isfinite(x) & (x >= lower) & (x <= upper) & (x < high), axis=1)
        center = low + (np.floor((np.where(np.isfinite(x), x, low) - low) / width) + 0.5) * width
        for lo, hi in obstacles:
            valid &= ~np.all(np.abs(difference(center, (lo + hi) / 2)) <= (hi - lo + width) / 2 - EPS, axis=1)
        return valid

    def goal(x):
        x = np.where(wrap, low + (np.asarray(x, dtype=float) - low) % span, x)
        cell = np.floor((x - low) / width).astype(int)
        return bool(np.any(np.all((cell >= first) & (cell <= last), axis=1)))

    def sample():
        for _ in range(100):
            if rng.random() < config['goal_bias']:
                i = rng.integers(len(goal_low))
                x = rng.uniform(goal_low[i], goal_high[i])
            else:
                x = rng.uniform(lower, upper)
            if safe(x)[0]:
                return x
        raise ValueError('Could not sample a safe state. Reduce clearance in core/rrt.py.')

    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    action_grid = np.asarray(list(product(*[np.linspace(lo, hi, int(n)) for lo, hi, n in
                                           zip(model.uMin, model.uMax, model.num_actions)])), dtype=dtype)
    def rollout(x, controls, noise):
        def step(x, inputs):
            u, w = inputs
            y = jnp.asarray(model.step(x, u, w), dtype=dtype)
            y = jnp.where(jnp.asarray(wrap), jnp.asarray(low, dtype=dtype) +
                          (y - jnp.asarray(low, dtype=dtype)) % jnp.asarray(span, dtype=dtype), y)
            return y, y
        return jax.lax.scan(step, x, (controls, noise))[1]
    simulate = jax.jit(jax.vmap(rollout, in_axes=(None, 0, 0)))
    start = np.where(wrap, low + (np.asarray(model.x0) - low) % span, model.x0).astype(dtype)
    if not safe(start)[0]:
        raise ValueError('Initial state violates planning clearance. Reduce clearance in core/rrt.py.')
    nodes, parents, edges, costs = [start], [-1], [None], [0]
    best, found = (0, 0) if goal(start) else (None, None)
    iterations, horizon = config['iterations'], config['step_horizon']
    logger = logging.getLogger(__name__)
    logger.info('RRT: bounds %s to %s; clearance %s; tube %s cells', lower, upper, margin, radius)
    for iteration in range(1, (0 if best == 0 else iterations) + 1):
        target = sample()
        # Shortlist geometrically, then select the parent using reachable rollout states.
        node_array = np.asarray(nodes)
        node_costs = np.asarray(costs)
        eligible = np.flatnonzero(node_costs + 1 < costs[best]) if best is not None else np.arange(len(nodes))
        nearby = eligible[np.argsort(np.linalg.norm(difference(node_array[eligible], target) / span, axis=1))[:config['parent_candidates']]]
        actions = action_grid[rng.integers(len(action_grid), size=config['control_samples'])]
        controls = np.repeat(actions[:, None], horizon, axis=1).astype(dtype)
        candidates = []
        for parent in map(int, nearby):
            noise = np.asarray(model.noise.sample(size=controls.shape[:2], rng=rng), dtype=dtype)
            traces = np.asarray(simulate(jnp.asarray(nodes[parent]), jnp.asarray(controls), jnp.asarray(noise)))
            for trace, inputs, disturbances in zip(traces, controls, noise):
                bad = np.flatnonzero(~safe(trace))
                n = int(bad[0]) if len(bad) else horizon
                if best is not None:
                    n = min(n, costs[best] - costs[parent] - 1)
                if not n:
                    continue
                reached = next((i + 1 for i, x in enumerate(trace[:n]) if goal(x)), None)
                distances = np.linalg.norm(difference(trace[:n], target) / span, axis=1)
                # Ignore stationary prefixes when choosing the closest endpoint.
                distances[np.linalg.norm(difference(trace[:n], nodes[parent]) / span, axis=1) <= 1e-8] = np.inf
                # Penalize elapsed time both within a rollout and across parent branches.
                scores = distances + config['time_weight'] * (costs[parent] + np.arange(1, n + 1)) / horizon
                n = reached or int(np.argmin(scores)) + 1
                if np.isfinite(scores[n - 1]):
                    score = (0, costs[parent] + n) if reached else (1, scores[n - 1])
                    candidates.append((score, parent, trace[:n], inputs[:n], disturbances[:n]))
        if candidates:
            _, parent, path, inputs, disturbances = min(candidates, key=lambda c: c[0])
            duplicate = np.linalg.norm(difference(node_array, path[-1]) / span, axis=1) <= 1e-8
            if not np.any(duplicate & (node_costs <= costs[parent] + len(inputs))):
                nodes.append(path[-1].copy())
                parents.append(parent)
                edges.append((path.copy(), inputs.copy(), disturbances.copy()))
                costs.append(costs[parent] + len(inputs))
                if goal(path[-1]) and (best is None or costs[-1] < costs[best]):
                    best, found = len(nodes) - 1, found if found is not None else iteration
        if iteration % 100 == 0:
            logger.info('RRT: %d/%d, nodes=%d, best steps=%s', iteration, iterations,
                        len(nodes), None if best is None else costs[best])
    if best is None:
        raise RuntimeError('RRT found no goal-reaching path. Adjust iterations or clearance in core/rrt.py.')
    chain, cursor = [], best
    while parents[cursor] >= 0:
        chain.append(edges[cursor])
        cursor = parents[cursor]
    chain.reverse()
    path = np.concatenate([start[None]] + [edge[0] for edge in chain])
    controls = np.concatenate([edge[1] for edge in chain]) if chain else np.empty((0, model.p))
    noise = np.concatenate([edge[2] for edge in chain]) if chain else np.empty((0, model.n))
    metrics = dict(planner='rrt', planner_seed=args.seed, planner_iterations=0 if best == 0 else iterations,
                   planner_dynamics='sampled_noise',
                   planner_nodes=len(nodes), planner_first_solution_iteration=found,
                   planner_path_steps=len(controls), planner_path_time=float(len(controls) * model.tau),
                   planner_obstacle_margin=margin.tolist(), planner_boundary=[lower.tolist(), upper.tolist()],
                   time_planning=perf_counter() - started, planner_cost='elapsed_time')
    started = perf_counter()
    # Replay the same open-loop controls with independent noise, retaining each
    # collision-free prefix rather than discarding an entire failed trajectory.
    count = config['tube_trajectories']
    trajectory_noise = np.asarray(model.noise.sample(size=(count, len(controls)), rng=rng), dtype=dtype)
    trajectories = np.broadcast_to(start, (count, len(controls) + 1, model.n)).copy()
    if len(controls):
        replay = jax.jit(jax.vmap(rollout, in_axes=(None, None, 0)))
        trajectories[:, 1:] = np.asarray(replay(jnp.asarray(start), jnp.asarray(controls), jnp.asarray(trajectory_noise)))
    # Use the actual obstacles, without the RRT's extra planning clearance.
    hit = np.zeros(trajectories.shape[:2], dtype=bool)
    for lo, hi in np.asarray(model.critical).reshape(-1, 2, model.n):
        hit |= np.all((trajectories >= lo) & (trajectories <= hi), axis=-1)
    stopped = np.logical_or.accumulate(hit, axis=1)
    trajectory_lengths = np.sum(~stopped, axis=1)
    trajectories[stopped] = np.nan
    metrics.update(time_replays=perf_counter() - started, tube_trajectories=count,
                   replay_obstacle_hits=int(stopped[:, -1].sum()))
    logger.info('RRT: sampled %d noisy trajectories using the same %d inputs in %.3fs',
                count, len(controls), metrics['time_replays'])
    logger.info('RRT: truncated %d replays at their first obstacle hit', metrics['replay_obstacle_hits'])
    logger.info('RRT: building tube around reference and %d noisy input replays', count)
    started = perf_counter()
    active = build_tube(np.concatenate((path[None], trajectories)), model, radius)
    metrics.update(time_tube=perf_counter() - started, tube_states=len(active), tube_radius_cells=radius.tolist())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'rrt_reference.npz', states=path, controls=controls, noise=noise, active_states=active,
                        trajectories=trajectories, trajectory_noise=trajectory_noise, trajectory_lengths=trajectory_lengths,
                        tube_radius_cells=radius, step_duration=float(model.tau), seed=args.seed)
    config.update(tube_radius=radius.tolist(), clearance=clearance.tolist())
    (output / 'rrt_plan.json').write_text(json.dumps(dict(config=config, metrics=metrics), indent=2))
    return active, dict.fromkeys(map(tuple, active), action_grid), path, trajectories, metrics


def _distance(a, b, low, high):
    """Exact squared segment-to-box distance, minimized between box-face crossings."""
    v = b - a
    faces, slopes = np.concatenate((low - a, high - a), axis=1), np.tile(v, 2)
    crossings = np.divide(faces, slopes, out=np.zeros_like(faces), where=slopes != 0)
    cuts = np.sort(np.c_[np.zeros(len(low)), np.clip(crossings, 0, 1), np.ones(len(low))], axis=1)
    left, right = cuts[:, :-1], cuts[:, 1:]
    mid = a + ((left + right) / 2)[..., None] * v
    below, above = mid < low[:, None], mid > high[:, None]
    c = (a - np.where(below, low[:, None], high[:, None])) * (below | above)
    d = v * (below | above)
    t = np.divide(-(c * d).sum(-1), (d * d).sum(-1), out=left.copy(), where=(d * d).sum(-1) > 0)
    points = a + np.clip(t, left, right)[..., None] * v
    error = np.maximum(np.maximum(low[:, None] - points, points - high[:, None]), 0)
    return (error * error).sum(-1).min(-1)


def build_tube(path, model, radius):
    """Union of swept ellipsoids for one path or a batch, including initial sets."""
    low, high = np.asarray(model.partition['boundary'], dtype=float)
    counts, wrap = np.asarray(model.partition['number_per_dim']), np.asarray(model.wrap)
    paths = np.asarray(path)
    if paths.ndim == 2:
        paths = paths[None]
    grid = (paths - low) / ((high - low) / counts)
    grid = np.where(wrap, grid % counts, grid)
    # Store linear cell IDs so repeated replays can skip cells already covered.
    active = set()
    ends = np.concatenate((grid[:, 1:], grid[:, -1:]), axis=1)
    # A NaN-padded trajectory still contributes its last collision-free point.
    ends = np.where(np.all(np.isfinite(ends), axis=-1, keepdims=True), ends, grid)
    starts, ends = grid.reshape(-1, model.n), ends.reshape(-1, model.n)
    logger = logging.getLogger(__name__)
    began = last_log = perf_counter()
    for segment, (start, end) in enumerate(zip(starts, ends), 1):
        if perf_counter() - last_log >= 5:
            logger.info('RRT tube: segment %d/%d, %d cells, %.1fs',
                        segment, len(starts), len(active), perf_counter() - began)
            last_log = perf_counter()
        if not np.all(np.isfinite([start, end])):
            continue
        delta = np.where(wrap, (end - start + counts / 2) % counts - counts / 2, end - start)
        # Only the part within one radius of the grid can contribute cells.
        # Clip the segment parametrically; clamping endpoints would distort it.
        moving = (delta != 0) & ~wrap
        if np.any(~wrap & ~moving & ((start < -radius) | (start > counts + radius))):
            continue
        near = np.divide(-radius - start, delta, out=np.full(model.n, -np.inf), where=moving)
        far = np.divide(counts + radius - start, delta, out=np.full(model.n, np.inf), where=moving)
        enter, leave = max(0.0, np.minimum(near, far).max()), min(1.0, np.maximum(near, far).min())
        if enter > leave:
            continue
        start, delta = start + enter * delta, (leave - enter) * delta
        # Radius-sized pieces avoid rechecking almost identical neighborhoods.
        pieces = max(1, int(np.ceil(np.max(np.abs(delta) / np.maximum(radius, 1)))))
        for i in range(pieces):
            a, b = start + delta * i / pieces, start + delta * (i + 1) / pieces
            first = np.ceil(np.minimum(a, b) - radius).astype(int) - 1
            last = np.floor(np.maximum(a, b) + radius).astype(int)
            first, last = np.where(wrap, first, np.maximum(first, 0)), np.where(wrap, last, np.minimum(last, counts - 1))
            shape = last - first + 1
            if np.any(shape <= 0):
                continue
            size = prod(int(n) for n in shape)
            for offset in range(0, size, 8192):
                cells = np.stack(np.unravel_index(np.arange(offset, min(size, offset + 8192)), tuple(shape)), axis=-1) + first
                ids = np.ravel_multi_index(np.where(wrap, cells % counts, cells).T, tuple(counts))
                unseen = np.fromiter((int(key) not in active for key in ids), dtype=bool, count=len(ids))
                cells, ids = cells[unseen], ids[unseen]
                if not len(cells):
                    continue
                # Distance to the segment's bounding box is a cheap lower bound.
                gap = np.maximum(np.maximum(cells - np.maximum(a, b), np.minimum(a, b) - (cells + 1)), 0) / radius
                possible = (gap * gap).sum(axis=1) <= 1 + 1e-10
                cells, ids = cells[possible], ids[possible]
                if not len(cells):
                    continue
                # Most interior cells already intersect an endpoint ellipsoid.
                gap_a = np.maximum(np.maximum(cells - a, a - (cells + 1)), 0) / radius
                gap_b = np.maximum(np.maximum(cells - b, b - (cells + 1)), 0) / radius
                hit = np.minimum((gap_a * gap_a).sum(axis=1), (gap_b * gap_b).sum(axis=1)) <= 1 + 1e-10
                remaining = ~hit
                if np.any(remaining):
                    hit[remaining] = _distance(a / radius, b / radius, cells[remaining] / radius,
                                               (cells[remaining] + 1) / radius) <= 1 + 1e-10
                active.update(map(int, ids[hit]))
    if getattr(model, 'x0_box', None) is not None:
        cells = box2grid_idxs(model.x0_box, [low, high], counts)
        active.update(map(int, np.ravel_multi_index(cells.T, tuple(counts))))
    logger.info('RRT tube: finished %d segments, %d cells in %.1fs', len(starts), len(active), perf_counter() - began)
    return np.stack(np.unravel_index(np.asarray(sorted(active), dtype=np.int64), tuple(counts)), axis=-1)


def plot_plan(args, model, partition, reference, traces, trajectories=None):
    from matplotlib import pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    from core.plotting.utils import plot_boxes, save_fig
    dims = list(model.plot_dimensions)[:2]
    if len(dims) != 2:
        return
    low, high = np.asarray(partition.boundary_lb)[dims], np.asarray(partition.boundary_ub)[dims]
    width = np.asarray(partition.cell_width)[dims]
    cells = np.unique(np.asarray(partition.region_idx_inv)[:, dims], axis=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.add_collection(PatchCollection([Rectangle(low + c * width, *width) for c in cells],
                                     facecolor='cornflowerblue', alpha=0.35, rasterized=True))
    plot_boxes(ax, args, model, plot_dimensions=dims)
    paths = []
    if trajectories is not None:
        paths.extend((path, 'seagreen', 0.2, 'Noisy input replays' if i == 0 else None)
                     for i, path in enumerate(trajectories))
    paths.extend((t['x'], 'darkorange', 0.4, 'Synthesized policy' if i == 0 else None)
                 for i, t in enumerate(list(traces.values())[:30]))
    paths.append((reference, 'black', 1, 'RRT reference'))
    for path, color, alpha, label in paths:
        if not len(path):
            continue
        points = np.asarray(path, dtype=float)[:, dims].copy()
        seams = np.any(np.asarray(model.wrap)[dims] & (np.abs(np.diff(points, axis=0)) > (high - low) / 2), axis=1)
        # Insert gaps at periodic seams without dropping any sample markers.
        line = np.insert(points, np.flatnonzero(seams) + 1, np.nan, axis=0)
        ax.plot(*line.T, color=color, alpha=alpha, label=label)
        ax.scatter(*points.T, color=color, alpha=alpha, s=12 if color == 'black' else 3,
                   zorder=4 if color == 'black' else 2, rasterized=True)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel=model.state_variables[dims[0]], ylabel=model.state_variables[dims[1]])
    if not args.plot_ticks:
        ax.set(xticks=[], yticks=[])
    if args.plot_title:
        ax.set_title('RRT reference, noisy input replays and tube')
    ax.legend()
    save_fig(fig, args.output_dir / 'rrt_trajectory_tube')
