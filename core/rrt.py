"""Grow a dynamics-based tree, then inflate its selected path."""
import json
import logging
import os
from functools import partial
from itertools import product
from math import prod
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from core.abstraction.partition import EPS, box2grid_idxs
from core.plotting.utils import plot_boxes, save_fig

logger = logging.getLogger(__name__)


def _plot_path(ax, path, dims, low, high, wrap, color, alpha, label):
    points = np.asarray(path, dtype=float)[:, dims]
    seams = np.any(wrap & (np.abs(np.diff(points, axis=0)) > (high - low) / 2), axis=1)
    line = np.insert(points, np.flatnonzero(seams) + 1, np.nan, axis=0)
    ax.plot(*line.T, color=color, alpha=alpha, label=label)
    ax.scatter(*points.T, color=color, alpha=alpha, s=12 if color == 'black' else 3,
               zorder=4 if color == 'black' else 2, rasterized=True)


def _plot_tree(args, model, nodes, segments, reference):
    """Plot every accepted branch using its simulated states, including node markers."""
    dims = model.plot_dimensions[:2]
    low, high = np.asarray(model.partition['boundary'])[:, dims]
    wrap = np.asarray(model.wrap)[dims]
    projected = segments[:, :, dims]
    seams = np.any(wrap & (np.abs(projected[:, 1] - projected[:, 0]) > (high - low) / 2), axis=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_boxes(ax, args, model, plot_dimensions=dims)
    ax.add_collection(LineCollection(projected[~seams], colors='steelblue', linewidths=0.6,
                                    alpha=0.35, label='RRT tree', rasterized=True))
    ax.scatter(*nodes[:, dims].T, color='steelblue', s=5, alpha=0.5, rasterized=True)
    _plot_path(ax, reference, dims, low, high, wrap, 'black', 1, 'Selected path')
    ax.scatter(*nodes[0, dims], color='darkorange', marker='*', s=100, zorder=5, label='Start')
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
           xlabel=model.state_variables[dims[0]], ylabel=model.state_variables[dims[1]])
    if not args.plot_ticks:
        ax.set(xticks=[], yticks=[])
    if args.plot_title:
        ax.set_title('Full RRT tree and selected path')
    ax.legend()
    save_fig(fig, Path(args.output_dir) / 'rrt_tree')


def _difference(a, b, *, wrap, span, xp=np):
    delta = a - b
    return xp.where(wrap, (delta + span / 2) % span - span / 2, delta)


def _safe(x, *, low, high, width, wrap, lower, upper, obstacles, xp=np):
    x = xp.atleast_2d(x).astype(float)
    span = high - low
    x = xp.where(wrap, low + (x - low) % span, x)
    valid = xp.all(xp.isfinite(x) & (x >= lower) & (x <= upper) & (x < high), axis=-1)
    center = low + (xp.floor((xp.where(xp.isfinite(x), x, low) - low) / width) + 0.5) * width
    for lo, hi in obstacles:
        delta = _difference(center, (lo + hi) / 2, wrap=wrap, span=span, xp=xp)
        valid &= ~xp.all(xp.abs(delta) <= (hi - lo + width) / 2 - EPS, axis=-1)
    return valid


def _goal(x, *, low, width, span, wrap, first, last, xp=np):
    x = xp.where(wrap, low + (xp.asarray(x, dtype=float) - low) % span, x)
    cell = xp.floor((x - low) / width).astype(int)
    return xp.any(xp.all((cell[..., None, :] >= first) & (cell[..., None, :] <= last), axis=-1), axis=-1)


def _sample(rng, goal_bias, goal_low, goal_high, lower, upper, safe):
    for _ in range(100):
        if rng.random() < goal_bias:
            i = rng.integers(len(goal_low))
            x = rng.uniform(goal_low[i], goal_high[i])
        else:
            x = rng.uniform(lower, upper)
        if safe(x)[0]:
            return x
    raise ValueError('Could not sample a safe state. Reduce clearance in the run settings.')


def _step(x, inputs, *, model, low, span, wrap, dtype):
    u, w = inputs
    y = jnp.asarray(model.step(x, u, w), dtype=dtype)
    y = jnp.where(jnp.asarray(wrap), jnp.asarray(low, dtype=dtype) +
                  (y - jnp.asarray(low, dtype=dtype)) % jnp.asarray(span, dtype=dtype), y)
    return y, y


def _rollout(x, controls, noise, *, model, low, span, wrap, dtype):
    step = partial(_step, model=model, low=low, span=span, wrap=wrap, dtype=dtype)
    return jax.lax.scan(step, x, (controls, noise))[1]


def _select_candidate(traces, starts, costs, target, best_cost, parent_count, *,
                      safe, goal, difference, span, time_weight):
    """Select the first minimum in parent/action order, using only valid prefixes."""
    horizon = traces.shape[2]
    steps = jnp.arange(1, horizon + 1)
    valid = jnp.cumprod(safe(traces, xp=jnp), axis=-1).astype(bool)
    valid &= costs[:, None, None] + steps < best_cost
    valid &= jnp.arange(len(starts))[:, None, None] < parent_count
    reached = goal(traces, xp=jnp) & valid
    has_goal = jnp.any(reached, axis=-1)
    distances = jnp.linalg.norm(difference(traces, target, xp=jnp) / span, axis=-1)
    stationary = jnp.linalg.norm(difference(traces, starts[:, None, None], xp=jnp) / span, axis=-1) <= 1e-8
    scores = distances + time_weight * (costs[:, None, None] + steps) / horizon
    scores = jnp.where(valid & ~stationary, scores, jnp.inf)
    endpoint = jnp.where(has_goal, jnp.argmax(reached, axis=-1), jnp.argmin(scores, axis=-1))
    selected_score = jnp.take_along_axis(scores, endpoint[..., None], axis=-1)[..., 0]
    usable = jnp.isfinite(selected_score)
    # Goal-reaching candidates always beat other candidates; ties keep traversal order.
    goal_candidate = has_goal & usable
    rank = jnp.where(jnp.any(goal_candidate),
                     jnp.where(goal_candidate, costs[:, None] + endpoint + 1, jnp.inf),
                     jnp.where(usable, selected_score, jnp.inf))
    winner = jnp.argmin(rank)
    parent, action = winner // traces.shape[1], winner % traces.shape[1]
    length = jnp.where(usable[parent, action], endpoint[parent, action] + 1, 0)
    return parent, action, length, traces[parent, action]


def _candidate_evaluator(rollout, *, safe, goal, difference, span, time_weight):
    """Fuse simulation and selection, keeping dynamics at the configured precision."""
    simulation_x64 = jax.config.x64_enabled
    simulate = jax.vmap(jax.vmap(rollout, in_axes=(None, 0, 0)), in_axes=(0, None, 0))

    @jax.jit
    def evaluate(starts, controls, noise, costs, target, best_cost, parent_count):
        with jax.enable_x64(simulation_x64):
            traces = simulate(starts, controls, noise)
        # The original NumPy geometry and scores use float64 even with float32 dynamics.
        with jax.enable_x64():
            return _select_candidate(traces, starts, costs, target, best_cost, parent_count,
                                     safe=safe, goal=goal, difference=difference,
                                     span=span, time_weight=time_weight)

    return evaluate


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
    paths = paths.reshape((-1,) + paths.shape[-2:])
    grid = (paths - low) / ((high - low) / counts)
    grid = np.where(wrap, grid % counts, grid)
    # Store linear cell IDs so overlapping segments can skip cells already covered.
    active = set()
    ends = np.concatenate((grid[:, 1:], grid[:, -1:]), axis=1)
    # A NaN-padded trajectory still contributes its last collision-free point.
    ends = np.where(np.all(np.isfinite(ends), axis=-1, keepdims=True), ends, grid)
    starts, ends = grid.reshape(-1, model.n), ends.reshape(-1, model.n)
    valid = np.all(np.isfinite(starts), axis=1)
    starts, ends = starts[valid], ends[valid]
    began = last_log = perf_counter()
    for segment, (start, end) in enumerate(zip(starts, ends), 1):
        if perf_counter() - last_log >= 5:
            logger.info('RRT tube: segment %d/%d, %d cells, %.1fs',
                        segment, len(starts), len(active), perf_counter() - began)
            last_log = perf_counter()
        delta = np.where(wrap, (end - start + counts / 2) % counts - counts / 2, end - start)
        # Radius-sized pieces avoid rechecking almost identical neighborhoods.
        pieces = max(1, int(np.ceil(np.max(np.abs(delta) / np.maximum(radius, 1)))))
        for i in range(pieces):
            a, b = start + delta * i / pieces, start + delta * (i + 1) / pieces
            first = np.ceil(np.minimum(a, b) - radius).astype(int) - 1
            last = np.floor(np.maximum(a, b) + radius).astype(int)
            first, last = np.where(wrap, first, np.maximum(first, 0)), np.where(wrap, last, np.minimum(last, counts - 1))
            shape = np.maximum(last - first + 1, 0)
            size = prod(int(n) for n in shape)
            for offset in range(0, size, 8192):
                cells = np.stack(np.unravel_index(np.arange(offset, min(size, offset + 8192)), tuple(shape)), axis=-1) + first
                ids = np.ravel_multi_index(np.where(wrap, cells % counts, cells).T, tuple(counts))
                unseen = np.fromiter((int(key) not in active for key in ids), dtype=bool, count=len(ids))
                cells, ids = cells[unseen], ids[unseen]
                # Distance to the segment's bounding box is a cheap lower bound.
                gap = np.maximum(np.maximum(cells - np.maximum(a, b), np.minimum(a, b) - (cells + 1)), 0) / radius
                possible = (gap * gap).sum(axis=1) <= 1 + 1e-10
                cells, ids = cells[possible], ids[possible]
                # Most interior cells already intersect an endpoint ellipsoid.
                gap_a = np.maximum(np.maximum(cells - a, a - (cells + 1)), 0) / radius
                gap_b = np.maximum(np.maximum(cells - b, b - (cells + 1)), 0) / radius
                hit = np.minimum((gap_a * gap_a).sum(axis=1), (gap_b * gap_b).sum(axis=1)) <= 1 + 1e-10
                remaining = ~hit
                hit[remaining] = _distance(a / radius, b / radius, cells[remaining] / radius,
                                           (cells[remaining] + 1) / radius) <= 1 + 1e-10
                active.update(map(int, ids[hit]))
    if getattr(model, 'x0_box', None) is not None:
        cells = box2grid_idxs(model.x0_box, [low, high], counts)
        active.update(map(int, np.ravel_multi_index(cells.T, tuple(counts))))
    logger.info('RRT tube: finished %d segments, %d cells in %.1fs', len(starts), len(active), perf_counter() - began)
    return np.stack(np.unravel_index(np.asarray(sorted(active), dtype=np.int64), tuple(counts)), axis=-1)


def find_active(model, args, config=None):
    # The launcher passes settings only to this subprocess, without extra CLI flags.
    config = dict(iterations=2000, step_horizon=64, control_samples=16, parent_candidates=8,
                  goal_bias=0.1, time_weight=0.5, tube_radius=(2,), clearance=(2,)) \
             | json.loads(os.environ.get('IMDP_RRT_SETTINGS', '{}')) | (config or {})
    started, rng = perf_counter(), np.random.default_rng(args.seed)
    low, high = np.asarray(model.partition['boundary'], dtype=float)
    counts = np.asarray(model.partition['number_per_dim'], dtype=int)
    span, wrap = high - low, np.asarray(model.wrap)
    width = span / counts
    radius = np.broadcast_to(np.asarray(config['tube_radius']), (model.n,))
    clearance = np.broadcast_to(np.asarray(config['clearance']), (model.n,))
    margin = clearance * width
    lower, upper = low + np.where(wrap, 0, margin), high - np.where(wrap, 0, margin)
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

    difference = partial(_difference, wrap=wrap, span=span)
    safe = partial(_safe, low=low, high=high, width=width, wrap=wrap,
                   lower=lower, upper=upper, obstacles=obstacles)
    goal = partial(_goal, low=low, width=width, span=span, wrap=wrap, first=first, last=last)
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    action_grid = np.asarray(list(product(*[np.linspace(lo, hi, int(n)) for lo, hi, n in
                                           zip(model.uMin, model.uMax, model.num_actions)])), dtype=dtype)
    rollout = partial(_rollout, model=model, low=low, span=span, wrap=wrap, dtype=dtype)
    evaluate = _candidate_evaluator(rollout, safe=safe, goal=goal, difference=difference,
                                   span=span, time_weight=config['time_weight'])
    start = np.where(wrap, low + (np.asarray(model.x0) - low) % span, model.x0).astype(dtype)
    iterations, horizon = config['iterations'], config['step_horizon']
    nodes = np.empty((iterations + 1, model.n), dtype=dtype)
    costs = np.empty(iterations + 1, dtype=np.int64)
    nodes[0], costs[0] = start, 0
    parents, edges = [-1], [None]
    best, found = (0, 0) if goal(start) else (None, None)
    parent_limit = config['parent_candidates']
    batch_noise = np.zeros((parent_limit, config['control_samples'], horizon, model.n), dtype=dtype)
    batch_parents = np.zeros(parent_limit, dtype=int)
    logger.info('RRT: bounds %s to %s; clearance %s; tube %s cells', lower, upper, margin, radius)
    for iteration in range(1, (0 if best == 0 else iterations) + 1):
        target = _sample(rng, config['goal_bias'], goal_low, goal_high, lower, upper, safe)
        # Shortlist geometrically, then select the parent using reachable rollout states.
        node_array, node_costs = nodes[:len(parents)], costs[:len(parents)]
        eligible = np.flatnonzero(node_costs + 1 < costs[best]) if best is not None else np.arange(len(parents))
        nearby = eligible[np.argsort(np.linalg.norm(difference(node_array[eligible], target) / span, axis=1))[:parent_limit]]
        actions = action_grid[rng.integers(len(action_grid), size=config['control_samples'])]
        controls = np.repeat(actions[:, None], horizon, axis=1).astype(dtype)
        # Pad early/small shortlists so their shape does not trigger recompilation.
        batch_parents[:len(nearby)] = nearby
        for i in range(len(nearby)):
            # Preserve the original per-parent RNG calls, including distribution semantics.
            batch_noise[i] = model.noise.sample(size=controls.shape[:2], rng=rng)
        if len(nearby):
            with jax.enable_x64():
                parent_slot, action, length, trace = jax.device_get(evaluate(
                    nodes[batch_parents], controls, batch_noise, costs[batch_parents], target,
                    int(costs[best]) if best is not None else iterations * horizon + 1, len(nearby)))
            parent, action, length = int(batch_parents[parent_slot]), int(action), int(length)
            if length:
                path = trace[:length]
                inputs = controls[action, :length]
                disturbances = batch_noise[parent_slot, action, :length]
                duplicate = np.linalg.norm(difference(node_array, path[-1]) / span, axis=1) <= 1e-8
                if not np.any(duplicate & (node_costs <= costs[parent] + length)):
                    node = len(parents)
                    nodes[node], costs[node] = path[-1], costs[parent] + length
                    parents.append(parent)
                    edges.append((path.copy(), inputs.copy(), disturbances.copy()))
                    if goal(path[-1]) and (best is None or costs[node] < costs[best]):
                        best, found = node, found if found is not None else iteration
        if iteration % 100 == 0:
            logger.info('RRT: %d/%d, nodes=%d, best steps=%s', iteration, iterations,
                        len(parents), None if best is None else costs[best])
    if best is None:
        raise RuntimeError('RRT found no goal-reaching path. Adjust iterations or clearance in the run settings.')
    nodes, costs = nodes[:len(parents)], costs[:len(parents)]
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
    # Each edge includes the parent-to-first-state segment and every rollout step.
    tree_segments = np.concatenate([np.empty((0, 2, model.n))] + [
        np.stack((np.vstack((nodes[parent], edge[0][:-1])), edge[0]), axis=1)
        for parent, edge in zip(parents[1:], edges[1:])
    ])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'rrt_tree.npz', nodes=np.asarray(nodes), parents=parents,
                        segments=tree_segments, costs=costs, reference=path)
    _plot_tree(args, model, np.asarray(nodes), tree_segments, path)
    logger.info('RRT: saved full tree with %d nodes to %s', len(nodes), output / 'rrt_tree.png')
    logger.info('RRT: building tube around the selected path')
    started = perf_counter()
    active = build_tube(path, model, radius)
    metrics.update(time_tube=perf_counter() - started, tube_states=len(active), tube_radius_cells=radius.tolist())
    np.savez_compressed(output / 'rrt_reference.npz', states=path, controls=controls, noise=noise, active_states=active,
                        tube_radius_cells=radius, step_duration=float(model.tau), seed=args.seed)
    config.update(tube_radius=radius.tolist(), clearance=clearance.tolist())
    (output / 'rrt_plan.json').write_text(json.dumps(dict(config=config, metrics=metrics), indent=2))
    return active, dict.fromkeys(map(tuple, active), action_grid), path, metrics
