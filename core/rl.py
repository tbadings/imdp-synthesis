import logging
import multiprocessing
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import time

import gymnasium as gym
from gymnasium import spaces
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from tqdm import tqdm
import itertools
import benchmarks

from core.abstraction.partition import _compute_linear_strides

logger = logging.getLogger(__name__)

# Candidate neighbors held per batch when inflating visited cells, which bounds peak memory there.
INFLATE_IDS_PER_BATCH = 1 << 23

@dataclass
class RLConfig:
    max_steps: int
    goal_reward: float
    unsafe_penalty: float
    out_of_bounds_penalty: float
    revisit_penalty: float
    distance_reward: float = 0.0
    per_step_reward: float = 0.0

class BenchmarkRLEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, model, cfg: RLConfig, previous_cells=None):
        super().__init__()
        self.model = model
        self.cfg = cfg

        boundary = np.asarray(self.model.partition["boundary"], dtype=np.float32)
        self.obs_low = boundary[0]
        self.obs_high = boundary[1]

        self.observation_space = spaces.Box(self.obs_low, self.obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.asarray(self.model.uMin, dtype=np.float32),
            high=np.asarray(self.model.uMax, dtype=np.float32),
            dtype=np.float32,
        )

        self.bin_widths = (self.obs_high - self.obs_low) / self.model.partition['number_per_dim']
        self.goal = np.asarray(getattr(self.model, "goal", np.empty((0, 2, self.model.n))), dtype=np.float32)
        self.critical = np.asarray(
            getattr(self.model, "critical", np.empty((0, 2, self.model.n))), dtype=np.float32
        )

        self.previous_cells = set() if previous_cells is None else set(previous_cells)
        self.state = None
        self.steps = 0
        self.prev_dist = None

    def set_previous_cells(self, previous_cells):
        self.previous_cells = set(previous_cells)

    def state_to_cell(self, obs):
        indices = np.floor((np.asarray(obs, dtype=np.float64) - self.obs_low) / self.bin_widths).astype(int)
        return tuple(np.clip(indices, 0, self.model.partition['number_per_dim'] - 1).tolist())

    def _in_boxes(self, state, boxes, inflate=0):
        if boxes.size == 0:
            return False
        mins = boxes[:, 0, :]
        maxs = boxes[:, 1, :]
        return bool(np.any(np.all((state >= mins - inflate) & (state <= maxs + inflate), axis=1)))

    def _goal_center(self):
        if self.goal.size == 0:
            return None
        first_goal = self.goal[0]
        return 0.5 * (first_goal[0] + first_goal[1])

    def _distance_reward(self, state):
        center = self._goal_center()
        if center is None:
            return 0.0
        dist = float(np.linalg.norm(state - center))
        scale = max(float(np.linalg.norm(self.obs_high - self.obs_low)), 1e-6)
        reward = self.cfg.distance_reward * (self.prev_dist - dist) / scale
        self.prev_dist = dist
        return reward

    def _wrap_periodic_dims(self, state):
        wrap = np.asarray(getattr(self.model, "wrap", np.zeros(self.model.n, dtype=bool)), dtype=bool)
        if not np.any(wrap):
            return state
        wrapped = state.copy()
        lengths = self.obs_high - self.obs_low
        periodic_idx = np.where(wrap)[0]
        for idx in periodic_idx:
            length = lengths[idx]
            if length <= 0:
                continue
            wrapped[idx] = ((wrapped[idx] - self.obs_low[idx]) % length) + self.obs_low[idx]
        return wrapped

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        testing = bool(options.get("testing", False)) if options else False

        if testing:
            cell = np.asarray(self.state_to_cell(self.model.x0), dtype=np.float32)
            cell_lb = self.obs_low + cell * self.bin_widths
            cell_ub = cell_lb + self.bin_widths

            # Inflate the initial-state cell by a per-dim (lo, hi) number of cells (same format
            # as inflation_rate), so the active-set rollouts fan across a neighborhood of x0
            # rather than a single trajectory. Defaults to zero -> exactly the single-cell reset.
            reset_infl = getattr(self.model, "reset_inflation", None)
            if reset_infl is None:
                reset_infl = [(0, 0)] * self.model.n
            lo = np.array([l for l, h in reset_infl], dtype=np.float32)
            hi = np.array([h for l, h in reset_infl], dtype=np.float32)

            # Add a small epsilon, to make sure we appropriately cover the initial state cell
            eps = 0.1 * self.bin_widths
            low = np.clip(cell_lb + lo * self.bin_widths - eps, self.obs_low, self.obs_high)
            high = np.clip(cell_ub + hi * self.bin_widths + eps, self.obs_low, self.obs_high)
            state = self.np_random.uniform(low, high).astype(np.float32)
            while self._in_boxes(state, self.critical, inflate=0):
                state = self.np_random.uniform(low, high).astype(np.float32)
        else:
            state = self.np_random.uniform(self.obs_low, self.obs_high).astype(np.float32)
            while self._in_boxes(state, self.critical, inflate=0):
                state = self.np_random.uniform(self.obs_low, self.obs_high).astype(np.float32)

        self.state = state
        self.steps = 0

        center = self._goal_center()
        if center is not None:
            self.prev_dist = float(np.linalg.norm(self.state - center))
        else:
            self.prev_dist = 0.0

        return self.state.copy(), {}

    def step(self, action, noise_factor=2):
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)

        noise = noise_factor * np.asarray(self.model.noise.sample(), dtype=np.float32)
        next_state = np.asarray(self.model.step(self.state, action, noise), dtype=np.float32)
        # next_state = self._wrap_periodic_dims(next_state)

        self.state = next_state
        self.steps += 1

        in_goal = self._in_boxes(self.state, self.goal)
        in_critical = self._in_boxes(self.state, self.critical, inflate=1)
        out_of_bounds = bool(np.any(self.state < self.obs_low) or np.any(self.state > self.obs_high))

        if in_goal:
            reward = self.cfg.goal_reward
        elif in_critical:
            reward = self.cfg.unsafe_penalty
        elif out_of_bounds:
            reward = self.cfg.out_of_bounds_penalty
        else:
            reward = self._distance_reward(self.state) + self.cfg.per_step_reward

        cell = self.state_to_cell(self.state)
        flat_idx = np.ravel_multi_index(cell, self.model.partition['number_per_dim'])
        if flat_idx in self.previous_cells:
            reward -= self.cfg.revisit_penalty

        terminated = in_goal or in_critical or out_of_bounds
        truncated = self.steps >= self.cfg.max_steps

        info = {
            "in_goal": in_goal,
            "in_critical": in_critical,
            "out_of_bounds": out_of_bounds,
            "cell": cell,
        }
        return self.state.copy(), float(reward), terminated, truncated, info

def evaluate_policy(model, norm_env, base_model, cfg, episodes, dims, args, discrete_actions=None, seed=0):
    norm_env.training = False
    norm_env.norm_reward = False

    t = time()

    eval_env = BenchmarkRLEnv(base_model, cfg)
    reached_goal = 0
    visited_cells = set()
    trajectories = []

    if discrete_actions is not None:
        discrete_actions = np.asarray(discrete_actions, dtype=np.float32)

    for ep_idx in range(episodes):
        obs, _ = eval_env.reset(seed=seed if ep_idx == 0 else None, options={"testing": True})
        visited_cells.add(eval_env.state_to_cell(obs))

        trace = [obs.copy()]
        for _ in range(cfg.max_steps):
            # Quantize observation to the center of its partition cell.
            if discrete_actions is not None:
                cell = eval_env.state_to_cell(obs)
                obs_q = (eval_env.obs_low + (np.asarray(cell, dtype=np.float32) + 0.5) * eval_env.bin_widths)
                obs_q = np.clip(obs_q, eval_env.obs_low, eval_env.obs_high)

                # print(f"Original obs: {obs}, quantized obs: {obs_q}, cell: {cell}")
            else:
                obs_q = obs

            norm_obs = norm_env.normalize_obs(np.expand_dims(obs_q, axis=0))[0]
            action, _ = model.predict(norm_obs, deterministic=True)

            # Quantize the continuous action to the nearest discrete action.
            if discrete_actions is not None:
                dists = np.linalg.norm(discrete_actions - action, axis=1)
                action = discrete_actions[np.argmin(dists)]

                # print(f"Original action: {action}, quantized action: {action}")

            obs, _, terminated, truncated, info = eval_env.step(action, noise_factor=1)

            visited_cells.add(info["cell"])
            trace.append(obs.copy())

            if terminated or truncated:
                if terminated and info["in_goal"]:
                    reached_goal += 1
                break

        trajectories.append(np.asarray(trace))

    if len(dims) != 2:
        raise ValueError("This runner currently supports plotting exactly 2 dimensions.")

    logger.info(f"- Evaluation rollouts completed in {time() - t:.2f} seconds.")
    t = time()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    legend_handles = []

    def add_boxes_collection(ax, boxes, dims, color, alpha, label):
        if boxes is None or boxes.size == 0:
            return None
        rects = [
            mpatches.Rectangle(
                (box[0, dims[0]], box[0, dims[1]]),
                box[1, dims[0]] - box[0, dims[0]],
                box[1, dims[1]] - box[0, dims[1]],
            )
            for box in boxes
        ]
        collection = PatchCollection(
            rects, facecolor=color, edgecolor="none", alpha=alpha, rasterized=True
        )
        ax.add_collection(collection)
        return mpatches.Patch(color=color, alpha=alpha, label=label)

    if eval_env.critical.size > 0:
        h = add_boxes_collection(ax, eval_env.critical, dims, "red", 0.15, "Critical")
        if h:
            legend_handles.append(h)

    if eval_env.goal.size > 0:
        h = add_boxes_collection(ax, eval_env.goal, dims, "green", 0.25, "Goal")
        if h:
            legend_handles.append(h)

    if hasattr(eval_env.model, "charging_station") and eval_env.model.charging_station.size > 0:
        h = add_boxes_collection(ax, eval_env.model.charging_station, dims, "blue", 0.25, "Charging station")
        if h:
            legend_handles.append(h)

    # Only plot max 100 trajectories
    i_max = 100
    if len(trajectories) > 0:
        selected_traces = [trace[:, dims] for trace in trajectories[:i_max]]
        nan_pad = np.full((1, 2), np.nan)
        stacked_traces = []
        for trace in selected_traces:
            stacked_traces.append(trace)
            stacked_traces.append(nan_pad)
        combined = np.vstack(stacked_traces)
        ax.plot(
            combined[:, 0],
            combined[:, 1],
            linewidth=1.0,
            alpha=0.9,
            color="black",
            marker=".",
            markersize=3.0,
            markeredgecolor="red",
            markerfacecolor="red",
        )

    ax.set_xlim(eval_env.obs_low[dims[0]], eval_env.obs_high[dims[0]])
    ax.set_ylim(eval_env.obs_low[dims[1]], eval_env.obs_high[dims[1]])
    ax.set_xlabel(base_model.state_variables[dims[0]])
    ax.set_ylabel(base_model.state_variables[dims[1]])
    ax.set_title(f"PPO trajectories ({base_model.__class__.__name__})")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="best")
    plt.tight_layout()
    output_dir = Path(getattr(args, 'output_dir', 'output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'rl_trajectories.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'rl_trajectories.png', format='png', bbox_inches='tight')
    plt.close(fig)

    logger.info(f"- Rollouts plotted completed in {time() - t:.2f} seconds.")

    total_cells = int(np.prod(base_model.partition['number_per_dim']))
    return reached_goal, visited_cells, total_cells

def _build_vec_env(base_model, cfg, n_envs, use_subproc, previous_cells, seed=0):
    env_kwargs = {"model": base_model, "cfg": cfg, "previous_cells": previous_cells}
    vec_env_cls = SubprocVecEnv if use_subproc else DummyVecEnv
    vec_env = make_vec_env(BenchmarkRLEnv, n_envs=n_envs, env_kwargs=env_kwargs, vec_env_cls=vec_env_cls, seed=seed)
    return VecNormalize(vec_env, norm_obs=True, norm_reward=True)

def find_policy_actions_batch(obs_batch, ppo, vec_env, discrete_actions, num):
    """Return the `num` nearest discrete actions for each observation in obs_batch.

    obs_batch : (N, obs_dim) float32
    Returns   : (N, num, action_dim) float32
    """
    norm_obs = vec_env.normalize_obs(obs_batch)                          # (N, obs_dim)
    actions, _ = ppo.predict(norm_obs, deterministic=True)               # (N, action_dim)
    if num >= discrete_actions.shape[0]:
        top_k_idx = np.tile(np.arange(discrete_actions.shape[0]), (obs_batch.shape[0], 1))
    else:
        diff = actions[:, None, :] - discrete_actions[None, :, :]
        sq_dists = np.sum(diff * diff, axis=2)                            # (N, N_discrete)
        top_k_idx = np.argpartition(sq_dists, num - 1, axis=1)[:, :num]     # (N, num)
    return discrete_actions[top_k_idx], discrete_actions[top_k_idx[:, 0]]

def _inflation_offsets(inflation_rate):
    '''
    Every integer offset inside the inflation box, shape (num_offsets, dim). The box is identical for
    every cell, so it is built once and broadcast against all visited cells at once.
    '''
    axes = [np.arange(int(lo), int(hi) + 1) for lo, hi in inflation_rate]
    grids = np.meshgrid(*axes, indexing='ij')
    return np.stack([grid.ravel() for grid in grids], axis=1)


@jax.jit
def _neighbors_in_bounds(cells, offsets, number_per_dim):
    '''
    Mask of the neighbors of each cell that lie inside the partition, shape (num_cells, num_offsets).
    '''
    def in_bounds_of(cell):
        neighbors = cell + offsets
        return jnp.all((neighbors >= 0) & (neighbors < number_per_dim), axis=1)

    return jax.vmap(in_bounds_of)(cells)


def _inflate_cells(visited_cells, inflation_rate, number_per_dim):
    '''
    Inflate every visited cell into its neighborhood and return the unique in-bounds cells, shape
    (num_states, dim), ordered by linear id.
    '''
    dim = len(number_per_dim)
    cells = np.asarray(list(visited_cells), dtype=np.int64).reshape(-1, dim)
    offsets = _inflation_offsets(inflation_rate)
    strides = _compute_linear_strides(number_per_dim)

    # Cells are deduplicated through their linear id. That id indexes the full nominal grid (~6e11
    # cells for CartPole) however few cells are active, so it needs int64 and stays in numpy: JAX
    # truncates to int32 unless x64 is on. The id is linear in the coordinates, so
    # id(cell + offset) == id(cell) + id(offset), and a batch of ids is one outer sum.
    base_ids = cells @ strides
    offset_ids = offsets @ strides

    # The coordinates themselves always fit int32, so the bounds check can run on device.
    offsets_jax = jnp.asarray(offsets, dtype=jnp.int32)
    number_per_dim_jax = jnp.asarray(number_per_dim, dtype=jnp.int32)
    batch = max(1, INFLATE_IDS_PER_BATCH // len(offsets))

    # Deduplicate per batch, not just at the end: the inflation boxes of nearby cells overlap heavily,
    # which keeps the accumulated ids far below num_cells * num_offsets.
    batches = []
    for start in tqdm(range(0, len(cells), batch), desc='Inflate visited cells'):
        in_bounds = _neighbors_in_bounds(
            jnp.asarray(cells[start:start + batch], dtype=jnp.int32), offsets_jax, number_per_dim_jax)
        ids = base_ids[start:start + batch, None] + offset_ids[None, :]
        batches.append(np.unique(np.where(np.asarray(in_bounds), ids, -1)))

    ids = np.unique(np.concatenate(batches)) if batches else np.zeros(0, dtype=np.int64)
    ids = ids[ids >= 0]

    # Undo the linearization: strides are row-major, so entry d is (id // stride[d]) % number[d].
    return ((ids[:, None] // strides) % number_per_dim).astype(int)


def _build_prefix_sum(active_mask, number_per_dim, wrap):
    """Build a D-dimensional prefix sum (summed area table) from a flat boolean mask.

    For wrapping dimensions, the grid is tiled (doubled) along that axis so that
    any contiguous sub-range of length <= num_d can be queried without splitting.

    Returns (prefix_flat, prefix_strides) where:
    - prefix_flat: 1-D int32 array, the flattened padded prefix sum
    - prefix_strides: (D,) int64 array, row-major strides for flat indexing
    """
    nd = np.asarray(number_per_dim, dtype=np.int64)
    D = len(nd)

    grid = active_mask.reshape(tuple(nd)).astype(np.int32)

    for d in range(D):
        if wrap[d]:
            grid = np.concatenate([grid, grid], axis=d)

    grid_shape = grid.shape
    padded_shape = tuple(s + 1 for s in grid_shape)
    P = np.zeros(padded_shape, dtype=np.int32)
    slices = tuple(slice(1, None) for _ in range(D))
    P[slices] = grid

    for d in range(D):
        np.cumsum(P, axis=d, out=P)

    padded_strides = np.ones(D, dtype=np.int64)
    for d in range(D - 2, -1, -1):
        padded_strides[d] = padded_strides[d + 1] * padded_shape[d + 1]

    return P.ravel(), padded_strides


def _box_count_prefix_sum(prefix_flat, prefix_strides, raw_lbs, raw_ubs,
                           number_per_dim, wrap):
    """Count active cells in axis-aligned boxes via inclusion-exclusion on a prefix sum.

    For D dimensions, each box query requires only 2^D lookups (e.g. 64 for D=6)
    instead of enumerating potentially hundreds or thousands of cells.

    Parameters
    ----------
    prefix_flat : 1-D int32 array
        Flattened padded prefix sum from _build_prefix_sum.
    prefix_strides : (D,) int64 array
        Row-major strides for flat indexing into the padded prefix sum.
    raw_lbs, raw_ubs : (..., D) int arrays
        Lower and upper bounds of the boxes (may be outside the grid).
    number_per_dim : (D,) int-like array
        Grid dimensions.
    wrap : (D,) bool array
        Which dimensions wrap periodically.

    Returns
    -------
    counts : (...) int64 array
        Number of active cells in each box.
    """
    D = len(number_per_dim)
    nd = np.asarray(number_per_dim, dtype=np.int64)
    batch_shape = raw_lbs.shape[:-1]

    query_lbs = np.empty_like(raw_lbs, dtype=np.int64)
    query_ubs = np.empty_like(raw_ubs, dtype=np.int64)
    empty = np.zeros(batch_shape, dtype=bool)

    for d in range(D):
        if wrap[d]:
            raw_span = raw_ubs[..., d] - raw_lbs[..., d] + 1
            span = np.clip(raw_span, 0, nd[d])
            query_lbs[..., d] = raw_lbs[..., d] % nd[d]
            query_ubs[..., d] = query_lbs[..., d] + span - 1
            empty |= (span <= 0)
        else:
            empty |= (raw_ubs[..., d] < 0) | (raw_lbs[..., d] >= nd[d])
            query_lbs[..., d] = np.clip(raw_lbs[..., d], 0, nd[d] - 1)
            query_ubs[..., d] = np.clip(raw_ubs[..., d], 0, nd[d] - 1)

    # Precompute per-dimension contributions for lb and ub+1 corners
    lb_contrib = query_lbs * prefix_strides
    ub_contrib = (query_ubs + 1) * prefix_strides

    # 2^D inclusion-exclusion corners
    corners = np.array(list(itertools.product([False, True], repeat=D)))
    signs = np.where(np.sum(corners, axis=1) % 2 == 0, 1, -1).astype(np.int64)

    counts = np.zeros(batch_shape, dtype=np.int64)
    for i in range(len(corners)):
        contrib = np.where(corners[i], lb_contrib, ub_contrib)
        flat_idx = np.sum(contrib, axis=-1)
        counts += signs[i] * prefix_flat[flat_idx].astype(np.int64)

    counts[empty] = 0
    return counts


def _smart_inflate_cells(
    visited,
    model,
    val_env,
    ppo,
    vec_env,
    discrete_actions,
    args,
    number_per_dim,
    noise_support_ratio=0.0,
):
    dim = len(number_per_dim)
    visited_arr = (
        np.asarray(list(visited), dtype=np.int64).reshape(-1, dim)
        if len(visited) > 0
        else np.zeros((0, dim), dtype=np.int64)
    )

    strides = _compute_linear_strides(number_per_dim)
    total_grid_size = int(np.prod(number_per_dim))

    active_states_mask = np.zeros(total_grid_size, dtype=bool)
    if len(visited_arr) > 0:
        visited_flats = np.dot(visited_arr, strides)
        active_states_mask[visited_flats] = True

    wrap = np.asarray(getattr(model, "wrap", np.zeros(model.n, dtype=bool)), dtype=bool)

    @partial(jax.jit, static_argnums=(0,))
    def _compute_batch_frs_bounds(step_set_fn, s_mins, s_maxs, actions_batch):
        def _per_state(s_min, s_max, actions):
            return jax.vmap(lambda u: step_set_fn(s_min, s_max, u, u))(actions)
        return jax.vmap(_per_state)(s_mins, s_maxs, actions_batch)

    def _extract_frs_cells_vectorized(raw_idx_lbs, raw_idx_ubs):
        N, K, D = raw_idx_lbs.shape
        spans = np.maximum(1, raw_idx_ubs - raw_idx_lbs + 1)
        max_spans = np.max(spans, axis=(0, 1))

        offset_axes = [np.arange(s, dtype=np.int64) for s in max_spans]
        if D == 1:
            offsets = offset_axes[0][:, None]
        else:
            grids = np.meshgrid(*offset_axes, indexing='ij')
            offsets = np.stack([g.ravel() for g in grids], axis=-1)

        M = offsets.shape[0]
        flat_cells = np.zeros((N, K, M), dtype=np.int64)
        valid_mask = np.ones((N, K, M), dtype=bool)

        for d in range(D):
            coord_d = raw_idx_lbs[:, :, d, None] + offsets[:, d][None, None, :]
            ub_d = raw_idx_ubs[:, :, d, None]
            num_d = number_per_dim[d]
            stride_d = strides[d]

            if wrap[d]:
                valid_mask &= (coord_d <= ub_d)
                flat_cells += (coord_d % num_d) * stride_d
            else:
                valid_mask &= (coord_d <= ub_d) & (coord_d >= 0) & (coord_d < num_d)
                flat_cells += np.clip(coord_d, 0, num_d - 1) * stride_d

        return flat_cells, valid_mask

    if len(visited_arr) > 0:
        obs_batch_init = np.asarray(
            val_env.obs_low + (visited_arr.astype(np.float32) + 0.5) * val_env.bin_widths,
            dtype=np.float32,
        )
        _, init_rl_actions = find_policy_actions_batch(obs_batch_init, ppo, vec_env, discrete_actions, num=1)
    else:
        init_rl_actions = np.zeros((0, discrete_actions.shape[1]), dtype=np.float32)

    noise_support = 0.0
    if hasattr(model, 'noise') and isinstance(model.noise, dict) and 'support_radius' in model.noise:
        noise_support = model.noise['support_radius'] * noise_support_ratio

    CHUNK_SIZE = 16384

    phase1_added_count = 0
    pbar_p1 = tqdm(total=len(visited_arr), desc="Phase 1: Batched FRS expansion")

    visited_arr_all = np.asarray(visited_arr, dtype=np.float32)
    init_rl_actions_all_batch = init_rl_actions[:, None, :]  # shape (M, 1, action_dim)
    p1_new_queue_flats = []

    for ch_start in range(0, len(visited_arr), CHUNK_SIZE):
        ch_end = min(ch_start + CHUNK_SIZE, len(visited_arr))
        visited_chunk = visited_arr_all[ch_start:ch_end]
        init_actions_chunk = init_rl_actions_all_batch[ch_start:ch_end]

        s_mins_p1 = val_env.obs_low + visited_chunk * val_env.bin_widths
        s_maxs_p1 = s_mins_p1 + val_env.bin_widths

        frs_mins_p1, frs_maxs_p1 = _compute_batch_frs_bounds(
            model.step_set, s_mins_p1, s_maxs_p1, jnp.asarray(init_actions_chunk, dtype=jnp.float32)
        )
        frs_mins_p1_noise = np.asarray(frs_mins_p1[:, 0, :], dtype=np.float32) - noise_support
        frs_maxs_p1_noise = np.asarray(frs_maxs_p1[:, 0, :], dtype=np.float32) + noise_support

        raw_idx_lbs_p1 = np.floor((frs_mins_p1_noise - val_env.obs_low) / val_env.bin_widths).astype(int)[:, None, :]
        raw_idx_ubs_p1 = np.floor((frs_maxs_p1_noise - val_env.obs_low) / val_env.bin_widths).astype(int)[:, None, :]

        flat_cells_p1, valid_mask_p1 = _extract_frs_cells_vectorized(raw_idx_lbs_p1, raw_idx_ubs_p1)

        flats_p1 = flat_cells_p1[:, 0, :]
        valid_p1 = valid_mask_p1[:, 0, :]

        is_new_p1 = valid_p1 & (~active_states_mask[flats_p1])
        new_flats_p1 = flats_p1[is_new_p1]

        if len(new_flats_p1) > 0:
            unique_flats = np.unique(new_flats_p1)
            active_states_mask[unique_flats] = True
            p1_new_queue_flats.append(unique_flats)
            phase1_added_count += len(unique_flats)

        pbar_p1.update(ch_end - ch_start)

    pbar_p1.close()

    if len(p1_new_queue_flats) > 0:
        queue_flats = np.unique(np.concatenate(p1_new_queue_flats))
    else:
        queue_flats = np.array([], dtype=np.int64)

    logger.info(f"- Phase 1 complete: Added {phase1_added_count} states. Queue size: {len(queue_flats)}.")

    # Build initial prefix sum for Phase 2 box-counting.
    # Uses a summed area table (N-D prefix sum) to count active cells in any
    # axis-aligned box in O(2^D) lookups instead of enumerating all cells.
    prefix_flat, prefix_strides = _build_prefix_sum(active_states_mask, number_per_dim, wrap)

    phase2_added_count = 0
    p2_iter = 0
    while len(queue_flats) > 0:
        p2_iter += 1
        N_q = len(queue_flats)
        pbar_p2 = tqdm(total=N_q, desc=f"Phase 2 (iter {p2_iter}): Queue FRS expansion", unit="state")

        # Vectorized inverse lookup: flat integer indices to 6D grid coordinates
        queue_coords = np.stack(np.unravel_index(queue_flats, number_per_dim), axis=-1)

        obs_batch_queue = np.asarray(
            val_env.obs_low + (queue_coords.astype(np.float32) + 0.5) * val_env.bin_widths,
            dtype=np.float32,
        )
        queue_actions_all, _ = find_policy_actions_batch(
            obs_batch_queue, ppo, vec_env, discrete_actions, num=args.RL_actions_per_state
        )

        p2_iter_new_flats = []
        for ch_start in range(0, N_q, CHUNK_SIZE):
            ch_end = min(ch_start + CHUNK_SIZE, N_q)
            queue_chunk = queue_coords[ch_start:ch_end]
            actions_chunk = queue_actions_all[ch_start:ch_end]

            s_mins_q = val_env.obs_low + queue_chunk.astype(np.float32) * val_env.bin_widths
            s_maxs_q = s_mins_q + val_env.bin_widths
            actions_batch_jnp = jnp.asarray(actions_chunk, dtype=jnp.float32)

            frs_mins_q, frs_maxs_q = _compute_batch_frs_bounds(
                model.step_set, s_mins_q, s_maxs_q, actions_batch_jnp
            )
            frs_mins_q_noise = np.asarray(frs_mins_q, dtype=np.float32) - noise_support
            frs_maxs_q_noise = np.asarray(frs_maxs_q, dtype=np.float32) + noise_support

            raw_idx_lbs_q = np.floor((frs_mins_q_noise - val_env.obs_low) / val_env.bin_widths).astype(int)
            raw_idx_ubs_q = np.floor((frs_maxs_q_noise - val_env.obs_low) / val_env.bin_widths).astype(int)

            # --- COUNT PHASE: O(2^D) prefix-sum box queries instead of cell enumeration ---
            active_counts = _box_count_prefix_sum(
                prefix_flat, prefix_strides, raw_idx_lbs_q, raw_idx_ubs_q,
                number_per_dim, wrap
            )

            best_act_indices = np.argmax(active_counts, axis=1)

            # --- EXPAND PHASE: Only enumerate cells for the winning action (1 not K) ---
            N_chunk = ch_end - ch_start
            row_idx = np.arange(N_chunk)
            win_lbs = raw_idx_lbs_q[row_idx, best_act_indices][:, None, :]
            win_ubs = raw_idx_ubs_q[row_idx, best_act_indices][:, None, :]

            flat_cells_q, valid_mask_q = _extract_frs_cells_vectorized(win_lbs, win_ubs)
            win_flats = flat_cells_q[:, 0, :]
            win_valid = valid_mask_q[:, 0, :]

            is_new_cell = win_valid & (~active_states_mask[win_flats])
            chunk_new_flats = win_flats[is_new_cell]

            if len(chunk_new_flats) > 0:
                p2_iter_new_flats.append(chunk_new_flats)

            pbar_p2.update(N_chunk)

        if len(p2_iter_new_flats) > 0:
            combined_new = np.concatenate(p2_iter_new_flats)
            unique_new = np.unique(combined_new)
            active_states_mask[unique_new] = True
            queue_flats = unique_new
            added_this_iter = len(unique_new)
            phase2_added_count += added_this_iter
            # Rebuild prefix sum for next BFS iteration
            prefix_flat, prefix_strides = _build_prefix_sum(
                active_states_mask, number_per_dim, wrap
            )
        else:
            queue_flats = np.array([], dtype=np.int64)
            added_this_iter = 0

        pbar_p2.set_postfix({
            "added_iter": added_this_iter,
            "total_active": int(np.sum(active_states_mask))
        })
        pbar_p2.close()
    logger.info(f"- Phase 2 complete: Added {phase2_added_count} states. Total active states: {int(np.sum(active_states_mask))}.")

    all_active_flats = np.where(active_states_mask)[0]
    active_states = np.stack(np.unravel_index(all_active_flats, number_per_dim), axis=-1).astype(int)
    return active_states

def find_active(model, args, previous_cells):
    cfg = RLConfig(
        max_steps=args.max_steps,
        goal_reward=args.goal_reward,
        unsafe_penalty=args.unsafe_penalty,
        out_of_bounds_penalty=args.out_of_bounds_penalty,
        revisit_penalty=args.revisit_penalty,
        distance_reward=args.distance_reward,
        per_step_reward=args.per_step_reward,
    )

    vec_env = _build_vec_env(
        base_model=model,
        cfg=cfg,
        n_envs=args.n_envs,
        use_subproc=args.subproc,
        previous_cells=previous_cells,
        seed=args.seed,
    )

    val_env = BenchmarkRLEnv(model, cfg)

    pi_arch = args.pi_arch if args.pi_arch is not None else model.pi_arch
    vf_arch = args.vf_arch if args.vf_arch is not None else model.vf_arch
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=dict(pi=pi_arch, vf=vf_arch))

    ppo = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        batch_size=args.rl_batch_size,
        n_steps=args.n_steps,
        seed=args.seed,
    )

    ppo.learn(total_timesteps=args.total_timesteps, progress_bar=True)

    discrete_actions_per_dim = [
        np.linspace(model.uMin[i], model.uMax[i], num=model.num_actions[i])
        for i in range(len(model.num_actions))
    ]
    discrete_actions = np.array(list(itertools.product(*discrete_actions_per_dim)), dtype=np.float32)

    goal_reached, newly_visited, total_cells = evaluate_policy(
        model=ppo,
        norm_env=vec_env,
        base_model=model,
        cfg=cfg,
        episodes=args.eval_episodes,
        dims=list(model.plot_dimensions),
        args=args,
        discrete_actions=discrete_actions,
        seed=args.seed,
    )

    logger.info(f"Goal reached in {goal_reached}/{args.eval_episodes} episodes.")
    t = time()

    # Inflate visited cells to include neighbors within a certain radius to account for discretization
    # errors and encourage exploration of nearby states.
    if args.tube_method == "inflation":
        number_per_dim = np.asarray(model.partition['number_per_dim'], dtype=np.int64)
        active_states = _inflate_cells(newly_visited, model.inflation_rate, number_per_dim)
    elif args.tube_method == "smart":
        number_per_dim = np.asarray(model.partition['number_per_dim'], dtype=np.int64)
        active_states = _smart_inflate_cells(
            visited=newly_visited,
            model=model,
            val_env=val_env,
            ppo=ppo,
            vec_env=vec_env,
            discrete_actions=discrete_actions,
            args=args,
            number_per_dim=number_per_dim,
        )
    else:
        raise ValueError(f"Unknown tube_method: {args.tube_method}")

    logger.info(f"- Active states extraction completed in {time() - t:.2f} seconds.")
    t = time()

    # One batched forward pass for all active states instead of one call per state.
    obs_batch = np.asarray(
        val_env.obs_low + (active_states.astype(np.float32) + 0.5) * val_env.bin_widths,
        dtype=np.float32,
    )
    top_k, rl_policy = find_policy_actions_batch(obs_batch, ppo, vec_env, discrete_actions, num=args.RL_actions_per_state)
    active_actions = {tuple(cell): top_k[i] for i, cell in enumerate(active_states.tolist())}

    logger.info(f"- Active state/action extraction completed in {time() - t:.2f} seconds.")

    return active_states, active_actions, rl_policy
