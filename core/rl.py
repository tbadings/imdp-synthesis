import itertools
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import time
import gymnasium as gym
from gymnasium import spaces
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from tqdm import tqdm
from core.abstraction.partition import _compute_linear_strides

logger = logging.getLogger(__name__)

# Max neighbor indices processed per chunk to bound peak memory during grid inflation
INFLATE_IDS_PER_BATCH = 1 << 23
# Batch chunk size for forward reachable set expansions
CHUNK_SIZE = 16384


# Configuration dataclass holding reward weights and horizon limits for RL training
@dataclass
class RLConfig:
    max_steps: int
    goal_reward: float
    unsafe_penalty: float
    out_of_bounds_penalty: float
    revisit_penalty: float
    distance_reward: float = 0.0
    per_step_reward: float = 0.0


# Gymnasium environment wrapping continuous benchmark system models for RL exploration
class BenchmarkRLEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, model, cfg: RLConfig, previous_cells=None):
        super().__init__()
        self.model = model
        self.cfg = cfg

        # Extract partition boundary coordinates from model
        boundary = np.asarray(self.model.partition["boundary"], dtype=np.float32)
        self.obs_low = boundary[0]
        self.obs_high = boundary[1]

        # Define continuous observation and action box spaces
        self.observation_space = spaces.Box(self.obs_low, self.obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.asarray(self.model.uMin, dtype=np.float32),
            high=np.asarray(self.model.uMax, dtype=np.float32),
            dtype=np.float32,
        )

        # Compute grid cell widths along each dimension
        self.number_per_dim = np.asarray(self.model.partition["number_per_dim"], dtype=np.int64)
        self.bin_widths = (self.obs_high - self.obs_low) / self.number_per_dim

        # Extract goal and critical region bounding boxes
        self.goal = np.asarray(getattr(self.model, "goal", np.empty((0, 2, self.model.n))), dtype=np.float32)
        self.critical = np.asarray(getattr(self.model, "critical", np.empty((0, 2, self.model.n))), dtype=np.float32)

        # Track previous cells, step counts, and distance baseline
        self.previous_cells = set() if previous_cells is None else set(previous_cells)
        self.state = None
        self.steps = 0
        self.prev_dist = 0.0

        # Precompute static goal center and domain scale for reward normalization
        self._goal_center = 0.5 * (self.goal[0, 0] + self.goal[0, 1]) if self.goal.size > 0 else None
        self._domain_scale = max(float(np.linalg.norm(self.obs_high - self.obs_low)), 1e-6)

    # Update flat cell indices previously visited to penalize revisitation
    def set_previous_cells(self, previous_cells):
        self.previous_cells = set(previous_cells)

    # Map continuous state observation to discrete grid cell multi-index
    def state_to_cell(self, obs):
        indices = np.floor((np.asarray(obs, dtype=np.float64) - self.obs_low) / self.bin_widths).astype(int)
        return tuple(np.clip(indices, 0, self.number_per_dim - 1).tolist())

    # Convert discrete grid cell index to its continuous center point
    def cell_to_center(self, cell):
        return self.obs_low + (np.asarray(cell, dtype=np.float32) + 0.5) * self.bin_widths

    # Check if a state is contained within any given bounding box
    @staticmethod
    def _in_boxes(state, boxes, inflate=0.0):
        if boxes.size == 0:
            return False
        return bool(np.any(np.all((state >= boxes[:, 0, :] - inflate) & (state <= boxes[:, 1, :] + inflate), axis=1)))

    # Compute potential-based distance reward towards goal center
    def _distance_reward(self, state):
        if self._goal_center is None:
            return 0.0
        dist = float(np.linalg.norm(state - self._goal_center))
        reward = self.cfg.distance_reward * (self.prev_dist - dist) / self._domain_scale
        self.prev_dist = dist
        return reward

    # Reset environment state: local neighborhood of x0 for testing, global bounds for training
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        testing = bool(options.get("testing", False)) if options else False

        if testing:
            # Expand neighborhood around initial state x0 by reset_inflation offsets
            cell = np.asarray(self.state_to_cell(self.model.x0), dtype=np.float32)
            cell_lb = self.obs_low + cell * self.bin_widths
            cell_ub = cell_lb + self.bin_widths
            reset_infl = getattr(self.model, "reset_inflation", None) or ([(0, 0)] * self.model.n)
            lo = np.array([l for l, h in reset_infl], dtype=np.float32)
            hi = np.array([h for l, h in reset_infl], dtype=np.float32)
            eps = 0.1 * self.bin_widths
            low = np.clip(cell_lb + lo * self.bin_widths - eps, self.obs_low, self.obs_high)
            high = np.clip(cell_ub + hi * self.bin_widths + eps, self.obs_low, self.obs_high)
        else:
            low, high = self.obs_low, self.obs_high

        # Sample uniform state within bounds rejecting critical collision states
        state = self.np_random.uniform(low, high).astype(np.float32)
        while self._in_boxes(state, self.critical):
            state = self.np_random.uniform(low, high).astype(np.float32)

        self.state = state
        self.steps = 0
        self.prev_dist = float(np.linalg.norm(self.state - self._goal_center)) if self._goal_center is not None else 0.0
        return self.state.copy(), {}

    # Step continuous dynamics with additive disturbance noise and compute step reward
    def step(self, action, noise_factor=2.0):
        # Clip action and sample disturbance noise
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        noise = noise_factor * np.asarray(self.model.noise.sample(), dtype=np.float32)
        self.state = np.asarray(self.model.step(self.state, action, noise), dtype=np.float32)
        self.steps += 1

        # Check reach, safety, and boundary containment
        in_goal = self._in_boxes(self.state, self.goal)
        in_critical = self._in_boxes(self.state, self.critical, inflate=1.0)
        out_of_bounds = bool(np.any(self.state < self.obs_low) or np.any(self.state > self.obs_high))

        # Assign reward based on terminal condition or continuous potential progress
        if in_goal:
            reward = self.cfg.goal_reward
        elif in_critical:
            reward = self.cfg.unsafe_penalty
        elif out_of_bounds:
            reward = self.cfg.out_of_bounds_penalty
        else:
            reward = self._distance_reward(self.state) + self.cfg.per_step_reward

        # Apply revisit penalty if current state was previously explored
        cell = self.state_to_cell(self.state)
        if np.ravel_multi_index(cell, self.number_per_dim) in self.previous_cells:
            reward -= self.cfg.revisit_penalty

        terminated = in_goal or in_critical or out_of_bounds
        truncated = self.steps >= self.cfg.max_steps
        info = {"in_goal": in_goal, "in_critical": in_critical, "out_of_bounds": out_of_bounds, "cell": cell}
        return self.state.copy(), float(reward), terminated, truncated, info


# Create vectorized environment with observation and reward normalization
def _build_vec_env(base_model, cfg, n_envs, use_subproc, previous_cells, seed=0):
    vec_env_cls = SubprocVecEnv if use_subproc else DummyVecEnv
    vec_env = make_vec_env(
        BenchmarkRLEnv,
        n_envs=n_envs,
        env_kwargs={"model": base_model, "cfg": cfg, "previous_cells": previous_cells},
        vec_env_cls=vec_env_cls,
        seed=seed,
    )
    return VecNormalize(vec_env, norm_obs=True, norm_reward=True)


# Query policy for continuous actions and match to top-k nearest discrete grid actions
def find_policy_actions_batch(obs_batch, ppo, vec_env, discrete_actions, num):
    norm_obs = vec_env.normalize_obs(obs_batch)
    actions, _ = ppo.predict(norm_obs, deterministic=True)
    if num >= discrete_actions.shape[0]:
        top_k_idx = np.tile(np.arange(discrete_actions.shape[0]), (obs_batch.shape[0], 1))
    else:
        diff = actions[:, None, :] - discrete_actions[None, :, :]
        top_k_idx = np.argpartition(np.sum(diff * diff, axis=2), num - 1, axis=1)[:, :num]
    return discrete_actions[top_k_idx], discrete_actions[top_k_idx[:, 0]]


# Render 2D projections of trajectories, obstacle regions, and goal regions
def plot_rl_trajectories(base_model, eval_env, trajectories, dims, output_dir, max_trajectories=100):
    if len(dims) != 2:
        raise ValueError("This runner currently supports plotting exactly 2 dimensions.")

    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(8, 8))
    legend_handles = []

    # Helper to convert bounding boxes to a Matplotlib PatchCollection
    def _add_boxes(boxes, color, alpha, label):
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
        ax.add_collection(PatchCollection(rects, facecolor=color, edgecolor="none", alpha=alpha, rasterized=True))
        return mpatches.Patch(color=color, alpha=alpha, label=label)

    # Add critical, goal, and charging station region patches
    if eval_env.critical.size > 0:
        h = _add_boxes(eval_env.critical, "red", 0.15, "Critical")
        if h:
            legend_handles.append(h)
    if eval_env.goal.size > 0:
        h = _add_boxes(eval_env.goal, "green", 0.25, "Goal")
        if h:
            legend_handles.append(h)
    if hasattr(eval_env.model, "charging_station") and eval_env.model.charging_station.size > 0:
        h = _add_boxes(eval_env.model.charging_station, "blue", 0.25, "Charging station")
        if h:
            legend_handles.append(h)

    # Stack trajectory traces with NaN separators for fast vectorized line drawing
    if trajectories:
        selected_traces = [trace[:, dims] for trace in trajectories[:max_trajectories] if len(trace) > 0]
        if selected_traces:
            n_total = sum(len(t) for t in selected_traces) + len(selected_traces)
            combined = np.full((n_total, 2), np.nan, dtype=np.float32)
            offset = 0
            for trace in selected_traces:
                n = len(trace)
                combined[offset : offset + n] = trace
                offset += n + 1

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
                rasterized=True,
            )

    # Set axes limits, labels, title, and save to output directory
    ax.set_xlim(eval_env.obs_low[dims[0]], eval_env.obs_high[dims[0]])
    ax.set_ylim(eval_env.obs_low[dims[1]], eval_env.obs_high[dims[1]])
    ax.set_xlabel(base_model.state_variables[dims[0]])
    ax.set_ylabel(base_model.state_variables[dims[1]])
    ax.set_title(f"PPO trajectories ({base_model.__class__.__name__})")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "rl_trajectories.pdf", format="pdf", bbox_inches="tight", dpi=200)
    plt.savefig(output_dir / "rl_trajectories.png", format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# Rollout evaluation episodes under trained policy and record visited cells and trajectories
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

    # Execute rollouts from initial state neighborhood
    for ep_idx in range(episodes):
        obs, _ = eval_env.reset(seed=seed if ep_idx == 0 else None, options={"testing": True})
        visited_cells.add(eval_env.state_to_cell(obs))

        trace = [obs.copy()]
        for _ in range(cfg.max_steps):
            # Quantize observation to cell center if discrete action grid is enabled
            if discrete_actions is not None:
                obs_q = np.clip(eval_env.cell_to_center(eval_env.state_to_cell(obs)), eval_env.obs_low, eval_env.obs_high)
            else:
                obs_q = obs

            # Query policy and snap continuous action to nearest discrete action
            norm_obs = norm_env.normalize_obs(np.expand_dims(obs_q, axis=0))[0]
            action, _ = model.predict(norm_obs, deterministic=True)

            if discrete_actions is not None:
                dists = np.linalg.norm(discrete_actions - action, axis=1)
                action = discrete_actions[np.argmin(dists)]

            obs, _, terminated, truncated, info = eval_env.step(action, noise_factor=1.0)
            visited_cells.add(info["cell"])
            trace.append(obs.copy())

            if terminated or truncated:
                if terminated and info["in_goal"]:
                    reached_goal += 1
                break

        trajectories.append(np.asarray(trace))

    logger.info(f"- Evaluation rollouts completed in {time() - t:.2f} seconds.")
    t = time()

    # Generate 2D rollout visualization
    plot_rl_trajectories(base_model, eval_env, trajectories, dims, Path(getattr(args, "output_dir", "output")))
    logger.info(f"- Rollouts plotted completed in {time() - t:.2f} seconds.")

    total_cells = int(np.prod(base_model.partition["number_per_dim"]))
    return reached_goal, visited_cells, total_cells


# D-dimensional summed area table (prefix sum) with periodic wrapping for O(2^D) box counting
class SpatialPrefixSum:
    def __init__(self, active_mask, number_per_dim, wrap):
        self.nd = np.asarray(number_per_dim, dtype=np.int64)
        self.D = len(self.nd)

        # Reshape flat active mask to grid, doubling wrapping dimensions
        grid = active_mask.reshape(tuple(self.nd)).astype(np.int32)
        for d in range(self.D):
            if wrap[d]:
                grid = np.concatenate([grid, grid], axis=d)

        # Pad prefix sum grid with 1-element zero margin along each axis
        grid_shape = grid.shape
        padded_shape = tuple(s + 1 for s in grid_shape)
        P = np.zeros(padded_shape, dtype=np.int32)
        P[tuple(slice(1, None) for _ in range(self.D))] = grid

        # Compute cumulative sum along each dimension
        for d in range(self.D):
            np.cumsum(P, axis=d, out=P)

        # Precompute flat row-major strides into padded prefix sum array
        padded_strides = np.ones(self.D, dtype=np.int64)
        for d in range(self.D - 2, -1, -1):
            padded_strides[d] = padded_strides[d + 1] * padded_shape[d + 1]

        self.prefix_flat = P.ravel()
        self.prefix_strides = padded_strides
        self.wrap = np.asarray(wrap, dtype=bool)

        # Precompute 2^D inclusion-exclusion corners and alternating parity signs
        corners = np.array(list(itertools.product([False, True], repeat=self.D)))
        self.corners = corners
        self.signs = np.where(np.sum(corners, axis=1) % 2 == 0, 1, -1).astype(np.int64)

    # Count active cells in axis-aligned bounding boxes in O(2^D) via inclusion-exclusion
    def count_boxes(self, raw_lbs, raw_ubs):
        batch_shape = raw_lbs.shape[:-1]
        query_lbs = np.empty_like(raw_lbs, dtype=np.int64)
        query_ubs = np.empty_like(raw_ubs, dtype=np.int64)
        empty = np.zeros(batch_shape, dtype=bool)

        # Normalize bounds handling wrapping dimensions and clipping out-of-grid queries
        for d in range(self.D):
            if self.wrap[d]:
                raw_span = raw_ubs[..., d] - raw_lbs[..., d] + 1
                span = np.clip(raw_span, 0, self.nd[d])
                query_lbs[..., d] = raw_lbs[..., d] % self.nd[d]
                query_ubs[..., d] = query_lbs[..., d] + span - 1
                empty |= (span <= 0)
            else:
                empty |= (raw_ubs[..., d] < 0) | (raw_lbs[..., d] >= self.nd[d])
                query_lbs[..., d] = np.clip(raw_lbs[..., d], 0, self.nd[d] - 1)
                query_ubs[..., d] = np.clip(raw_ubs[..., d], 0, self.nd[d] - 1)

        # Precompute lower and upper bound index contributions
        lb_contrib = query_lbs * self.prefix_strides
        ub_contrib = (query_ubs + 1) * self.prefix_strides

        # Accumulate signed corner contributions from flattened prefix sum table
        counts = np.zeros(batch_shape, dtype=np.int64)
        for i in range(len(self.corners)):
            contrib = np.where(self.corners[i], lb_contrib, ub_contrib)
            counts += self.signs[i] * self.prefix_flat[np.sum(contrib, axis=-1)].astype(np.int64)

        counts[empty] = 0
        return counts


# Generate Cartesian product grid of integer coordinate offsets for inflation box
def _inflation_offsets(inflation_rate):
    axes = [np.arange(int(lo), int(hi) + 1) for lo, hi in inflation_rate]
    return np.stack([grid.ravel() for grid in np.meshgrid(*axes, indexing="ij")], axis=1)


# JAX jitted vectorized check for cell neighbor coordinates within grid bounds
@jax.jit
def _neighbors_in_bounds(cells, offsets, number_per_dim):
    def in_bounds_of(cell):
        neighbors = cell + offsets
        return jnp.all((neighbors >= 0) & (neighbors < number_per_dim), axis=1)

    return jax.vmap(in_bounds_of)(cells)


# Inflate set of visited cells into neighbor box and return deduplicated valid grid cells
def _inflate_cells(visited_cells, inflation_rate, number_per_dim):
    dim = len(number_per_dim)
    cells = np.asarray(list(visited_cells), dtype=np.int64).reshape(-1, dim)
    if len(cells) == 0:
        return np.zeros((0, dim), dtype=int)

    offsets = _inflation_offsets(inflation_rate)
    strides = _compute_linear_strides(number_per_dim)

    # Linearize multi-index coordinates using row-major grid strides
    base_ids = cells @ strides
    offset_ids = offsets @ strides

    offsets_jax = jnp.asarray(offsets, dtype=jnp.int32)
    number_per_dim_jax = jnp.asarray(number_per_dim, dtype=jnp.int32)
    batch = max(1, INFLATE_IDS_PER_BATCH // len(offsets))

    # Process batches to limit peak memory, deduplicating valid neighbor IDs per batch
    batches = []
    for start in tqdm(range(0, len(cells), batch), desc="Inflate visited cells"):
        in_bounds = _neighbors_in_bounds(
            jnp.asarray(cells[start:start + batch], dtype=jnp.int32),
            offsets_jax,
            number_per_dim_jax,
        )
        ids = base_ids[start:start + batch, None] + offset_ids[None, :]
        batches.append(np.unique(np.where(np.asarray(in_bounds), ids, -1)))

    # Concatenate unique IDs and unravel to D-dimensional grid indices
    ids = np.unique(np.concatenate(batches)) if batches else np.zeros(0, dtype=np.int64)
    ids = ids[ids >= 0]
    return ((ids[:, None] // strides) % number_per_dim).astype(int)


# JAX jitted batched forward reachable set (FRS) interval bounds evaluation
@partial(jax.jit, static_argnums=(0,))
def _compute_batch_frs_bounds(step_set_fn, s_mins, s_maxs, actions_batch):
    def _per_state(s_min, s_max, actions):
        return jax.vmap(lambda u: step_set_fn(s_min, s_max, u, u))(actions)
    return jax.vmap(_per_state)(s_mins, s_maxs, actions_batch)


# Convert cell coordinates to continuous FRS interval boxes accounting for noise support
def _compute_frs_index_boxes(model, val_env, cell_coords, actions_batch, noise_support):
    s_mins = val_env.obs_low + cell_coords.astype(np.float32) * val_env.bin_widths
    s_maxs = s_mins + val_env.bin_widths
    frs_mins, frs_maxs = _compute_batch_frs_bounds(
        model.step_set, s_mins, s_maxs, jnp.asarray(actions_batch, dtype=jnp.float32)
    )
    raw_idx_lbs = np.floor((np.asarray(frs_mins, dtype=np.float32) - noise_support - val_env.obs_low) / val_env.bin_widths).astype(int)
    raw_idx_ubs = np.floor((np.asarray(frs_maxs, dtype=np.float32) + noise_support - val_env.obs_low) / val_env.bin_widths).astype(int)
    return raw_idx_lbs, raw_idx_ubs


# Vectorized extraction of discrete grid cells covered by reachability bounding boxes
def _extract_frs_cells_vectorized(raw_idx_lbs, raw_idx_ubs, number_per_dim, strides, wrap):
    N, K, D = raw_idx_lbs.shape
    spans = np.maximum(1, raw_idx_ubs - raw_idx_lbs + 1)
    max_spans = np.max(spans, axis=(0, 1))

    # Build relative offset grid for box rasterization
    offset_axes = [np.arange(s, dtype=np.int64) for s in max_spans]
    offsets = offset_axes[0][:, None] if D == 1 else np.stack([g.ravel() for g in np.meshgrid(*offset_axes, indexing="ij")], axis=-1)

    M = offsets.shape[0]
    flat_cells = np.zeros((N, K, M), dtype=np.int64)
    valid_mask = np.ones((N, K, M), dtype=bool)

    # Compute flat linear cell IDs for each offset with periodic wrapping and bounds checks
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


# Forward reachable set (FRS) guided BFS expansion using prefix sum box queries
def _smart_inflate_cells(
    visited,
    model,
    val_env,
    ppo,
    vec_env,
    discrete_actions,
    args,
    number_per_dim,
    noise_support_ratio=0.5,
):
    dim = len(number_per_dim)
    visited_arr = np.asarray(list(visited), dtype=np.int64).reshape(-1, dim) if len(visited) > 0 else np.zeros((0, dim), dtype=np.int64)
    strides = _compute_linear_strides(number_per_dim)
    total_grid_size = int(np.prod(number_per_dim))

    # Initialize boolean occupancy mask of active grid states
    active_states_mask = np.zeros(total_grid_size, dtype=bool)
    if len(visited_arr) > 0:
        active_states_mask[np.dot(visited_arr, strides)] = True

    wrap = np.asarray(getattr(model, "wrap", np.zeros(model.n, dtype=bool)), dtype=bool)

    # Query policy for greedy action in each initially visited cell
    if len(visited_arr) > 0:
        obs_batch_init = np.asarray(val_env.obs_low + (visited_arr.astype(np.float32) + 0.5) * val_env.bin_widths, dtype=np.float32)
        _, init_rl_actions = find_policy_actions_batch(obs_batch_init, ppo, vec_env, discrete_actions, num=1)
    else:
        init_rl_actions = np.zeros((0, discrete_actions.shape[1]), dtype=np.float32)

    noise_support = 0.0
    if hasattr(model, "noise") and isinstance(model.noise, dict) and "support_radius" in model.noise:
        noise_support = model.noise["support_radius"] * noise_support_ratio

    # Phase 1: Expand reachable cells from initial rollout trajectory under greedy RL actions
    phase1_added_count = 0
    pbar_p1 = tqdm(total=len(visited_arr), desc="Phase 1: Batched FRS expansion")

    visited_arr_all = np.asarray(visited_arr, dtype=np.float32)
    init_rl_actions_all_batch = init_rl_actions[:, None, :]
    p1_new_queue_flats = []

    for ch_start in range(0, len(visited_arr), CHUNK_SIZE):
        ch_end = min(ch_start + CHUNK_SIZE, len(visited_arr))
        visited_chunk = visited_arr_all[ch_start:ch_end]
        init_actions_chunk = init_rl_actions_all_batch[ch_start:ch_end]

        raw_idx_lbs_p1, raw_idx_ubs_p1 = _compute_frs_index_boxes(model, val_env, visited_chunk, init_actions_chunk, noise_support)
        flat_cells_p1, valid_mask_p1 = _extract_frs_cells_vectorized(raw_idx_lbs_p1, raw_idx_ubs_p1, number_per_dim, strides, wrap)

        flats_p1 = flat_cells_p1[:, 0, :]
        valid_p1 = valid_mask_p1[:, 0, :]

        new_flats_p1 = flats_p1[valid_p1 & (~active_states_mask[flats_p1])]
        if len(new_flats_p1) > 0:
            unique_flats = np.unique(new_flats_p1)
            active_states_mask[unique_flats] = True
            p1_new_queue_flats.append(unique_flats)
            phase1_added_count += len(unique_flats)

        pbar_p1.update(ch_end - ch_start)

    pbar_p1.close()

    queue_flats = np.unique(np.concatenate(p1_new_queue_flats)) if p1_new_queue_flats else np.array([], dtype=np.int64)
    logger.info(f"- Phase 1 complete: Added {phase1_added_count} states. Queue size: {len(queue_flats)}.")

    # Phase 2: Breadth-first search queue expansion with prefix-sum action scoring
    prefix_sum = SpatialPrefixSum(active_states_mask, number_per_dim, wrap)
    phase2_added_count = 0
    p2_iter = 0

    while len(queue_flats) > 0:
        p2_iter += 1
        N_q = len(queue_flats)
        pbar_p2 = tqdm(total=N_q, desc=f"Phase 2 (iter {p2_iter}): Queue FRS expansion", unit="state")

        # Query RL policy for top-K candidate actions per queued state
        queue_coords = np.stack(np.unravel_index(queue_flats, number_per_dim), axis=-1)
        obs_batch_queue = np.asarray(val_env.obs_low + (queue_coords.astype(np.float32) + 0.5) * val_env.bin_widths, dtype=np.float32)
        queue_actions_all, _ = find_policy_actions_batch(obs_batch_queue, ppo, vec_env, discrete_actions, num=args.RL_actions_per_state)

        p2_iter_new_flats = []
        for ch_start in range(0, N_q, CHUNK_SIZE):
            ch_end = min(ch_start + CHUNK_SIZE, N_q)
            queue_chunk = queue_coords[ch_start:ch_end]
            actions_chunk = queue_actions_all[ch_start:ch_end]

            # Compute reachability boxes and evaluate prefix-sum active counts for all actions
            raw_idx_lbs_q, raw_idx_ubs_q = _compute_frs_index_boxes(model, val_env, queue_chunk, actions_chunk, noise_support)
            active_counts = prefix_sum.count_boxes(raw_idx_lbs_q, raw_idx_ubs_q)
            best_act_indices = np.argmax(active_counts, axis=1)

            # Rasterize and collect newly discovered cells for winning action only
            N_chunk = ch_end - ch_start
            win_lbs = raw_idx_lbs_q[np.arange(N_chunk), best_act_indices][:, None, :]
            win_ubs = raw_idx_ubs_q[np.arange(N_chunk), best_act_indices][:, None, :]

            flat_cells_q, valid_mask_q = _extract_frs_cells_vectorized(win_lbs, win_ubs, number_per_dim, strides, wrap)
            win_flats = flat_cells_q[:, 0, :]
            win_valid = valid_mask_q[:, 0, :]

            chunk_new_flats = win_flats[win_valid & (~active_states_mask[win_flats])]
            if len(chunk_new_flats) > 0:
                p2_iter_new_flats.append(chunk_new_flats)

            pbar_p2.update(N_chunk)

        # Update occupancy mask and rebuild spatial prefix sum index for next iteration
        if p2_iter_new_flats:
            unique_new = np.unique(np.concatenate(p2_iter_new_flats))
            active_states_mask[unique_new] = True
            queue_flats = unique_new
            added_this_iter = len(unique_new)
            phase2_added_count += added_this_iter
            prefix_sum = SpatialPrefixSum(active_states_mask, number_per_dim, wrap)
        else:
            queue_flats = np.array([], dtype=np.int64)
            added_this_iter = 0

        pbar_p2.set_postfix({"added_iter": added_this_iter, "total_active": int(np.sum(active_states_mask))})
        pbar_p2.close()

    logger.info(f"- Phase 2 complete: Added {phase2_added_count} states. Total active states: {int(np.sum(active_states_mask))}.")
    all_active_flats = np.where(active_states_mask)[0]
    return np.stack(np.unravel_index(all_active_flats, number_per_dim), axis=-1).astype(int)


# Main pipeline: PPO training, policy evaluation, tube expansion, and action extraction
def find_active(model, args, previous_cells=None):
    cfg = RLConfig(
        max_steps=args.max_steps,
        goal_reward=args.goal_reward,
        unsafe_penalty=args.unsafe_penalty,
        out_of_bounds_penalty=args.out_of_bounds_penalty,
        revisit_penalty=args.revisit_penalty,
        distance_reward=getattr(args, "distance_reward", 0.0),
        per_step_reward=getattr(args, "per_step_reward", 0.0),
    )

    # Build vectorized training environment and single evaluation environment
    vec_env = _build_vec_env(
        base_model=model,
        cfg=cfg,
        n_envs=args.n_envs,
        use_subproc=args.subproc,
        previous_cells=previous_cells,
        seed=args.seed,
    )
    val_env = BenchmarkRLEnv(model, cfg)

    # Configure neural network policy and value function architectures
    pi_arch = args.pi_arch if args.pi_arch is not None else model.pi_arch
    vf_arch = args.vf_arch if args.vf_arch is not None else model.vf_arch
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=pi_arch, vf=vf_arch))

    # Train PPO agent using continuous action spaces
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

    # Discretize continuous control space into Cartesian grid of discrete actions
    discrete_actions_per_dim = [
        np.linspace(model.uMin[i], model.uMax[i], num=model.num_actions[i])
        for i in range(len(model.num_actions))
    ]
    discrete_actions = np.array(list(itertools.product(*discrete_actions_per_dim)), dtype=np.float32)

    # Run evaluation rollouts to identify visited state cells
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

    # Expand visited cell tube using standard box inflation or smart FRS reachable expansion
    number_per_dim = np.asarray(model.partition["number_per_dim"], dtype=np.int64)

    if args.tube_method == "inflation":
        active_states = _inflate_cells(newly_visited, model.inflation_rate, number_per_dim)
    elif args.tube_method == "smart":
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

    # Extract top-K discrete actions and greedy policy for all active states
    obs_batch = np.asarray(val_env.obs_low + (active_states.astype(np.float32) + 0.5) * val_env.bin_widths, dtype=np.float32)
    top_k, rl_policy = find_policy_actions_batch(obs_batch, ppo, vec_env, discrete_actions, num=args.RL_actions_per_state)
    active_actions = {tuple(cell): top_k[i] for i, cell in enumerate(active_states.tolist())}

    logger.info(f"- Active state/action extraction completed in {time() - t:.2f} seconds.")
    return active_states, active_actions, rl_policy
