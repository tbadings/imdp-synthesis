from functools import partial
import logging

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from core.abstraction.partition import _compute_linear_strides
from .config import CHUNK_SIZE
from .env import JaxBenchmarkEnv
from .models import ActorCritic, RunningMeanStd, find_policy_actions_batch

logger = logging.getLogger(__name__)


# =============================================================================
# Fixed-rate Inflation Method
# =============================================================================

def _inflate_cells(visited_cells, inflation_rate, number_per_dim):
    """Inflate visited grid cells"""
    dim = len(number_per_dim)
    cells = np.asarray(list(visited_cells), dtype=np.int64).reshape(-1, dim)

    # Generate all coordinate offsets inside the inflation box
    axes = [np.arange(int(lo), int(hi) + 1, dtype=np.int64) for lo, hi in inflation_rate]
    offsets = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim)

    # Broadcast add offsets in chunks
    strides = _compute_linear_strides(number_per_dim)
    unique_ids = []

    for i in range(0, len(cells), CHUNK_SIZE):
        chunk = (cells[i : i + CHUNK_SIZE, None, :] + offsets[None, :, :]).reshape(-1, dim)
        valid = chunk[np.all((chunk >= 0) & (chunk < number_per_dim), axis=1)]
        if len(valid) > 0:
            unique_ids.append(np.unique(valid @ strides))

    all_ids = np.unique(np.concatenate(unique_ids))
    return np.stack(np.unravel_index(all_ids, number_per_dim), axis=-1).astype(int)


# =============================================================================
# Smart Reachability-Guided Tube Method
# =============================================================================

@partial(jax.jit, static_argnums=(0,))
def _compute_batch_frs(step_set_fn, s_mins, s_maxs, actions):
    """Batched evaluation of forward reachable sets over state intervals and candidate actions."""
    return jax.vmap(
        lambda s_min, s_max, acts: jax.vmap(lambda u: step_set_fn(s_min, s_max, u, u))(acts)
    )(s_mins, s_maxs, actions)


def _compute_frs_boxes(model, val_env, cell_coords, actions_batch, noise_support):
    """Compute discrete grid coordinate bounding boxes [lbs, ubs] for reachable sets."""
    s_mins = val_env.obs_low + cell_coords.astype(np.float32) * val_env.bin_widths
    s_maxs = s_mins + val_env.bin_widths
    frs_mins, frs_maxs = _compute_batch_frs(
        model.step_set, s_mins, s_maxs, jnp.asarray(actions_batch, dtype=jnp.float32)
    )
    lbs = np.floor((np.asarray(frs_mins, dtype=np.float32) - noise_support - val_env.obs_low) / val_env.bin_widths).astype(np.int64)
    ubs = np.floor((np.asarray(frs_maxs, dtype=np.float32) + noise_support - val_env.obs_low) / val_env.bin_widths).astype(np.int64)
    return lbs, ubs


def _extract_box_cells(lbs, ubs, number_per_dim, wrap, strides):
    """Extract flat cell indices and validity masks for arbitrary batches of bounding boxes."""
    spans = np.maximum(1, ubs - lbs + 1)
    max_spans = np.max(spans.reshape(-1, spans.shape[-1]), axis=0)
    axes = [np.arange(s, dtype=np.int64) for s in max_spans]
    offsets = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(number_per_dim))

    coords = lbs[..., None, :] + offsets
    valid = np.all(coords <= ubs[..., None, :], axis=-1)

    for d, is_wrap in enumerate(wrap):
        if is_wrap:
            coords[..., d] %= number_per_dim[d]
        else:
            valid &= (coords[..., d] >= 0) & (coords[..., d] < number_per_dim[d])
            coords[..., d] = np.clip(coords[..., d], 0, number_per_dim[d] - 1)

    flat_ids = np.sum(coords * strides, axis=-1)
    return flat_ids, valid


def _smart_inflate_cells(
    visited,
    model,
    val_env: JaxBenchmarkEnv,
    actor_critic: ActorCritic,
    params,
    rms_obs: RunningMeanStd,
    discrete_actions,
    args,
    number_per_dim,
    noise_support_ratio=0.5,
):
    """Reachability-guided BFS tube expansion using forward reachable sets and policy actions."""
    dim = len(number_per_dim)
    visited_arr = np.asarray(list(visited), dtype=np.int64).reshape(-1, dim) if len(visited) > 0 else np.zeros((0, dim), dtype=np.int64)
    if len(visited_arr) == 0:
        return np.zeros((0, dim), dtype=int)

    strides = _compute_linear_strides(number_per_dim)
    active_mask = np.zeros(int(np.prod(number_per_dim)), dtype=bool)
    active_mask[visited_arr @ strides] = True
    wrap = np.asarray(getattr(model, "wrap", np.zeros(model.n, dtype=bool)), dtype=bool)

    noise_support = 0.0
    if hasattr(model, "noise") and isinstance(model.noise, dict) and "support_radius" in model.noise:
        noise_support = model.noise["support_radius"] * noise_support_ratio

    def _expand_cells(coords, actions, desc):
        """Helper to compute FRS, score candidate actions, and collect new active cells."""
        new_flats = []
        pbar = tqdm(total=len(coords), desc=desc, unit="state")
        for i in range(0, len(coords), CHUNK_SIZE):
            chunk_c = coords[i : i + CHUNK_SIZE]
            chunk_a = actions[i : i + CHUNK_SIZE]

            lbs, ubs = _compute_frs_boxes(model, val_env, chunk_c, chunk_a, noise_support)
            flats, valid = _extract_box_cells(lbs, ubs, number_per_dim, wrap, strides)

            if chunk_a.shape[1] > 1:
                scores = np.sum(active_mask[flats] & valid, axis=-1)
                best_idx = np.argmax(scores, axis=-1)
                flats = flats[np.arange(len(flats)), best_idx]
                valid = valid[np.arange(len(valid)), best_idx]
            else:
                flats = flats[:, 0]
                valid = valid[:, 0]

            new_chunk = flats[valid & ~active_mask[flats]]
            if len(new_chunk) > 0:
                new_flats.append(new_chunk)
            pbar.update(len(chunk_c))

        pbar.close()
        if not new_flats:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(new_flats))

    # Phase 1: Expand initial trajectory under greedy RL actions
    obs_init = np.asarray(val_env.obs_low + (visited_arr.astype(np.float32) + 0.5) * val_env.bin_widths, dtype=np.float32)
    init_actions, _ = find_policy_actions_batch(obs_init, actor_critic, params, rms_obs, discrete_actions, num=1)

    queue = _expand_cells(visited_arr, init_actions, desc="Phase 1: Batched FRS expansion")
    active_mask[queue] = True
    logger.info(f"- Phase 1 complete: Added {len(queue)} states. Queue size: {len(queue)}.")

    # Phase 2: BFS tube expansion using top-K candidate actions
    phase2_added = 0
    p2_iter = 0
    while len(queue) > 0:
        p2_iter += 1
        queue_coords = np.stack(np.unravel_index(queue, number_per_dim), axis=-1)
        obs_queue = np.asarray(val_env.obs_low + (queue_coords.astype(np.float32) + 0.5) * val_env.bin_widths, dtype=np.float32)
        cand_actions, _ = find_policy_actions_batch(
            obs_queue, actor_critic, params, rms_obs, discrete_actions, num=args.RL_actions_per_state
        )

        new_flats = _expand_cells(queue_coords, cand_actions, desc=f"Phase 2 (iter {p2_iter}): Queue FRS expansion")
        active_mask[new_flats] = True
        queue = new_flats
        phase2_added += len(new_flats)

    logger.info(f"- Phase 2 complete: Added {phase2_added} states. Total active states: {int(np.sum(active_mask))}.")
    all_active = np.where(active_mask)[0]
    return np.stack(np.unravel_index(all_active, number_per_dim), axis=-1).astype(int)
