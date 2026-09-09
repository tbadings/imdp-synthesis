from functools import partial
import itertools
import logging
import jax
import jax.numpy as jnp
import numpy as np

from core.abstraction.partition import _compute_linear_strides
from .config import RLConfig

logger = logging.getLogger(__name__)

CHUNK_SIZE = 16384


def _inflate_cells(visited_cells, inflation_rate, number_per_dim, wrap):
    """Inflate visited grid cells by a fixed ratio."""
    dim = len(number_per_dim)
    cells = np.asarray(list(visited_cells), dtype=np.int64).reshape(-1, dim)

    axes = [np.arange(int(lo), int(hi) + 1, dtype=np.int64) for lo, hi in inflation_rate]
    offsets = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim)

    strides = _compute_linear_strides(number_per_dim)
    unique_ids = []

    for i in range(0, len(cells), CHUNK_SIZE):
        chunk = (cells[i : i + CHUNK_SIZE, None, :] + offsets[None, :, :]).reshape(-1, dim)
        valid = np.all((chunk >= 0) & (chunk < number_per_dim) | wrap, axis=1)
        valid_cells = np.where(wrap, chunk[valid] % number_per_dim, chunk[valid])
        unique_ids.append(valid_cells @ strides)

    all_ids = np.unique(np.concatenate(unique_ids))
    return np.stack(np.unravel_index(all_ids, number_per_dim), axis=-1).astype(int)


@partial(jax.jit, static_argnums=(0,))
def _compute_batch_frs_bounds(step_set_fn, s_mins, s_maxs, actions_batch):
    def _per_state(s_min, s_max, actions):
        return jax.vmap(lambda u: step_set_fn(s_min, s_max, u, u))(actions)
    return jax.vmap(_per_state)(s_mins, s_maxs, actions_batch)


def _extract_frs_cells(lbs, ubs, number_per_dim, strides, wrap):
    spans = ubs - lbs + 1
    offsets = np.stack(
        [g.ravel() for g in np.meshgrid(*[np.arange(s) for s in np.max(spans, axis=0)], indexing="ij")],
        axis=-1,
    )
    coords = lbs[:, None, :] + offsets
    valid_mask = np.all(coords <= ubs[:, None, :], axis=-1)

    flat_cells = np.zeros(coords.shape[:-1], dtype=np.int64)
    for d, (num_d, stride_d, is_wrap) in enumerate(zip(number_per_dim, strides, wrap)):
        c_d = coords[..., d]
        if is_wrap:
            flat_cells += (c_d % num_d) * stride_d
        else:
            valid_mask &= (c_d >= 0) & (c_d < num_d)
            flat_cells += np.clip(c_d, 0, num_d - 1) * stride_d

    return flat_cells, valid_mask


def _build_prefix_sum(active_mask, number_per_dim):
    grid = active_mask.reshape(number_per_dim).astype(np.int32)
    for d in range(len(number_per_dim)):
        grid = np.cumsum(grid, axis=d)
    prefix_table = np.pad(grid, [(1, 0)] * len(number_per_dim), mode="constant")
    return prefix_table.ravel(), _compute_linear_strides(prefix_table.shape)


def _box_count_prefix_sum(prefix_flat, prefix_strides, lbs, ubs, number_per_dim):
    D = len(number_per_dim)
    lbs_clamped = np.clip(lbs, 0, number_per_dim)
    ubs_clamped = np.clip(ubs + 1, lbs_clamped, number_per_dim)

    corners = np.array(list(itertools.product([0, 1], repeat=D)), dtype=np.int64)
    signs = np.array([(-1) ** (D - np.sum(c)) for c in corners], dtype=np.int32)

    counts = np.zeros(lbs.shape[:-1], dtype=np.int32)
    for c, sign in zip(corners, signs):
        corner_coords = np.where(c == 0, lbs_clamped, ubs_clamped)
        counts += sign * prefix_flat[np.dot(corner_coords, prefix_strides)]
    return counts


def _expand_cells_batch(
    coords,
    actions_batch,
    model,
    val_env,
    number_per_dim,
    strides,
    active_mask,
    prefix_data=None,
    noise_support=0.0,
):
    num_states = len(coords)
    new_flats_list = []

    for start in range(0, num_states, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, num_states)
        c_chunk = coords[start:end]
        a_chunk = actions_batch[start:end]

        s_mins = val_env.obs_low + c_chunk * val_env.bin_widths
        s_maxs = s_mins + val_env.bin_widths

        frs_mins, frs_maxs = _compute_batch_frs_bounds(
            model.step_set, s_mins, s_maxs, jnp.asarray(a_chunk, dtype=jnp.float32)
        )
        lbs = np.floor((np.asarray(frs_mins, dtype=np.float32) - noise_support - val_env.obs_low) / val_env.bin_widths).astype(int)
        ubs = np.floor((np.asarray(frs_maxs, dtype=np.float32) + noise_support - val_env.obs_low) / val_env.bin_widths).astype(int)

        if actions_batch.shape[1] > 1 and prefix_data is not None:
            prefix_flat, prefix_strides = prefix_data
            counts = _box_count_prefix_sum(prefix_flat, prefix_strides, lbs, ubs, number_per_dim)
            best_acts = np.argmax(counts, axis=1)
            row_idx = np.arange(end - start)
            lbs, ubs = lbs[row_idx, best_acts], ubs[row_idx, best_acts]
        else:
            lbs, ubs = lbs[:, 0, :], ubs[:, 0, :]

        flat_cells, valid_mask = _extract_frs_cells(lbs, ubs, number_per_dim, strides, model.wrap)
        is_new = valid_mask & (~active_mask[flat_cells])
        if np.any(is_new):
            new_flats_list.append(flat_cells[is_new])

    return np.unique(np.concatenate(new_flats_list)) if new_flats_list else np.empty(0, dtype=np.int64)


def _smart_inflate_cells(
    visited,
    model,
    val_env,
    agent,
    discrete_actions,
    cfg: RLConfig,
    number_per_dim,
):
    """Reachability-guided tube expansion (smart inflate)."""
    dim = len(number_per_dim)
    strides = _compute_linear_strides(number_per_dim)
    active_mask = np.zeros(int(np.prod(number_per_dim)), dtype=bool)

    visited_arr = np.asarray(list(visited), dtype=np.int64).reshape(-1, dim)
    active_mask[np.dot(visited_arr, strides)] = True
    noise_support = model.noise["support_radius"] * cfg.smart_tube_rate

    def _get_policy_actions(coords, num_actions):
        top_k, _ = agent.get_policy_actions(coords, discrete_actions, num=num_actions)
        return top_k

    logger.info("Phase 1: Expanding FRS for visited states...")
    init_actions = _get_policy_actions(visited_arr, num_actions=1)
    queue_flats = _expand_cells_batch(
        visited_arr.astype(np.float32), init_actions, model, val_env, number_per_dim, strides, active_mask, noise_support=noise_support
    )
    active_mask[queue_flats] = True
    logger.info("- Phase 1 complete: %d active cells. Queue size: %d.", int(np.sum(active_mask)), len(queue_flats))

    p2_iter = 0
    while len(queue_flats) > 0:
        p2_iter += 1
        queue_coords = np.stack(np.unravel_index(queue_flats, number_per_dim), axis=-1).astype(np.float32)
        queue_actions = _get_policy_actions(queue_coords, num_actions=cfg.RL_actions_per_state)
        prefix_data = _build_prefix_sum(active_mask, number_per_dim)

        new_flats = _expand_cells_batch(
            queue_coords, queue_actions, model, val_env, number_per_dim, strides, active_mask,
            prefix_data=prefix_data, noise_support=noise_support
        )
        active_mask[new_flats] = True
        queue_flats = new_flats
        logger.info("- Phase 2 iter %d: added %d cells. Total active: %d.", p2_iter, len(queue_flats), int(np.sum(active_mask)))

    logger.info("- Phase 2 complete. Total active states: %d.", int(np.sum(active_mask)))
    all_active_flats = np.where(active_mask)[0]
    return np.stack(np.unravel_index(all_active_flats, number_per_dim), axis=-1).astype(int)


def build_tube(visited, cfg: RLConfig, model, env, agent=None, discrete_actions=None):
    """Build active state space tube around RL rollouts using inflation or smart reachability."""
    number_per_dim = np.asarray(model.partition["number_per_dim"], dtype=np.int64)
    logger.info("Growing the tube around the RL rollouts (method: %s)...", cfg.tube_method)
    if cfg.tube_method == "smart":
        return _smart_inflate_cells(
            visited=visited, model=model, val_env=env,
            agent=agent, discrete_actions=discrete_actions,
            cfg=cfg, number_per_dim=number_per_dim,
        )
    rate = cfg.inflation_rate
    return _inflate_cells(visited, rate, number_per_dim, model.wrap)
