from functools import partial
import itertools
import logging
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from core.abstraction.partition import _compute_linear_strides
from .config import RLConfig

logger = logging.getLogger(__name__)

CHUNK_SIZE = 16384
_DENSE_INFLATION_BYTES = 128 * 1024 * 1024
_ID_CHUNK_SIZE = 262144
_PREFIX_BYTES = 256 * 1024 * 1024


@partial(jax.jit, static_argnums=(1, 2))
def _dilate_mask(mask, rates, wrap):
    """Compile only Boolean dilation; no floating-point cell decisions change."""
    dim = mask.ndim
    for d, (lo, hi) in enumerate(rates):
        width = hi - lo + 1
        if lo == hi == 0:
            continue
        if wrap[d] and width >= mask.shape[d]:
            mask = jnp.broadcast_to(jnp.any(mask, axis=d, keepdims=True), mask.shape)
            continue
        window = [1] * dim
        window[d] = width
        padding = [(0, 0)] * dim
        if wrap[d]:
            # Move the first requested offset to zero. The remaining offsets
            # form one nonnegative window, which can be padded periodically
            # even when the original interval lies wholly above or below zero.
            mask = jnp.roll(mask, lo, axis=d)
            padding[d] = (width - 1, 0)
            mask = jnp.pad(mask, padding, mode="wrap")
            padding = [(0, 0)] * dim
        else:
            padding[d] = (hi, -lo)
        mask = jax.lax.reduce_window(mask, False, jax.lax.bitwise_or, window, [1] * dim, padding)
    return mask


class _SparseActiveCells:
    """Sorted occupancy IDs for grids too large for a dense Boolean mask."""

    def __init__(self, size):
        self.size = size
        self.ids = np.empty(0, dtype=np.int64)

    def __getitem__(self, ids):
        positions = np.searchsorted(self.ids, ids)
        if not len(self.ids):
            return np.zeros(np.shape(ids), dtype=bool)
        return (positions < len(self.ids)) & (self.ids[np.minimum(positions, len(self.ids) - 1)] == ids)

    def add(self, ids):
        self.ids = np.union1d(self.ids, ids)


def _cells_from_flat_ids(ids, number_per_dim):
    """Decode sorted IDs without allocating one full array per dimension."""
    cells = np.empty((len(ids), len(number_per_dim)), dtype=np.int64)
    for d, stride in enumerate(_compute_linear_strides(number_per_dim)):
        cells[:, d] = (ids // stride) % number_per_dim[d]
    return cells


def _unique_id_chunks(chunks):
    """Merge incrementally instead of retaining every duplicate candidate."""
    levels = []
    for chunk in chunks:
        ids = np.unique(chunk)
        if not len(ids):
            continue
        level = 0
        while level < len(levels) and levels[level] is not None:
            ids = np.union1d(levels[level], ids)
            levels[level] = None
            level += 1
        if level == len(levels):
            levels.append(ids)
        else:
            levels[level] = ids
    result = np.empty(0, dtype=np.int64)
    for ids in levels:
        if ids is not None:
            result = np.union1d(result, ids)
    return result


def _inflate_cells(visited_cells, inflation_rate, number_per_dim, wrap):
    """Inflate visited cells with a cropped dense Boolean mask."""
    dim = len(number_per_dim)
    # Accept both the ndarray produced by rollout extraction and the legacy set
    # of coordinate tuples. From here onward cells always has shape [N, D].
    cells = np.asarray(visited_cells if isinstance(visited_cells, np.ndarray) else list(visited_cells), dtype=np.int64).reshape(-1, dim)
    rates = np.array(inflation_rate, dtype=np.int64)
    wrap = np.asarray(wrap, dtype=bool)

    # Offsets farther than a complete nonperiodic axis cannot add cells, so
    # trim them before allocating the mask. Periodic axes retain their full
    # rates because their offsets wrap around the grid.
    rates[:, 0] = np.where(wrap, rates[:, 0], np.maximum(rates[:, 0], 1 - number_per_dim))
    rates[:, 1] = np.where(wrap, rates[:, 1], np.minimum(rates[:, 1], number_per_dim - 1))
    if not len(cells) or np.any(rates[:, 1] < rates[:, 0]):
        return np.empty((0, dim), dtype=int)

    # Determine the exact output box. Periodic axes retain their full extent
    # so dilation can wrap at the original grid boundary.
    cell_lower = cells.min(axis=0)
    cell_upper = cells.max(axis=0) + 1
    lower = np.where(wrap, 0, np.maximum(0, cell_lower + rates[:, 0]))
    upper = np.where(wrap, number_per_dim, np.minimum(number_per_dim, cell_upper + rates[:, 1]))
    if np.any(upper <= lower):
        return np.empty((0, dim), dtype=int)

    # Inflation intervals normally contain zero, in which case this canvas is
    # exactly the output box. Including the visited box as well also preserves
    # the legacy behavior for one-sided intervals such as (1, 3).
    canvas_lower = np.where(wrap, 0, np.minimum(lower, cell_lower))
    canvas_upper = np.where(wrap, number_per_dim, np.maximum(upper, cell_upper))
    canvas_shape = canvas_upper - canvas_lower

    # Translate into the cropped coordinate system and mark every visited
    # cell. Rectangular inflation is separable, so the jitted kernel dilates
    # the mask one state dimension at a time without constructing all offsets.
    mask = np.zeros(tuple(canvas_shape), dtype=bool)
    mask[tuple((cells - canvas_lower).T)] = True
    mask = np.asarray(
        _dilate_mask(mask, tuple(map(tuple, rates.tolist())), tuple(wrap.tolist()))
    )

    # Discard the extra source-only portion needed by one-sided intervals.
    output_slices = tuple(
        slice(int(lo - canvas_lo), int(hi - canvas_lo))
        for lo, hi, canvas_lo in zip(lower, upper, canvas_lower)
    )
    mask = mask[output_slices]

    # flatnonzero returns row-major IDs, so decoding them preserves the
    # deterministic ordering obtained by sorting global linear cell IDs.
    ids = np.flatnonzero(mask)
    result = _cells_from_flat_ids(ids, upper - lower)
    result += lower
    return result


@partial(jax.jit, static_argnums=(0,))
def _compute_batch_frs_bounds(step_set_fn, s_mins, s_maxs, actions_batch):
    def _per_state(s_min, s_max, actions):
        return jax.vmap(lambda u: step_set_fn(s_min, s_max, u, u))(actions)
    return jax.vmap(_per_state)(s_mins, s_maxs, actions_batch)


def _build_prefix_sum(active_mask, number_per_dim):
    dtype = np.int32 if active_mask.size <= np.iinfo(np.int32).max else np.int64
    prefix_table = np.zeros(tuple(np.asarray(number_per_dim) + 1), dtype=dtype)
    prefix_table[(slice(1, None),) * len(number_per_dim)] = active_mask.reshape(number_per_dim)
    for d in range(len(number_per_dim)):
        np.cumsum(prefix_table, axis=d, out=prefix_table)
    return prefix_table.ravel(), _compute_linear_strides(prefix_table.shape)


def _iter_box_cell_ids(lbs, ubs, number_per_dim, strides, wrap):
    """Yield (flat cell IDs, box indices) without padding to the largest box.

    A linear position in the concatenated boxes identifies its owning box via
    searchsorted. Mixed-radix decoding then needs only O(chunk size) scratch,
    even when a single reachable box spans millions of cells.
    """
    lbs = np.asarray(lbs).reshape(-1, len(number_per_dim))
    ubs = np.asarray(ubs).reshape(lbs.shape)
    wrap = np.asarray(wrap, dtype=bool)
    lower = np.where(wrap, lbs, np.maximum(lbs, 0))
    upper = np.where(wrap, ubs, np.minimum(ubs, number_per_dim - 1))
    spans = np.maximum(0, upper - lower + 1)
    spans = np.where(wrap, np.minimum(spans, number_per_dim), spans)
    volumes = np.prod(spans, axis=1, dtype=np.int64)
    ends = np.cumsum(volumes, dtype=np.int64)
    if not len(ends):
        return
    starts = ends - volumes
    for start in range(0, int(ends[-1]), _ID_CHUNK_SIZE):
        positions = np.arange(start, min(start + _ID_CHUNK_SIZE, int(ends[-1])), dtype=np.int64)
        owners = np.searchsorted(ends, positions, side="right")
        remainder = positions - starts[owners]
        ids = np.zeros(len(positions), dtype=np.int64)
        for d in range(len(number_per_dim) - 1, -1, -1):
            width = spans[owners, d]
            coords = lower[owners, d] + remainder % width
            remainder //= width
            if wrap[d]:
                coords %= number_per_dim[d]
            ids += coords * strides[d]
        yield ids, owners


def _box_count_active(active_mask, lbs, ubs, number_per_dim, strides):
    # The legacy prefix query clamps at every boundary, including periodic
    # dimensions. Counting and successor enumeration intentionally differ here.
    counts = np.zeros(int(np.prod(lbs.shape[:-1])), dtype=np.int64)
    for ids, owners in _iter_box_cell_ids(lbs, ubs, number_per_dim, strides, np.zeros(len(number_per_dim), dtype=bool)):
        counts += np.bincount(owners[active_mask[ids]], minlength=len(counts))
    return counts.reshape(lbs.shape[:-1]).astype(np.int32)


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
    def new_chunks():
        for start in range(0, len(coords), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(coords))
            c_chunk = coords[start:end]
            a_chunk = actions_batch[start:end]

            s_mins = val_env.obs_low + c_chunk * val_env.bin_widths
            s_maxs = s_mins + val_env.bin_widths

            frs_mins, frs_maxs = _compute_batch_frs_bounds(
                model.step_set, s_mins, s_maxs, jnp.asarray(a_chunk, dtype=jnp.float32)
            )
            lbs = np.floor((np.asarray(frs_mins) - noise_support - val_env.obs_low) / val_env.bin_widths).astype(int)
            ubs = np.floor((np.asarray(frs_maxs) + noise_support - val_env.obs_low) / val_env.bin_widths).astype(int)

            if actions_batch.shape[1] > 1 and prefix_data is not None:
                if prefix_data is False:
                    counts = _box_count_active(active_mask, lbs, ubs, number_per_dim, strides)
                else:
                    prefix_flat, prefix_strides = prefix_data
                    counts = _box_count_prefix_sum(prefix_flat, prefix_strides, lbs, ubs, number_per_dim)
                best_acts = np.argmax(counts, axis=1)
                row_idx = np.arange(end - start)
                lbs, ubs = lbs[row_idx, best_acts], ubs[row_idx, best_acts]
            else:
                lbs, ubs = lbs[:, 0, :], ubs[:, 0, :]

            for ids, _ in _iter_box_cell_ids(lbs, ubs, number_per_dim, strides, model.wrap):
                yield ids[~active_mask[ids]]

    return _unique_id_chunks(new_chunks())


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
    grid_size = int(np.prod(number_per_dim))
    active_mask = np.zeros(grid_size, dtype=bool) if grid_size <= _DENSE_INFLATION_BYTES else _SparseActiveCells(grid_size)

    def add_active(ids):
        if isinstance(active_mask, _SparseActiveCells):
            active_mask.add(ids)
        else:
            active_mask[ids] = True

    visited_arr = np.asarray(visited if isinstance(visited, np.ndarray) else list(visited), dtype=np.int64).reshape(-1, dim)
    visited_ids = np.unique(np.dot(visited_arr, strides))
    add_active(visited_ids)
    active_count = len(visited_ids)
    noise_support = model.noise["support_radius"] * cfg.smart_tube_rate

    logger.info("Phase 1: Expanding FRS for visited states...")
    init_actions = agent.get_policy_actions(visited_arr, discrete_actions, num=1)
    queue_flats = _expand_cells_batch(
        visited_arr.astype(np.float32), init_actions, model, val_env, number_per_dim, strides, active_mask, noise_support=noise_support
    )
    add_active(queue_flats)
    active_count += len(queue_flats)
    logger.info("- Phase 1 complete: %d active cells. Queue size: %d.", active_count, len(queue_flats))

    p2_iter = 0
    while len(queue_flats) > 0:
        p2_iter += 1
        queue_coords = _cells_from_flat_ids(queue_flats, number_per_dim).astype(np.float32)
        queue_actions = agent.get_policy_actions(queue_coords, discrete_actions, num=cfg.RL_actions_per_state)
        prefix_data = None
        if queue_actions.shape[1] > 1:
            prefix_itemsize = 4 if active_mask.size <= np.iinfo(np.int32).max else 8
            prefix_bytes = int(np.prod(number_per_dim + 1, dtype=object)) * prefix_itemsize
            prefix_data = _build_prefix_sum(active_mask, number_per_dim) if prefix_bytes <= _PREFIX_BYTES and not isinstance(active_mask, _SparseActiveCells) else False

        new_flats = _expand_cells_batch(
            queue_coords, queue_actions, model, val_env, number_per_dim, strides, active_mask,
            prefix_data=prefix_data, noise_support=noise_support
        )
        add_active(new_flats)
        active_count += len(new_flats)
        queue_flats = new_flats
        logger.info("- Phase 2 iter %d: added %d cells. Total active: %d.", p2_iter, len(queue_flats), active_count)

    logger.info("- Phase 2 complete. Total active states: %d.", active_count)
    all_active_flats = active_mask.ids if isinstance(active_mask, _SparseActiveCells) else np.flatnonzero(active_mask)
    return _cells_from_flat_ids(all_active_flats, number_per_dim)


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
    return _inflate_cells(visited, cfg.inflation_rate, number_per_dim, model.wrap)


def rollout_sweep_priority(trajectories, active_states, env, chunk_size=1 << 20):
    """Per active state, the steps-to-goal of the nearest cell on a goal-reaching rollout.

    Lower values are closer to the goal, so sweeping states in ascending order carries value
    back along the rollouts within a single sweep. Returns None if no rollout reached the goal.
    """
    ref_cells, ref_steps = [], []
    for tr in trajectories:
        if not np.any(np.all((tr[-1] >= env.goal[:, 0]) & (tr[-1] <= env.goal[:, 1]), axis=-1)):
            continue
        ref_cells.append(np.clip((tr - env.obs_low) // env.bin_widths, 0, env.number_per_dim - 1).astype(np.int64))
        ref_steps.append(np.arange(len(tr) - 1, -1, -1))
    if not ref_cells:
        return None

    # A cell visited at several steps keeps its smallest steps-to-goal, so lookups are unambiguous.
    ref_steps = np.concatenate(ref_steps)
    order = np.argsort(ref_steps, kind="stable")
    ref_cells, first = np.unique(np.concatenate(ref_cells)[order], axis=0, return_index=True)
    ref_steps = ref_steps[order][first]

    # L-inf distance in grid cells matches the box-shaped tube inflation. Querying in chunks keeps
    # the float64 copy of the queried states small.
    tree = cKDTree(ref_cells)
    priority = np.empty(len(active_states), dtype=np.int32)
    for start in range(0, len(active_states), chunk_size):
        _, idx = tree.query(active_states[start:start + chunk_size], p=np.inf, workers=-1)
        priority[start:start + chunk_size] = ref_steps[idx]
    return priority
