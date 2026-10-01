"""On-the-fly composition of forward-reachable-set successor state IDs from compact boxes.

The SVMDP abstraction stores, per (state, action, noise cell), only the grid-index box
[idx_lb, idx_ub] of the forward reachable set. The dynamic program needs the state IDs of the
cells inside that box (it takes a nondeterministic min over them). Materialising those IDs for
every (state, action, noise cell) up front costs prod(max_span) int32 per entry, which is
prohibitive (tens of GB for 3-D models); instead we recompose them on the fly inside the DP.

Compositional structure exploited here: a cell's linear key is separable across dimensions,

    key(g_0, ..., g_{D-1}) = sum_d clip(g_d) * stride_d,

so the M = prod(max_span) keys of a box are the outer sum of D per-dimension contribution
vectors (total length sum_d max_span_d). We never materialise the (M, D) index grid (no meshgrid).
For a dense (rectangular) partition the sorted linear keys are exactly 0..S-1, so the key maps to
a state ID by a direct gather into region_linear_state (no binary search). A sparse partition uses
a full-grid lookup table when the grid is small, and otherwise a two-level row lookup (see
_build_row_lookup); searchsorted over the keys is the last resort.
"""

import logging
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# Largest dense row table of the two-level lookup (int32 entries, i.e. at most 128 MB). Beyond this, the
# row of a cell is found by a binary search over the (much fewer) row keys instead.
ROW_LUT_MAX_CELLS = 1 << 25

# The padded state-ID runs of the rows may hold at most this many entries per state (gaps in the last
# dimension are padded); otherwise the two-level lookup is not built.
PAD_MAX_FACTOR = 4


class RowLookup(NamedTuple):
    '''Two-level successor lookup of a sparse partition (see _build_row_lookup).'''
    row_lut: jax.Array    # Row ID per cell of the leading-coordinate bounding box, -1 if absent (empty: not built)
    bb_lo: jax.Array      # Lower corner of that bounding box (shape [D-1])
    bb_shape: jax.Array   # Its extent (shape [D-1])
    bb_mult: jax.Array    # Its row-major strides (shape [D-1])
    row_key: jax.Array    # Sorted linear key of each row's leading coordinates (for the binary-search variant)
    row_mult: jax.Array   # Strides of those row keys (shape [D-1])
    row_start: jax.Array  # Start of each row's run in pad
    row_min: jax.Array    # First last-dimension index of each row
    row_max: jax.Array    # Last last-dimension index of each row
    pad: jax.Array        # State IDs of the rows' runs [row_min, row_max]; missing_state for gaps


def _build_row_lookup(num_per_dim, strides, region_linear_idx, region_linear_state, missing_state):
    '''
    Split the partition's sorted linear keys into rows of cells that share their leading D-1 grid
    coordinates. The last dimension has stride 1, so the cells of a row are consecutive keys, and a box
    then needs one row lookup per combination of its leading coordinates, after which its cells along
    the last dimension are consecutive entries of the row's (gap-padded) state-ID run.

    :return: RowLookup, or None if the padded runs would be much larger than the partition
    '''
    keys = np.asarray(region_linear_idx).astype(np.int64)
    if len(keys) == 0:
        return None
    key_state = np.asarray(region_linear_state).astype(np.int32)
    npd = np.asarray(num_per_dim).astype(np.int64)
    strides64 = np.asarray(strides).astype(np.int64)
    n_last = npd[-1]

    # Rows: runs of sorted keys with equal leading coordinates
    row_of_key = keys // n_last
    first = np.r_[True, row_of_key[1:] != row_of_key[:-1]]
    first_idx = np.flatnonzero(first)
    row_id = np.cumsum(first) - 1
    last_coord = keys % n_last
    row_min = last_coord[first_idx]
    row_max = last_coord[np.r_[first_idx[1:] - 1, len(keys) - 1]]
    hull = row_max - row_min + 1
    if hull.sum() > PAD_MAX_FACTOR * len(keys) + 1024:
        return None
    row_start = np.r_[0, np.cumsum(hull)[:-1]]
    pad = np.full(int(hull.sum()), int(missing_state), dtype=np.int32)
    pad[row_start[row_id] + last_coord - row_min[row_id]] = key_state

    # Dense row table over the bounding box of the rows' leading coordinates, if small enough
    lead = (keys[first_idx, None] // strides64[None, :-1]) % npd[None, :-1]
    bb_lo = lead.min(axis=0)
    bb_shape = lead.max(axis=0) - bb_lo + 1
    bb_mult = np.ones(len(bb_shape), dtype=np.int64)
    if len(bb_shape) > 1:
        bb_mult[:-1] = np.cumprod(bb_shape[1:][::-1])[::-1]
    if int(np.prod(bb_shape)) <= ROW_LUT_MAX_CELLS:
        row_lut = np.full(int(np.prod(bb_shape)), -1, dtype=np.int32)
        row_lut[(lead - bb_lo) @ bb_mult] = np.arange(len(first_idx), dtype=np.int32)
    else:
        row_lut = np.zeros(0, dtype=np.int32)

    # jnp.asarray (not jnp.array) keeps int64 keys intact, see core/jax_config.py
    key_dtype = jnp.asarray(strides).dtype
    lookup = RowLookup(
        row_lut=jax.device_put(row_lut), bb_lo=jnp.asarray(bb_lo, dtype=jnp.int32),
        bb_shape=jnp.asarray(bb_shape, dtype=jnp.int32), bb_mult=jnp.asarray(bb_mult, dtype=jnp.int32),
        row_key=jnp.asarray(row_of_key[first_idx], dtype=key_dtype), row_mult=jnp.asarray(strides64[:-1] // n_last, dtype=key_dtype),
        row_start=jnp.asarray(row_start, dtype=jnp.int32), row_min=jnp.asarray(row_min, dtype=jnp.int32),
        row_max=jnp.asarray(row_max, dtype=jnp.int32), pad=jax.device_put(pad))
    logger.info(f'- Successor lookup: {len(first_idx)} rows, '
                + (f'row table {row_lut.nbytes / 1e6:.0f} MB' if len(row_lut) else 'binary search over rows')
                + f', padded state IDs {pad.nbytes / 1e6:.0f} MB')
    return lookup


def _box_to_ids_rows(idx_lb, idx_ub, max_span, wrap, num_per_dim, rows, missing_state):
    '''State IDs of the cells of one box via the row lookup; same output (and order) as box_to_ids_single.'''
    lead_dims = len(max_span) - 1

    # Indices along every dimension, padded to max_span[d] by repeating idx_ub[d]. Wrapped dims fold
    # modulo; non-wrapped out-of-range indices are flagged and clipped to stay valid gather indices.
    res, oob = [], []
    for d in range(len(max_span)):
        col = jnp.minimum(jnp.arange(max_span[d]) + idx_lb[d], idx_ub[d])
        res.append(jnp.where(wrap[d], col % num_per_dim[d], jnp.clip(col, 0, num_per_dim[d] - 1)))
        oob.append((~wrap[d]) & ((col < 0) | (col >= num_per_dim[d])))

    # Row of every combination of leading coordinates (outer product over the leading dims)
    ok = jnp.ones((), dtype=bool)
    if rows.row_lut.shape[0] > 0:
        idx = jnp.zeros((), dtype=jnp.int32)
        for d in range(lead_dims):
            t = res[d] - rows.bb_lo[d]
            idx = idx[..., None] + jnp.clip(t, 0, rows.bb_shape[d] - 1) * rows.bb_mult[d]
            ok = ok[..., None] & ~oob[d] & (t >= 0) & (t < rows.bb_shape[d])
        row = jnp.where(ok, rows.row_lut[idx], -1).reshape(-1)
    else:
        rk = jnp.zeros((), dtype=rows.row_key.dtype)
        for d in range(lead_dims):
            rk = rk[..., None] + res[d].astype(rows.row_key.dtype) * rows.row_mult[d]
            ok = ok[..., None] & ~oob[d]
        rk, ok = rk.reshape(-1), ok.reshape(-1)
        pos = jnp.minimum(jnp.searchsorted(rows.row_key, rk, side='left'), rows.row_key.shape[0] - 1)
        row = jnp.where(ok & (rows.row_key[pos] == rk), pos, -1)

    # Cells along the last dimension are consecutive entries of the row's padded run
    z, z_oob = res[lead_dims], oob[lead_dims]
    r = jnp.maximum(row, 0)
    lo, hi = rows.row_min[r][:, None], rows.row_max[r][:, None]
    inside = (row[:, None] >= 0) & ~z_oob[None, :] & (z[None, :] >= lo) & (z[None, :] <= hi)
    pos = jnp.clip(rows.row_start[r][:, None] + z[None, :] - lo, 0, rows.pad.shape[0] - 1)
    return jnp.where(inside, rows.pad[pos], missing_state).reshape(-1)


def box_to_ids_single(idx_lb, idx_ub, max_span, wrap, num_per_dim, strides,
                      region_linear_idx, region_linear_state, missing_state, dense,
                      key_to_state=None, row_lookup=None):
    """Expand one box into the state IDs of the M = prod(max_span) cells it spans.

    :param idx_lb: Lower grid-index bound of the box (shape [D])
    :param idx_ub: Upper grid-index bound of the box (shape [D])
    :param max_span: Static per-dimension span (tuple of D Python ints; M = prod(max_span))
    :param wrap: Per-dimension wrap flags (shape [D], bool)
    :param num_per_dim: Cells per dimension (shape [D])
    :param strides: Row-major linear strides of the grid (shape [D])
    :param region_linear_idx: Sorted linear keys of present cells (shape [num_cells])
    :param region_linear_state: State IDs aligned to region_linear_idx (shape [num_cells])
    :param missing_state: Sentinel ID for out-of-grid / absent cells
    :param dense: If True, use the rectangular fast path (gather, no searchsorted)
    :param key_to_state: Optional dense lookup table mapping every grid linear key -> state ID
        (or missing_state for absent cells), shape [prod(num_per_dim)]. When provided, the cell's
        state ID is a single O(1) gather key_to_state[key], avoiding the per-call O(log S)
        searchsorted of the sparse path. Correct for sparse partitions too (gaps hold missing_state).
    :param row_lookup: Optional two-level lookup (see _build_row_lookup) for sparse partitions whose grid
        is too large for key_to_state: one row lookup per combination of leading coordinates instead of
        a searchsorted per cell.
    :return: State IDs of the spanned cells (shape [M]); duplicates from padding are harmless
             because the DP takes a min over them.
    """
    if row_lookup is not None:
        return _box_to_ids_rows(idx_lb, idx_ub, max_span, wrap, num_per_dim, row_lookup, missing_state)

    # Build the per-dimension linear contribution vectors and out-of-bounds masks, then combine
    # them by an outer sum. key starts as the dim-0 vector and grows one axis per added dimension.
    key = None
    oob = None
    for d in range(len(max_span)):
        # Indices along this dimension, padded to max_span[d] by repeating idx_ub[d].
        col = jnp.minimum(jnp.arange(max_span[d]) + idx_lb[d], idx_ub[d])      # (max_span[d],)
        col_oob = (col < 0) | (col >= num_per_dim[d])
        # Wrapped dims fold modulo; non-wrapped out-of-range indices are marked -1.
        resolved = jnp.where(wrap[d], col % num_per_dim[d], jnp.where(col_oob, -1, col))
        contrib = jnp.clip(resolved, 0, num_per_dim[d] - 1).astype(strides.dtype) * strides[d]
        col_oob_nw = (~wrap[d]) & (resolved < 0)                              # non-wrapped OOB
        if d == 0:
            key, oob = contrib, col_oob_nw
        else:
            key = key[..., None] + contrib                                    # outer sum
            oob = oob[..., None] | col_oob_nw
    key = key.reshape(-1)                                                     # (M,)
    oob = oob.reshape(-1)                                                     # (M,)

    if dense:
        # Dense rectangular grid: sorted keys are exactly 0..S-1, so region_linear_state[key] is the
        # cell's state ID (no binary search needed). clip keeps the gather in-bounds; oob -> missing.
        ids = region_linear_state[jnp.clip(key, 0, region_linear_state.shape[0] - 1)]
        return jnp.where(oob, missing_state, ids)

    if key_to_state is not None:
        # Full grid lookup table: one O(1) gather instead of searchsorted. key is in [0, table-1]
        # for every in-grid cell (wrapped dims fold modulo, non-wrapped OOB are masked by `oob`),
        # so clip is only a safety bound. Absent cells already hold missing_state in the table.
        ids = key_to_state[jnp.clip(key, 0, key_to_state.shape[0] - 1)]
        return jnp.where(oob, missing_state, ids)

    # Sparse grid: binary-search the sorted keys and verify an exact match.
    pos = jnp.searchsorted(region_linear_idx, key, side='left')
    pos_clip = jnp.minimum(pos, region_linear_idx.shape[0] - 1)
    valid = (pos < region_linear_idx.shape[0]) & (region_linear_idx[pos_clip] == key)
    return jnp.where(valid & ~oob, region_linear_state[pos_clip], missing_state)


def make_box_to_ids(max_span, wrap, partition, key_lut_max_cells=200_000_000):
    """Bind the static partition/grid data, returning a single-box -> IDs function (shape [D]->[M]).

    Callers vmap the result over noise cells / actions / states as needed.

    Unless the partition uses the dense fast path (`partition.rectangular`), a full grid lookup
    table (linear key -> state ID) is built once and bound, so each per-sweep cell lookup is an
    O(1) gather instead of an O(log S) searchsorted. The table has prod(number_per_dim) int32
    entries (the grid bounding box; independent of A / noise cells / max_span) and is only built
    when that stays under `key_lut_max_cells`. Larger grids use the two-level row lookup (tens of MB),
    and searchsorted only if that cannot be built.
    """
    num_per_dim = np.asarray(partition.number_per_dim)
    dense = bool(partition.rectangular)

    key_to_state = None
    row_lookup = None
    if not dense:
        total_cells = int(np.prod(num_per_dim))
        if total_cells <= key_lut_max_cells:
            # lut[linear_key] = state ID for present cells, missing_state elsewhere.
            lut = np.full(total_cells, int(partition.missing_state), dtype=np.int32)
            lut[np.asarray(partition.region_linear_idx)] = np.asarray(partition.region_linear_state)
            key_to_state = jax.device_put(lut)
        else:
            row_lookup = _build_row_lookup(num_per_dim, partition.region_linear_strides, partition.region_linear_idx,
                                           partition.region_linear_state, partition.missing_state)

    return partial(
        box_to_ids_single,
        max_span=tuple(int(x) for x in max_span),
        wrap=jnp.asarray(wrap),
        num_per_dim=jnp.asarray(num_per_dim),
        strides=jax.device_put(partition.region_linear_strides),
        region_linear_idx=jax.device_put(partition.region_linear_idx),
        region_linear_state=jax.device_put(partition.region_linear_state),
        missing_state=partition.missing_state,
        dense=dense,
        key_to_state=key_to_state,
        row_lookup=row_lookup,
    )
