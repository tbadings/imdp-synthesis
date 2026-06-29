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
a state ID by a direct gather into region_linear_state (no binary search); a sparse partition falls
back to searchsorted.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def box_to_ids_single(idx_lb, idx_ub, max_span, wrap, num_per_dim, strides,
                      region_linear_idx, region_linear_state, missing_state, dense,
                      key_to_state=None):
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
    :return: State IDs of the spanned cells (shape [M]); duplicates from padding are harmless
             because the DP takes a min over them.
    """
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
    when that stays under `key_lut_max_cells`; otherwise we fall back to searchsorted.
    """
    num_per_dim = np.asarray(partition.number_per_dim)
    dense = bool(partition.rectangular)

    key_to_state = None
    if not dense:
        total_cells = int(np.prod(num_per_dim))
        if total_cells <= key_lut_max_cells:
            # lut[linear_key] = state ID for present cells, missing_state elsewhere.
            lut = np.full(total_cells, int(partition.missing_state), dtype=np.int32)
            lut[np.asarray(partition.region_linear_idx)] = np.asarray(partition.region_linear_state)
            key_to_state = jax.device_put(lut)

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
    )
