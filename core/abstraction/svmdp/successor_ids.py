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


def box_to_ids_single(idx_lb, idx_ub, max_span, wrap, num_per_dim, strides,
                      region_linear_idx, region_linear_state, missing_state, dense):
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

    # Sparse grid: binary-search the sorted keys and verify an exact match.
    pos = jnp.searchsorted(region_linear_idx, key, side='left')
    pos_clip = jnp.minimum(pos, region_linear_idx.shape[0] - 1)
    valid = (pos < region_linear_idx.shape[0]) & (region_linear_idx[pos_clip] == key)
    return jnp.where(valid & ~oob, region_linear_state[pos_clip], missing_state)


def make_box_to_ids(max_span, wrap, partition):
    """Bind the static partition/grid data, returning a single-box -> IDs function (shape [D]->[M]).

    Callers vmap the result over noise cells / actions / states as needed.
    """
    return partial(
        box_to_ids_single,
        max_span=tuple(int(x) for x in max_span),
        wrap=jnp.asarray(wrap),
        num_per_dim=jnp.asarray(partition.number_per_dim),
        strides=jax.device_put(partition.region_linear_strides),
        region_linear_idx=jax.device_put(partition.region_linear_idx),
        region_linear_state=jax.device_put(partition.region_linear_state),
        missing_state=partition.missing_state,
        dense=bool(partition.rectangular),
    )
