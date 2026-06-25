import logging
import time
import argparse
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from core.abstraction.svmdp.svmdp import SVMDP

logger = logging.getLogger(__name__)


def RVI_SVMDP(
    args: argparse.Namespace,
    svmdp: SVMDP,
    s0: Optional[int] = None,
    max_iterations: int = 1000,
    epsilon: float = 1e-6,
    RND_SWEEPS: bool = False,
    BATCH_SIZE: int = 2000,
    policy_iteration: bool = False,
    return_Q_values: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value iteration for set-valued MDPs.

    Bellman operator (adversary minimises, controller maximises):

        V(s) = max_{a in A(s)} [  noise_remainder * V(absorbing)
                                 + sum_c  noise_probs[c]
                                         * min_{s' in Succ(s,a,c)} V(s')  ]

    :param args:           Argument namespace (needs floatprecision, rvi_device).
    :param svmdp:          SVMDP instance.
    :param s0:             Initial state index (for progress reporting only).
    :param max_iterations: Maximum number of value-iteration sweeps.
    :param epsilon:        Convergence threshold on max |V_new - V_old|.
    :param RND_SWEEPS:     Unused; kept for interface parity with RVI_JAX.
    :param BATCH_SIZE:     Unused; kept for interface parity with RVI_JAX.
    :param policy_iteration: Unused; kept for interface parity with RVI_JAX.
    :param return_Q_values:  Unused; kept for interface parity with RVI_JAX.
    :return: (V, policy_labels) — V[s] is the value, policy_labels[s] is the
             global action ID chosen for state s, or -1 if none.
    """

    start_time = time.time()

    S         = len(svmdp.states)
    A         = svmdp.actions.num_actions
    C         = svmdp.num_noise_cells
    absorbing = svmdp.absorbing_state
    nr_states = svmdp.nr_states

    noise_probs     = np.asarray(svmdp.noise_probs, dtype=args.floatprecision)   # (C,)
    noise_remainder = float(svmdp.noise_remainder)

    # ------------------------------------------------------------------
    # Pre-build CSR successor arrays (once, before the iteration loop).
    #
    # succ_flat[a][c]: 1-D int32 array – partition state IDs for all S states
    #                  concatenated in state-index order.
    # seg[a][c]:       1-D int32 array – segment IDs (state index of each entry).
    #
    # Using pre-stored noise_frs_counts avoids an O(S) Python len() loop
    # per (a,c) pair.
    # ------------------------------------------------------------------
    logger.info('  Pre-building CSR successor arrays (A=%d, C=%d)…', A, C)

    linear_idx   = svmdp._linear_idx    # sorted flat partition keys
    linear_state = svmdp._linear_state  # state IDs aligned to sorted keys

    succ_flat: list = [[None] * C for _ in range(A)]
    seg_arr:   list = [[None] * C for _ in range(A)]

    for a in tqdm(range(A), desc='  Building CSR'):
        for c in range(C):
            counts  = np.asarray(svmdp.actions.noise_frs_counts[:, a, c], dtype=np.int64)
            total   = int(counts.sum())

            if total == 0:
                succ_flat[a][c] = np.empty(0, dtype=np.int32)
                seg_arr[a][c]   = np.empty(0, dtype=np.int32)
                continue

            ids_obj  = svmdp.actions.noise_frs_cell_ids[:, a, c]   # (S,) object array
            all_flat = np.concatenate(ids_obj)                       # (total,) flat keys

            # Map flat linear keys → partition state IDs via binary search.
            pos   = np.searchsorted(linear_idx, all_flat)
            valid = (pos < len(linear_idx)) & (linear_idx[pos] == all_flat)
            state_ids = np.where(valid, linear_state[pos], absorbing).astype(np.int32)

            seg_ids = np.repeat(np.arange(S, dtype=np.int32), counts)

            succ_flat[a][c] = state_ids
            seg_arr[a][c]   = seg_ids

    # Pre-transfer to device — avoids repeated H→D copies inside the hot loop.
    logger.info('  Transferring CSR arrays to device…')
    succ_flat_dev = [
        [jax.device_put(jnp.asarray(succ_flat[a][c], dtype=jnp.int32), args.rvi_device)
         for c in range(C)]
        for a in range(A)
    ]
    seg_dev = [
        [jax.device_put(jnp.asarray(seg_arr[a][c], dtype=jnp.int32), args.rvi_device)
         for c in range(C)]
        for a in range(A)
    ]

    # ------------------------------------------------------------------
    # Action mask: True where action a is enabled for state s.
    # ------------------------------------------------------------------
    action_mask = np.zeros((S, A), dtype=bool)
    for s in svmdp.states:
        if s in svmdp.A_id:
            action_mask[s, svmdp.A_id[s]] = True

    has_actions       = action_mask.any(axis=1)
    update_mask       = ~svmdp.goal_regions & ~svmdp.critical_regions & has_actions
    states_to_update  = svmdp.states[update_mask]
    states_not_update = svmdp.states[~update_mask]

    # Masked Q: set disabled-action slots to -inf so argmax ignores them.
    Q_neg_inf_mask = ~action_mask   # (S, A)

    # ------------------------------------------------------------------
    # Initialise value function and policy.
    # ------------------------------------------------------------------
    V = np.zeros(nr_states, dtype=args.floatprecision)
    if len(svmdp.goal_regions) > 0:
        V[:-1][svmdp.goal_regions] = 1.0   # [:-1] excludes absorbing state

    policy = np.full(nr_states, -1, dtype=np.int32)
    policy[states_not_update] = -1

    logger.info(
        '  SVMDP ready (%.3fs) — states: %d  updatable: %d  actions: %d  noise cells: %d',
        time.time() - start_time, S, len(states_to_update), A, C,
    )

    pbar = tqdm(desc='Iteration', total=None, unit='it', dynamic_ncols=True, leave=True)

    # ------------------------------------------------------------------
    # Value iteration.
    # ------------------------------------------------------------------
    for iteration in range(max_iterations):
        pbar.update(1)
        if s0 is not None:
            pbar.set_postfix({
                f'V[{s0}]': f'{V[s0]:.6f}',
                'V_avg':    f'{np.mean(V[states_to_update]):.6f}',
            })

        V_old = V.copy()

        # Upload current V once per iteration.
        V_dev       = jax.device_put(jnp.asarray(V, dtype=args.floatprecision), args.rvi_device)
        v_absorbing = float(V[absorbing])

        # Initialise Q with the noise-remainder contribution (constant across noise cells).
        Q = np.full((S, A), noise_remainder * v_absorbing, dtype=args.floatprecision)

        # Accumulate per-noise-cell contributions.
        for a in range(A):
            for c in range(C):
                sf = succ_flat_dev[a][c]
                sg = seg_dev[a][c]

                if sf.shape[0] == 0:
                    # All successors are absorbing → contribution already in Q init.
                    continue

                v_succ   = V_dev[sf]
                min_vals = jax.ops.segment_min(
                    v_succ, sg,
                    num_segments=S, indices_are_sorted=False,
                )
                # Empty segments → +inf; replace with V(absorbing) = 0 (worst case).
                min_vals = jnp.where(jnp.isinf(min_vals), v_absorbing, min_vals)

                Q[:, a] += noise_probs[c] * np.asarray(
                    jax.device_get(min_vals), dtype=args.floatprecision
                )

        # Policy improvement: argmax over enabled actions.
        Q[Q_neg_inf_mask] = -np.inf
        best_action = np.argmax(Q[states_to_update], axis=1)   # global action IDs
        V[states_to_update]      = Q[states_to_update, best_action]
        policy[states_to_update] = best_action

        if np.max(np.abs(V - V_old)) < epsilon:
            pbar.write(f'Converged after {iteration + 1} iterations')
            break

    pbar.close()

    # policy[s] already holds the global action ID (argmax over all A columns).
    policy_labels = policy.copy()

    return V, policy_labels
