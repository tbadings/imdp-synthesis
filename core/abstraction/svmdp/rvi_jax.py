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
    gamma: float = 0.999,
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
    absorbing = svmdp.absorbing_state
    nr_states = svmdp.nr_states

    noise_remainder = float(svmdp.noise_remainder)

    # ------------------------------------------------------------------
    # Build the merged-group successor CSR (once, before the iteration loop).
    #
    # forward_reachability already merged the (state, action, noise) entries
    # with identical successor sets into G groups, each carrying its owning
    # (state, action), a merged probability, and a successor cell-id set. Here
    # we only map the flat partition cell ids to partition state ids and lay
    # everything out for a single segment_min / segment_sum per Bellman backup.
    # ------------------------------------------------------------------
    logger.info('  Building merged-group successor CSR…')

    linear_idx   = svmdp._linear_idx    # sorted flat partition keys
    linear_state = svmdp._linear_state  # state IDs aligned to sorted keys

    fwd = svmdp.actions
    G            = int(fwd.num_groups)
    group_state  = np.asarray(fwd.group_state,  dtype=np.int64)   # (G,)
    group_action = np.asarray(fwd.group_action, dtype=np.int64)   # (G,)
    group_prob   = np.asarray(fwd.group_prob,   dtype=args.floatprecision)  # (G,)
    group_seg    = np.asarray(fwd.group_succ_seg, dtype=np.int32)           # (T,)
    group_flat   = np.asarray(fwd.group_succ_flat)                          # (T,) flat cell ids

    # Map flat linear keys → partition state IDs via binary search (once).
    if group_flat.size > 0:
        pos   = np.searchsorted(linear_idx, group_flat)
        pos_c = np.minimum(pos, len(linear_idx) - 1)
        valid = (pos < len(linear_idx)) & (linear_idx[pos_c] == group_flat)
        succ_state = np.where(valid, linear_state[pos_c], absorbing).astype(np.int32)
    else:
        succ_state = np.empty(0, dtype=np.int32)

    # Flattened (state, action) index of each group, for scatter-add into Q.
    group_sa = (group_state * A + group_action).astype(np.int32)   # (G,)

    # Pre-transfer to device.
    succ_state_dev = jax.device_put(jnp.asarray(succ_state, dtype=jnp.int32), args.rvi_device)
    group_seg_dev  = jax.device_put(jnp.asarray(group_seg,  dtype=jnp.int32), args.rvi_device)
    group_prob_dev = jax.device_put(jnp.asarray(group_prob, dtype=args.floatprecision), args.rvi_device)
    group_sa_dev   = jax.device_put(jnp.asarray(group_sa,   dtype=jnp.int32), args.rvi_device)

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

    logger.info(
        '  SVMDP ready (%.3fs) — states: %d  updatable: %d  actions: %d  groups: %d',
        time.time() - start_time, S, len(states_to_update), A, G,
    )

    # ------------------------------------------------------------------
    # Bellman Q-operator: Q[s, a] = noise_remainder * V(absorbing)
    #                               + sum_c noise_probs[c] * min_{Succ(s,a,c)} V.
    # With merged groups this is a single segment_min (worst successor per group)
    # followed by a segment_sum scattering probability-weighted group values into
    # the (S, A) grid. Returns the *undiscounted* one-step expectation.
    # ------------------------------------------------------------------
    def compute_Q(Vvec: np.ndarray) -> np.ndarray:
        V_dev       = jax.device_put(jnp.asarray(Vvec, dtype=args.floatprecision), args.rvi_device)
        v_absorbing = float(Vvec[absorbing])

        v_succ    = V_dev[succ_state_dev]                                   # (T,)
        group_min = jax.ops.segment_min(v_succ, group_seg_dev, num_segments=G)
        # Empty groups (all-absorbing) → +inf identity; replace with V(absorbing).
        group_min = jnp.where(jnp.isinf(group_min), v_absorbing, group_min)  # (G,)

        contrib = group_prob_dev * group_min                                # (G,)
        Q_added = jax.ops.segment_sum(contrib, group_sa_dev, num_segments=S * A)  # (S*A,)

        Q = noise_remainder * v_absorbing + np.asarray(
            jax.device_get(Q_added), dtype=args.floatprecision
        )
        return Q.reshape(S, A)

    # ------------------------------------------------------------------
    # Phase 1 — policy synthesis with a discount gamma < 1.
    #
    # The undiscounted reach probability saturates to 1 across the whole
    # basin of attraction (no probability leaks out when the noise has
    # bounded support), so a greedy argmax over the saturated value has no
    # gradient to follow and returns a degenerate policy. Discounting makes
    # states closer to the goal strictly more valuable, which yields a
    # goal-directed policy. The discounted values themselves are only used
    # to rank actions; the reported satisfaction probability comes from
    # phase 2.
    # ------------------------------------------------------------------
    Vd = np.zeros(nr_states, dtype=args.floatprecision)
    if len(svmdp.goal_regions) > 0:
        Vd[:-1][svmdp.goal_regions] = 1.0   # [:-1] excludes absorbing state

    policy = np.full(nr_states, -1, dtype=np.int32)

    pbar = tqdm(desc='Synthesis (γ=%.4g)' % gamma, total=None, unit='it', dynamic_ncols=True, leave=True)
    for iteration in range(max_iterations):
        pbar.update(1)
        if s0 is not None:
            pbar.set_postfix({f'Vd[{s0}]': f'{Vd[s0]:.6f}'})

        Vd_old = Vd.copy()
        Q = gamma * compute_Q(Vd)           # discounted one-step backup
        Q[Q_neg_inf_mask] = -np.inf         # mask disabled actions
        best_action = np.argmax(Q[states_to_update], axis=1)   # global action IDs
        Vd[states_to_update]     = Q[states_to_update, best_action]
        policy[states_to_update] = best_action

        if np.max(np.abs(Vd - Vd_old)) < epsilon:
            pbar.write(f'Synthesis converged after {iteration + 1} iterations')
            break
    pbar.close()

    # ------------------------------------------------------------------
    # Phase 2 — evaluate the synthesised policy undiscounted (gamma = 1)
    # to obtain its true reach (satisfaction) probability.
    # ------------------------------------------------------------------
    V = np.zeros(nr_states, dtype=args.floatprecision)
    if len(svmdp.goal_regions) > 0:
        V[:-1][svmdp.goal_regions] = 1.0

    rows = states_to_update
    cols = policy[states_to_update]
    pbar = tqdm(desc='Evaluation', total=None, unit='it', dynamic_ncols=True, leave=True)
    for iteration in range(max_iterations):
        pbar.update(1)
        if s0 is not None:
            pbar.set_postfix({f'V[{s0}]': f'{V[s0]:.6f}'})

        V_old = V.copy()
        Q = compute_Q(V)                    # undiscounted backup
        V[rows] = Q[rows, cols]             # follow the fixed policy action

        if np.max(np.abs(V - V_old)) < epsilon:
            pbar.write(f'Evaluation converged after {iteration + 1} iterations')
            break
    pbar.close()

    # policy[s] already holds the global action ID (argmax over all A columns).
    policy_labels = policy.copy()

    return V, policy_labels
