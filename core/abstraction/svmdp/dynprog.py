import logging
import numpy as np
from tqdm import tqdm
from copy import copy, deepcopy
import jax
import jax.numpy as jnp
import time
import argparse
from typing import Optional, Tuple
from jaxtyping import Array, UInt8, Bool, Float32, PyTree

from core.abstraction.svmdp.svmdp import SVMDP
from core.utils import jit_compile_count, create_batches

logger = logging.getLogger(__name__)


def SVMDP_DP(
    args: argparse.Namespace, 
    svmdp: SVMDP, 
    s0: Optional[int] = None, 
    max_iterations: int = 1000, 
    epsilon: float = 1e-6, 
    RND_SWEEPS: bool = False, 
    BATCH_SIZE: int = 2000, 
    policy_iteration: bool = False,
    return_Q_values: bool = False,
    prune_states: bool = True,
) -> Tuple[Float32[Array, "nr_states"], UInt8[Array, "nr_states"]]:

    """
    Robust value iteration for set-valued MDPs.

    :param args: Argument namespace
    :param imdp: Instance of IMDP class
    :param s0: Initial state for tracking
    :param max_iterations: Maximum number of iterations
    :param epsilon: Convergence threshold
    :param RND_SWEEPS: Whether to use random state sweeps
    :param BATCH_SIZE: Batch size for state updates
    :param policy_iteration: Whether to use policy iteration instead of value iteration
    :param return_Q_values: Whether to return Q-values for all state-action pairs
    :return: Tuple of (values, policy_labels) where policy_labels[s] is the global action ID chosen for state s, or -1
    """

    start_time = time.time()

    phase1_initial_it = 10
    phase1_increment_it = 10
    phase1_max_it = 100
    fix_policy_above_value = 2 # >1 means this feature is disabled

    #####

    def compute_lower_val(
        probs: Float32[Array, "nr_noise_cells"], 
        successor_values: Float32[Array, "nr_noise_cells nr_successors"],
    ) -> Float32:

        """
        Compute the robust value for a given action based on the probability intervals and successor values.

        :param probs: Transition probability for each noise cell
        :param successor_values: Values of the successor states for each noise cell
        :return: The robust value for the action
        """
        
        # Compute min (worst-case) value for every noise cell
        min_values = jnp.min(successor_values, axis=1)

        # Multiply these values with the respective probabilities
        lower_val = probs @ min_values
        
        # Clip the values to be within [0, 1], since they are probabilities
        return jnp.clip(lower_val, 0.0, 1.0)

    vmap_compute_lower_val = jax.jit(jax.vmap(compute_lower_val, in_axes=(0, 0), out_axes=0))

    # On-the-fly recomposition of successor state IDs from the compact forward-reachable boxes.
    # box_to_ids maps one box (idx_lb[D], idx_ub[D]) -> successor IDs [M = prod(max_span)] without
    # materialising the [S, A, nc, M] array. We vmap it over noise cells and actions as needed.
    box_to_ids = svmdp.box_to_ids
    ids_over_noise = jax.vmap(box_to_ids, in_axes=(0, 0), out_axes=0)              # [nc,D] -> [nc,M]
    ids_over_actions = jax.vmap(ids_over_noise, in_axes=(0, 0), out_axes=0)        # [A,nc,D] -> [A,nc,M]
    # Batched composer used by the state-pruning fix-point: [batch,A,nc,D] -> [batch,A,nc,M].
    batch_ids_over_actions = jax.jit(jax.vmap(ids_over_actions, in_axes=(0, 0), out_axes=0))

    def state_policy_improvement(
        idx_lb_slice: UInt8[Array, "nr_actions nr_noise_cells state_dim"],
        idx_ub_slice: UInt8[Array, "nr_actions nr_noise_cells state_dim"],
        prob_slice: Float32[Array, "nr_actions nr_noise_cells"],
        V: Float32[Array, "nr_states"],
    ) -> Tuple[Float32, UInt8]:

        """
        Perform policy improvement for a given state by computing the robust values for all actions.

        :param idx_lb_slice: Lower grid-index bounds of the forward-reachable boxes for all actions
        :param idx_ub_slice: Upper grid-index bounds of the forward-reachable boxes for all actions
        :param prob_slice: Slice of transition probabilities for all actions
        :param V: Current value function
        :return: Tuple of (maximum robust value, index of the action with maximum robust value)
        """

        # Recompose successor IDs from the boxes, then retrieve their values (incl. absorbing state)
        successors_slice = ids_over_actions(idx_lb_slice, idx_ub_slice)
        successor_values = V[successors_slice]

        # Compute lower value for all actions in parallel using JAX vectorization
        lower_vals = vmap_compute_lower_val(prob_slice, successor_values)

        return jnp.max(lower_vals), jnp.argmax(lower_vals)

    vmap_state_policy_improvement = jax.jit(jax.vmap(state_policy_improvement, in_axes=(0, 0, 0, None), out_axes=(0, 0)))

    def state_policy_evaluation(
        idx_lb_slice: UInt8[Array, "nr_noise_cells state_dim"],
        idx_ub_slice: UInt8[Array, "nr_noise_cells state_dim"],
        prob_slice: Float32[Array, "nr_noise_cells"],
        V: Float32[Array, "nr_states"],
    ) -> Float32:

        """
        Perform policy evaluation for a given state by computing the robust value for the action specified by the current policy.

        :param idx_lb_slice: Lower grid-index bounds of the forward-reachable boxes for the chosen action
        :param idx_ub_slice: Upper grid-index bounds of the forward-reachable boxes for the chosen action
        :param prob_slice: Slice of transition probabilities for the chosen action
        :param V: Current value function
        :return: The robust value for the action specified by the current policy
        """

        # Recompose successor IDs from the boxes, then retrieve their values (incl. absorbing state)
        successors_slice = ids_over_noise(idx_lb_slice, idx_ub_slice)
        successor_values = V[successors_slice]

        # Compute lower value for all actions in parallel using JAX vectorization
        lower_vals = compute_lower_val(prob_slice, successor_values)

        return lower_vals

    vmap_state_policy_evaluation = jax.jit(jax.vmap(state_policy_evaluation, in_axes=(0, 0, 0, None), out_axes=(0)))

    #####

    # Count the total number of actions
    total_actions = np.array([len(svmdp.A_id[s]) for s in svmdp.states if s in svmdp.A_id])
    max_actions = np.max(total_actions) if len(total_actions) > 0 else 0

    if policy_iteration:
        logger.info('=== Run robust policy iteration ===')
    else:
        logger.info('=== Run robust value iteration ===')

    logger.info('- Number of states: %d', len(svmdp.states))
    logger.info('- Total number of choices: %d (total number of state-action pairs)', np.sum(total_actions))
    logger.info('- Max number of actions per state: %d', max_actions)
    logger.info('- Batch size for dynamic programming over states: %d', BATCH_SIZE)

    #####

    logger.info('- Set states to update...')
    states_with_enabled_actions = np.array([True if s in svmdp.A_id and len(svmdp.A_id[s]) > 0 else False for s in svmdp.states])

    absorbing_mask = svmdp.critical_regions | (svmdp.states == svmdp.absorbing_state) | ~states_with_enabled_actions
    goal_mask = svmdp.goal_regions
    skip_mask = absorbing_mask | goal_mask

    states_to_update = svmdp.states[~skip_mask]
    states_not_to_update = svmdp.states[skip_mask]

    print(f'  - Active states in initial mask: {len(svmdp.states[~skip_mask])}')

    def fn1(successor, mask):
        ''' Check whether a successor is contained in skipped (successor: int)'''
        return mask[successor]
    
    # vmap over multiple successors (check for all successors if they are in the skip mask)
    vmap_fn1 = jax.vmap(fn1, in_axes=(0, None), out_axes=(0))

    def fn2(successors, probability, mask):
        ''' Check whether any successor of an action is contained in skipped (successors: int array, probability: real)'''

        # Skip action if for every noise cells, the successor set contains a skipped state or the probability is zero.
        return jnp.all(jnp.any(vmap_fn1(successors, mask), axis=1) + (probability == 0))
    
    # vmap over multiple actions
    vmap_fn2 = jax.vmap(fn2, in_axes=(0, 0, None), out_axes=(0))

    @jax.jit
    def fn3(successors, probabilities, mask):
        ''' Check whether every prob>0 cell has at least one successor in the skip_mask (successors: int 2D array, probability: real array) '''
        return jnp.all(vmap_fn2(successors, probabilities, mask))
    
    # vmap over multiple states
    vmap_fn3 = jax.jit(jax.vmap(fn3, in_axes=(0, 0, None), out_axes=(0)))

    if prune_states:
        done = False
        while not done:
            done = True

            starts, ends = create_batches(len(states_to_update), BATCH_SIZE)
            pbar = tqdm(zip(starts, ends), desc='Prune states', total=len(starts))
            for batch_start, batch_end in pbar:
                states = states_to_update[batch_start:batch_end]
                S_id_batch = batch_ids_over_actions(jnp.asarray(svmdp.S_idx_lb[states]), jnp.asarray(svmdp.S_idx_ub[states]))
                skip = vmap_fn3(S_id_batch, svmdp.P_full[states], jnp.concatenate((absorbing_mask, jnp.array([True])))) # Add one 'true' for the out-of-bounds state
                skip_mask[states] = skip
                absorbing_mask[states] = skip
                if any(skip):
                    # print(f'- Skip {np.sum(skip)} states')
                    done = False

            if not done:
                states_to_update = svmdp.states[~skip_mask]
                states_not_to_update = svmdp.states[skip_mask]

        print(f'  - States after pruning: {len(states_to_update)}')

    # Initialize value function and policy
    V = np.zeros(svmdp.nr_states, dtype=args.floatprecision)
    if len(svmdp.goal_regions) > 0:
        V[:-1][svmdp.goal_regions] = 1.0 # [:-1] to exclude the absorbing state

    policy = np.zeros(svmdp.nr_states, dtype=np.int32)
    policy[states_not_to_update] = -1  # Mark states that we do not update with a special action index (e.g., -1)
    
    logger.info(f'- SVMDP defined (took {time.time() - start_time:.3f}s)')
    start_time = time.time()

    # The policy-improvement inputs (lower/upper FRS index boxes and probabilities) span all actions per
    # state and depend only on the (fixed) FRS data, not on V or the policy, so they are identical on every
    # outer iteration. Gather them into compact arrays once here instead of re-slicing the multi-GB
    # [S, A, C, D] source arrays every iteration. Both the improvement step (all actions) and the evaluation
    # step (the policy-selected action) read from these.
    #
    # states_to_update == svmdp.states[~skip_mask] is ascending, so this single gather is a monotonic,
    # locality-friendly pass over the source (which is contiguous after the FRS truncation now uses
    # np.ascontiguousarray). We compact *first* and then derive batches, rather than fancy-indexing the
    # multi-GB source once per batch with scattered (permuted) indices.
    lb_c = svmdp.S_idx_lb[states_to_update]
    ub_c = svmdp.S_idx_ub[states_to_update]
    p_c = svmdp.P_full[states_to_update]
    # Delete the originals to save memory; everything below works off the compact arrays. (Not used after the DP returns.)
    svmdp.S_idx_lb = svmdp.S_idx_ub = svmdp.P_full = None

    logger.info(f'- Index arrays sliced (took {time.time() - start_time:.3f}s)')
    start_time = time.time()

    if RND_SWEEPS:
        # Randomise the sweep order by permuting the already-compacted (small, contiguous, in-RAM) rows
        # rather than re-scattering the source. states_to_update is permuted with the same permutation so
        # each batch's state IDs stay aligned with its FRS-input rows. The per-batch tuples below are then
        # plain contiguous slices (views) of the permuted compact arrays — no further copy.
        perm = np.random.permutation(len(states_to_update))
        states_to_update = states_to_update[perm]
        lb_c, ub_c, p_c = lb_c[perm], ub_c[perm], p_c[perm]
        starts = range(0, len(states_to_update), BATCH_SIZE)
        state_batches = [states_to_update[i:i + BATCH_SIZE] for i in starts]
        imp_batches = [(lb_c[i:i + BATCH_SIZE], ub_c[i:i + BATCH_SIZE], p_c[i:i + BATCH_SIZE]) for i in starts]
    else:
        state_batches = [states_to_update]
        imp_batches = [(lb_c, ub_c, p_c)]

    logger.info(f'- Number of batches: {len(state_batches)} (took {time.time() - start_time:.3f}s)')

    pbar = tqdm(desc='Iteration', total=None, unit='it', dynamic_ncols=True, leave=True)
    if not policy_iteration:
        # Value iteration
        for iteration in range(max_iterations):
            pbar.update(1)
            postfix_dict = {}
            if s0 is not None:
                postfix_dict[f'v[{s0}]'] = f'{V[s0]:.6f}'
                postfix_dict[f'v_avg'] = f'{np.mean(V[states_to_update]):.6f}'
            pbar.set_postfix(postfix_dict)
            
            V_old = V.copy()
                
            # Policy evaluation + improvement
            for state_batch, (lb_d, ub_d, p_d) in zip(state_batches, imp_batches):
                V_batch, policy_batch = vmap_state_policy_improvement(lb_d, ub_d, p_d, V)
                V_batch, policy_batch = jax.device_get((V_batch, policy_batch))
                V[state_batch] = np.asarray(V_batch, dtype=args.floatprecision)
                policy[state_batch] = np.asarray(policy_batch, dtype=np.int32)

            # Check convergence
            if np.max(np.abs(V - V_old)) < epsilon:
                pbar.write(f'Converged after {iteration + 1} iterations')
                break

    else:
        partial_convergence_reached = False
        sat_policy = False
        delta = float('inf')

        # Policy iteration
        for iteration in range(max_iterations):

            pbar.update(1)

            t = time.time()

            # Policy evaluation
            i = 0
            # The policy is fixed throughout policy evaluation, so slice the policy-selected action out
            # of the precomputed imp_batches once here, rather than re-slicing on every inner sweep.
            # Combined fancy-indexing ([rows, sel]) selects one action per state directly, avoiding the
            # [B, A, nc, D] intermediate copy that plain [rows][..., sel] would materialise.
            eval_batches = []
            for bi, (lb_a, ub_a, p_a) in enumerate(imp_batches):
                sel = policy[state_batches[bi]]
                rows = np.arange(len(sel))
                eval_batches.append((lb_a[rows, sel], ub_a[rows, sel], p_a[rows, sel]))

            print(f'- Prepare policy evaluation (took {time.time() - t:.3f}s)')
            t = time.time()

            # Keep V resident on the device across the whole evaluation phase. Each batch update is a
            # functional scatter (Gauss-Seidel: later batches see earlier updates, as before), so we
            # only pull V back to the host once per sweep for the convergence/postfix checks instead
            # of blocking on a device_get after every one of the ~len(state_batches) batches.
            Vd = jnp.asarray(V)
            while True:

                postfix_dict = {}
                if s0 is not None:
                    postfix_dict[f'eval_it'] = i
                    postfix_dict[f'v[{s0}]'] = f'{V[s0]:.6f}'
                    postfix_dict[f'v_avg'] = f'{np.mean(V[states_to_update]):.6f}'
                    postfix_dict[f'max(v-v_old)'] = f'{delta:.6f}'

                    # Check if policy is above the preset threshold quality
                    if V[s0] > fix_policy_above_value:
                        # Policy is already good enough, so skip policy improvement and only keep evaluating it until convergence
                        sat_policy = True
                    else:
                        sat_policy = False
                pbar.set_postfix(postfix_dict)

                # print(f'- Policy evaluation iteration {i + 1}...')
                V_old = V

                # Policy evaluation only (V stays on the device; scatter each batch's result back in)
                for state_batch, (ev_lb, ev_ub, ev_p) in zip(state_batches, eval_batches):
                    V_eval = vmap_state_policy_evaluation(ev_lb, ev_ub, ev_p, Vd)
                    Vd = Vd.at[state_batch].set(V_eval)
                V = np.asarray(jax.device_get(Vd), dtype=args.floatprecision)

                delta = np.max(np.abs(V - V_old))
                if delta < epsilon or (
                    not partial_convergence_reached
                    and i > min(phase1_initial_it + iteration * phase1_increment_it, phase1_max_it)
                ):
                    break

                i += 1

            jax.block_until_ready(V)
            print(f'- Policy evaluation took {time.time() - t:.3f}s')
            t = time.time()

            # Policy evaluation + improvement
            V_before_improvement = V.copy()

            print(f'- Prepare policy improvement (took {time.time() - t:.3f}s)')
            t = time.time()

            if not sat_policy:
                # Same device-resident, Gauss-Seidel pattern as evaluation: scatter each batch's
                # improved values back into the on-device V and only sync once at the end (for V and
                # for the whole policy update) rather than blocking after every batch.
                Vd = jnp.asarray(V)
                policy_refs = []
                for state_batch, (lb_d, ub_d, p_d) in zip(state_batches, imp_batches):
                    V_batch, policy_batch = vmap_state_policy_improvement(lb_d, ub_d, p_d, Vd)
                    Vd = Vd.at[state_batch].set(V_batch)
                    policy_refs.append(policy_batch)
                V = np.asarray(jax.device_get(Vd), dtype=args.floatprecision)
                for state_batch, policy_batch in zip(state_batches, jax.device_get(policy_refs)):
                    policy[state_batch] = np.asarray(policy_batch, dtype=np.int32)

            jax.block_until_ready(V)
            print(f'- Policy improvement took {time.time() - t:.3f}s')
            t = time.time()

            # Check convergence: improvement step is monotone, so max gain suffices
            # TODO: Better validate the convergence criterion based on max gain (rather than checking if the policy is unchanged; which is less stable in case of multiple optimal policies)
            if np.max(V - V_before_improvement) < epsilon:
                if partial_convergence_reached:
                    pbar.write(f'Converged after {iteration + 1} iterations')
                    break
                else:
                    pbar.write(
                        f'Partial convergence after {iteration + 1} iterations. '
                        'Decrease epsilon to refine values...'
                    )
                    partial_convergence_reached = True

            print(f'- Check convergence took {time.time() - t:.3f}s')

    pbar.close()

    # Extract policy inputs from policy
    policy_labels = np.full_like(policy, fill_value=-1)
    for s in svmdp.states:
        policy_labels[s] = svmdp.A_id[s][int(policy[s])] if policy[s] != -1 and s in svmdp.A_id else -1

    return V, policy_labels
