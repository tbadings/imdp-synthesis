import logging
from functools import partial
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import time
import argparse
from typing import Optional, Tuple
from jaxtyping import Array, UInt8, Float32

from core.abstraction.svmdp.svmdp import SVMDP
from core.utils import create_batches

logger = logging.getLogger(__name__)


def _union_kernels(union_box_to_ids, union_span, noise_dims):
    """
    Policy improvement and evaluation kernels (batched, as in SVMDP_DP) that read V once over the union box of
    a state-action pair's noise cells. The noise cells' boxes coincide along the dimensions without noise, so V
    is first minimised over those (once, for all noise cells), and each noise cell's minimum then needs a mask
    over the noise dimensions only.

    :param union_box_to_ids: box_to_ids bound to max_span=union_span
    :param union_span: Static span of the union boxes (tuple of D Python ints)
    :param noise_dims: Dimensions in which the noise cells' boxes differ (tuple of Python ints)
    :return: Tuple (vmap_state_policy_improvement, vmap_state_policy_evaluation)
    """
    other_dims = tuple(d for d in range(len(union_span)) if d not in noise_dims)
    noise_cols = [jnp.arange(union_span[d]) for d in noise_dims]

    def union_lower_val(idx_lb, idx_ub, probs, V):
        '''Robust value of one state-action pair from its noise cells' union box (same value as compute_lower_val).'''
        active = probs > 0
        lb, ub = idx_lb.astype(jnp.int32), idx_ub.astype(jnp.int32)
        ulb = jnp.where(jnp.any(active), jnp.min(jnp.where(active[:, None], lb, jnp.iinfo(jnp.int32).max), axis=0), lb[0])
        uub = jnp.where(jnp.any(active), jnp.max(jnp.where(active[:, None], ub, jnp.iinfo(jnp.int32).min), axis=0), ub[0])
        values = V[union_box_to_ids(ulb, uub)].reshape(union_span)

        # Along the dimensions without noise, every noise cell's box spans exactly the union, so minimise there first
        reduced = jnp.min(values, axis=other_dims) if other_dims else values

        # inside[j, cell]: the cell (over the noise dimensions) lies in noise cell j's box. Outer product of
        # per-dimension masks; the union's cells are padded by repeating its upper index, as in box_to_ids.
        inside = jnp.ones((lb.shape[0],) + (1,) * len(noise_dims), dtype=bool)
        for k, d in enumerate(noise_dims):
            col = jnp.minimum(noise_cols[k] + ulb[d], uub[d])
            mask = (col[None, :] >= lb[:, d, None]) & (col[None, :] <= ub[:, d, None])
            shape = [lb.shape[0]] + [1] * len(noise_dims)
            shape[1 + k] = union_span[d]
            inside = inside & mask.reshape(shape)
        min_values = jnp.min(jnp.where(inside.reshape(lb.shape[0], -1), reduced.reshape(1, -1), jnp.inf), axis=1)

        # Noise cells with probability 0 (padding) contribute 0, as in compute_lower_val
        min_values = jnp.where(active, min_values, 0.0)
        return jnp.clip(probs @ min_values, 0.0, 1.0)

    def union_policy_improvement(idx_lb_slice, idx_ub_slice, prob_slice, V):
        lower_vals = jax.vmap(union_lower_val, in_axes=(0, 0, 0, None))(idx_lb_slice, idx_ub_slice, prob_slice, V)
        return jnp.max(lower_vals), jnp.argmax(lower_vals)

    return (jax.jit(jax.vmap(union_policy_improvement, in_axes=(0, 0, 0, None), out_axes=(0, 0))),
            jax.jit(jax.vmap(union_lower_val, in_axes=(0, 0, 0, None), out_axes=0)))


def SVMDP_DP(
    args: argparse.Namespace, 
    svmdp: SVMDP, 
    s0: Optional[int] = None, 
    max_iterations: int = 1000, 
    epsilon: float = 1e-6, 
    RND_SWEEPS: bool = False,
    sweep_priority: Optional[np.ndarray] = None,
    BATCH_SIZE: int = 2000,
    policy_iteration: bool = False,
    prune_states: bool = True,
    phase1_initial_it: int = 10,
    phase1_increment_it: int = 10,
    phase1_max_it: int = 31,
    max_eval_it: int = 31,
) -> Tuple[Float32[Array, "nr_states"], UInt8[Array, "nr_states"]]:

    """
    Robust value iteration for set-valued MDPs.

    :param args: Argument namespace
    :param svmdp: Instance of SVMDP class
    :param s0: Initial state for tracking
    :param max_iterations: Maximum number of iterations
    :param epsilon: Convergence threshold
    :param RND_SWEEPS: Whether to use random state sweeps
    :param sweep_priority: Optional per-state priority (e.g. steps-to-goal along the RL rollouts); states are
        swept in ascending priority, with ties in random order. Overrides the random order of RND_SWEEPS.
    :param BATCH_SIZE: Batch size for state updates
    :param policy_iteration: Whether to use policy iteration instead of value iteration
    :param phase1_initial_it: Base cap on inner policy-evaluation sweeps in the first outer iteration
    :param phase1_increment_it: Per-outer-iteration growth of the inner-sweep cap
    :param phase1_max_it: Hard ceiling on the inner-sweep cap (before full convergence)
    :param max_eval_it: Maximum evaluation iteration index (keeps eval_it <= max_eval_it, strictly < 32 by default)
    :return: Tuple of (values, policy_labels) where policy_labels[s] is the global action ID chosen for state s, or -1
    """

    start_time = time.time()

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

    # Count the total number of actions. A_id is a single shared list of action ids:
    # every state has the same actions enabled.
    num_actions = len(svmdp.A_id)
    total_actions = np.full(len(svmdp.states), num_actions)
    max_actions = num_actions

    if policy_iteration:
        logger.info('(Algorithm: robust policy iteration)')
    else:
        logger.info('(Algorithm: robust value iteration)')

    logger.info('- Number of states: %d', len(svmdp.states))
    logger.info('- Total number of choices: %d (total number of state-action pairs)', np.sum(total_actions))
    logger.info('- Max number of actions per state: %d', max_actions)
    logger.info('- Batch size for dynamic programming over states: %d', BATCH_SIZE)

    #####

    logger.info('- Set states to update...')
    states_with_enabled_actions = np.full(len(svmdp.states), num_actions > 0)

    absorbing_mask = svmdp.critical_regions | (svmdp.states == svmdp.absorbing_state) | ~states_with_enabled_actions
    goal_mask = svmdp.goal_regions
    skip_mask = absorbing_mask | goal_mask

    states_to_update = svmdp.states[~skip_mask]
    states_not_to_update = svmdp.states[skip_mask]

    logger.info(f'  (Active states in initial mask: {len(svmdp.states[~skip_mask])})')

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
        s_init_skipped_before_pruning = skip_mask[svmdp.s_init]
        # Track the fix-point generation at which each sweep prunes states (and specifically when
        # s_init is pruned). The generation of s_init is the depth of the dead-end funnel from x0:
        # generation 1 means all of s_init's own successors are already absorbing; a larger value
        # means the tube reaches k cells deep before dead-ending. Diagnostic logging only.
        prune_generation = 0
        s_init_prune_generation = -1
        done = False
        while not done:
            done = True
            prune_generation += 1
            pruned_this_gen = 0

            starts, ends = create_batches(len(states_to_update), BATCH_SIZE)
            pbar = tqdm(zip(starts, ends), desc='Prune states', total=len(starts))
            for batch_start, batch_end in pbar:
                states = states_to_update[batch_start:batch_end]
                S_id_batch = batch_ids_over_actions(jnp.asarray(svmdp.S_idx_lb[states]), jnp.asarray(svmdp.S_idx_ub[states]))
                skip = vmap_fn3(S_id_batch, svmdp.P_full[states], jnp.concatenate((absorbing_mask, jnp.array([True])))) # Add one 'true' for the out-of-bounds state
                skip_mask[states] = skip
                absorbing_mask[states] = skip
                pruned_this_gen += int(np.sum(np.asarray(skip)))
                if any(skip):
                    done = False

            if not done:
                states_to_update = svmdp.states[~skip_mask]
                states_not_to_update = svmdp.states[skip_mask]

            if pruned_this_gen > 0:
                if s_init_prune_generation == -1 and skip_mask[svmdp.s_init] and not s_init_skipped_before_pruning:
                    s_init_prune_generation = prune_generation
                logger.info(f'  Prune generation {prune_generation}: pruned {pruned_this_gen} states '
                            f'(remaining active {len(states_to_update)})'
                            + ('  *** s_init pruned here ***' if s_init_prune_generation == prune_generation else ''))
                
                if len(states_to_update) == 0 or pruned_this_gen / len(states_to_update) < 0.01:
                    break  # Stop pruning if less than 1% of the remaining states were pruned in this generation

        logger.info(f'  (States after pruning: {len(states_to_update)})')

        if skip_mask[svmdp.s_init] and not s_init_skipped_before_pruning:
            logger.warning(f'  Initial state ({svmdp.s_init}) was pruned (all actions directly lead to '
                           f'absorbing states) at prune generation {s_init_prune_generation} '
                           f'(dead-end funnel depth from x0)')

    # Initialize value function and policy
    V = np.zeros(svmdp.nr_states, dtype=args.floatprecision)
    if len(svmdp.goal_regions) > 0:
        V[:-1][svmdp.goal_regions] = 1.0 # [:-1] to exclude the absorbing state

    policy = np.zeros(svmdp.nr_states, dtype=np.int32)
    policy[states_not_to_update] = -1  # Mark states that we do not update with a special action index (e.g., -1)

    # Sweep order. It is set before the FRS inputs are gathered below, so they are copied only once, in this order.
    if RND_SWEEPS or sweep_priority is not None:
        perm = np.random.permutation(len(states_to_update))
        if sweep_priority is not None:
            # States nearest the goal first, so one Gauss-Seidel sweep carries value back along the RL
            # rollouts; the stable sort keeps the random order above among equal priorities.
            perm = perm[np.argsort(sweep_priority[states_to_update[perm]], kind='stable')]
        logger.info('- State sweep order: %s', 'by sweep priority' if sweep_priority is not None else 'random')
        states_to_update = states_to_update[perm]

    # The policy-improvement inputs (lower/upper FRS index boxes and probabilities) span all actions per
    # state and depend only on the (fixed) FRS data, not on V or the policy, so they are identical on every
    # outer iteration. Gather them into compact arrays once here, in sweep order, instead of re-slicing the
    # multi-GB [S, A, C, D] source arrays every iteration; the batches below are then contiguous slices
    # (views) of these. Both the improvement step (all actions) and the evaluation step (the policy-selected
    # action) read from them. Each source array is released right after its copy, so at most one of them is
    # held twice. (Not used after the DP returns.)
    lb_c = svmdp.S_idx_lb[states_to_update]
    svmdp.S_idx_lb = None
    ub_c = svmdp.S_idx_ub[states_to_update]
    svmdp.S_idx_ub = None
    p_c = svmdp.P_full[states_to_update]
    svmdp.P_full = None

    if RND_SWEEPS or sweep_priority is not None:
        starts = range(0, len(states_to_update), BATCH_SIZE)
        state_batches = [states_to_update[i:i + BATCH_SIZE] for i in starts]
        imp_batches = [(lb_c[i:i + BATCH_SIZE], ub_c[i:i + BATCH_SIZE], p_c[i:i + BATCH_SIZE]) for i in starts]
    else:
        state_batches = [states_to_update]
        imp_batches = [(lb_c, ub_c, p_c)]

    # Union boxes of the noise cells. The successor boxes of one state-action pair are the same
    # forward-reachable box widened by each noise cell's interval, so they overlap. If their union (the
    # largest over all state-action pairs) has fewer cells than the noise cells' boxes together, V is read
    # once over the union and each noise cell's minimum is taken from it with a mask: the same minima with
    # fewer successor lookups. Decided once per model; otherwise the per-noise-cell kernels above are kept.
    box_kw = getattr(svmdp.box_to_ids, 'keywords', None)
    num_noise = p_c.shape[2]
    if box_kw is not None and num_noise > 1 and len(states_to_update) > 0:

        @jax.jit
        def union_extent(lb, ub, p):
            active = (p > 0)[..., None]
            lb, ub = lb.astype(jnp.int32), ub.astype(jnp.int32)
            lo = jnp.min(jnp.where(active, lb, jnp.iinfo(jnp.int32).max), axis=-2)
            hi = jnp.max(jnp.where(active, ub, jnp.iinfo(jnp.int32).min), axis=-2)
            span = jnp.where(jnp.any(active, axis=-2), hi - lo + 1, 1)
            # Dimensions in which the active noise cells' boxes of a state-action pair differ
            varies = jnp.any(active & ((lb != lo[..., None, :]) | (ub != hi[..., None, :])), axis=(0, 1, 2))
            return jnp.max(span.reshape(-1, span.shape[-1]), axis=0), varies

        union_span = np.ones(lb_c.shape[-1], dtype=np.int64)
        varies = np.zeros(lb_c.shape[-1], dtype=bool)
        for i in range(0, len(states_to_update), 65536):
            span_i, varies_i = union_extent(lb_c[i:i + 65536], ub_c[i:i + 65536], p_c[i:i + 65536])
            union_span, varies = np.maximum(union_span, np.asarray(span_i)), varies | np.asarray(varies_i)
        union_span = tuple(int(x) for x in union_span)
        noise_dims = tuple(int(d) for d in np.flatnonzero(varies))

        # Work per state-action pair: lookups of the union, plus a mask over the noise dimensions per noise cell,
        # against the lookups of all noise cells' boxes
        union_cells = int(np.prod(union_span))
        union_work = union_cells + num_noise * int(np.prod([union_span[d] for d in noise_dims]))
        slice_work = num_noise * int(np.prod(box_kw['max_span']))
        use_union = union_work < slice_work
        logger.info(f'- Noise-cell boxes: union span {union_span} = {union_cells} cells, noise in dims {noise_dims}; '
                    f'work {union_work} (union) vs {slice_work} (per noise cell) -> '
                    + ('union kernels' if use_union else 'per-noise-cell kernels'))

        if use_union:
            union_box_to_ids = partial(svmdp.box_to_ids.func, **{**box_kw, 'max_span': union_span})
            vmap_state_policy_improvement, vmap_state_policy_evaluation = _union_kernels(union_box_to_ids, union_span, noise_dims)

    # Per-batch updates that also write the batch's new values into V. V is donated, so XLA updates it
    # in place instead of copying the whole value vector after every batch (V.at[...].set outside jit).
    # Callers must not use a V after passing it in, and copy V at the host boundary (jnp.array / np.array).
    @partial(jax.jit, donate_argnums=(0,))
    def improve_batch(V, state_batch, idx_lb, idx_ub, probs):
        V_batch, policy_batch = vmap_state_policy_improvement(idx_lb, idx_ub, probs, V)
        return V.at[state_batch].set(V_batch), policy_batch

    @partial(jax.jit, donate_argnums=(0,))
    def evaluate_batch(V, state_batch, idx_lb, idx_ub, probs):
        return V.at[state_batch].set(vmap_state_policy_evaluation(idx_lb, idx_ub, probs, V))

    logger.info(f'- SVMDP defined (took {time.time() - start_time:.3f}s)')
    start_time = time.time()
    
    satprob = args.satprob
    pbar = tqdm(desc='Iteration', total=None, unit='it', dynamic_ncols=True, leave=True)
    if not policy_iteration:
        # Value iteration
        for iteration in range(max_iterations):
            pbar.update(1)
            postfix_dict = {}
            if s0 is not None:
                postfix_dict[f'v[{s0}]'] = f'{V[s0]:.8f}'
                postfix_dict[f'v_avg'] = f'{np.mean(V[states_to_update]):.8f}'
            pbar.set_postfix(postfix_dict)
            
            V_old = V.copy()
                
            # Policy evaluation + improvement (V stays on the device and each batch is written in place;
            # later batches see earlier updates, as before)
            Vd = jnp.array(V)
            policy_refs = []
            for state_batch, (lb_d, ub_d, p_d) in zip(state_batches, imp_batches):
                Vd, policy_batch = improve_batch(Vd, state_batch, lb_d, ub_d, p_d)
                policy_refs.append(policy_batch)
            V = np.array(Vd, dtype=args.floatprecision)
            for state_batch, policy_batch in zip(state_batches, jax.device_get(policy_refs)):
                policy[state_batch] = np.asarray(policy_batch, dtype=np.int32)

            if float(V[s0]) >= satprob:
                pbar.write(f'Threshold reached: v[{s0}]={float(V[s0]):.8f} >= {satprob} after {iteration + 1} iterations')
                break

            # Check convergence
            if np.max(np.abs(V - V_old)) < epsilon:
                pbar.write(f'Converged after {iteration + 1} iterations')
                break

    else:
        partial_convergence_reached = False
        sat_policy = False
        delta = float('inf')

        # Persistent caches for the incremental policy-evaluation gather (Opt 1, see prepare block below).
        # eval_batches holds the policy-selected FRS inputs per batch; eval_policy_state[s] records which
        # action is currently reflected for state s, so we only re-gather rows whose policy changed.
        eval_batches = [None] * len(imp_batches)
        eval_policy_state = np.empty(svmdp.nr_states, dtype=np.int32)
        num_batches = len(state_batches)

        # Policy iteration
        for iteration in range(max_iterations):

            pbar.update(1)

            # Policy evaluation
            i = 0
            # The policy is fixed throughout policy evaluation, so slice the policy-selected action out
            # of the precomputed imp_batches once here, rather than re-slicing on every inner sweep.
            # Combined fancy-indexing ([rows, sel]) selects one action per state directly, avoiding the
            # earlier double-indexing intermediate array.
            for bi, (sb, (lb_a, ub_a, p_a)) in enumerate(zip(state_batches, imp_batches)):
                sel = policy[sb]
                if eval_batches[bi] is None:
                    rows = np.arange(len(sel))
                    eval_batches[bi] = [lb_a[rows, sel], ub_a[rows, sel], p_a[rows, sel]]
                else:
                    changed = np.flatnonzero(sel != eval_policy_state[sb])
                    if changed.size:
                        csel = sel[changed]
                        ev_lb, ev_ub, ev_p = eval_batches[bi]
                        ev_lb[changed] = lb_a[changed, csel]
                        ev_ub[changed] = ub_a[changed, csel]
                        ev_p[changed] = p_a[changed, csel]
                eval_policy_state[sb] = sel

            # Keep V resident on the device across the whole evaluation phase. Each batch update is an
            # in-place scatter (Gauss-Seidel: later batches see earlier updates, as before), so we
            # only pull V back to the host once per sweep for the convergence/postfix checks instead
            # of blocking on a device_get after every one of the ~len(state_batches) batches.
            Vd = jnp.array(V)
            while True:

                postfix_dict = {}
                if s0 is not None:
                    postfix_dict[f'eval_it'] = f'{i}'
                    postfix_dict[f'v[{s0}]'] = f'{V[s0]:.8f}'
                    postfix_dict[f'v_avg'] = f'{np.mean(V[states_to_update]):.8f}'
                    postfix_dict[f'max(v-v_old)'] = f'{delta:.8f}'

                    # Check if policy is above the preset threshold quality
                    if float(V[s0]) >= satprob:
                        logger.info(f'Policy is above the satisfaction threshold {satprob:.2f} after {iteration + 1} iterations')
                        # Policy is already good enough, so skip policy improvement and only keep evaluating it until convergence
                        sat_policy = True
                    else:
                        sat_policy = False
                pbar.set_postfix(postfix_dict)

                V_old = V

                # Policy evaluation only (V stays on the device; scatter each batch's result back in)
                for j,(state_batch, (ev_lb, ev_ub, ev_p)) in enumerate(zip(state_batches, eval_batches)):
                    # postfix_dict[f'eval_it'] = f'{i} (eval batch {j}/{num_batches})'
                    # pbar.set_postfix(postfix_dict)

                    Vd = evaluate_batch(Vd, state_batch, ev_lb, ev_ub, ev_p)
                V = np.array(Vd, dtype=args.floatprecision)

                delta = np.max(np.abs(V - V_old))
                if (
                    delta < epsilon
                    or i >= max_eval_it
                    or (float(V[s0]) >= satprob)
                    or (
                        not partial_convergence_reached
                        and i >= min(phase1_initial_it + iteration * phase1_increment_it, phase1_max_it)
                    )
                ):
                    break

                i += 1

            # Policy evaluation + improvement
            V_before_improvement = V.copy()

            if not sat_policy:
                # Same device-resident, Gauss-Seidel pattern as evaluation: scatter each batch's
                # improved values back into the on-device V and only sync once at the end (for V and
                # for the whole policy update) rather than blocking after every batch.
                Vd = jnp.array(V)
                policy_refs = []
                for j,(state_batch, (lb_d, ub_d, p_d)) in enumerate(zip(state_batches, imp_batches)):
                    # postfix_dict[f'eval_it'] = f'{i} (improv batch {j}/{num_batches})'
                    # pbar.set_postfix(postfix_dict)

                    Vd, policy_batch = improve_batch(Vd, state_batch, lb_d, ub_d, p_d)
                    policy_refs.append(policy_batch)
                V = np.array(Vd, dtype=args.floatprecision)
                for state_batch, policy_batch in zip(state_batches, jax.device_get(policy_refs)):
                    policy[state_batch] = np.asarray(policy_batch, dtype=np.int32)

            if float(V[s0]) >= satprob:
                pbar.write(f'Threshold reached: v[{s0}]={float(V[s0]):.8f} >= {satprob} after {iteration + 1} iterations')
                break

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

    pbar.close()

    # Extract policy inputs from policy. A_id is the shared list of action ids, so the
    # chosen action index maps directly through it (identity when A_id == range(num_actions)).
    A_id_arr = np.asarray(svmdp.A_id)
    policy_labels = np.full_like(policy, fill_value=-1)
    valid = policy != -1
    policy_labels[valid] = A_id_arr[policy[valid]]

    logger.info(f'- Policy synthesis finished (took {time.time() - start_time:.3f}s)\n')

    return V, policy_labels
