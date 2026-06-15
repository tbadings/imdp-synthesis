import logging
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import time
import argparse
from typing import Optional, Tuple
from jaxtyping import Array, UInt8, Float32

from core.abstraction.imdp.imdp import IMDP
from core.utils import jit_compile_count

logger = logging.getLogger(__name__)


def RVI_JAX2(
    args: argparse.Namespace,
    imdp: IMDP,
    s0: Optional[int] = None,
    max_iterations: int = 1000,
    epsilon: float = 1e-6,
    RND_SWEEPS: bool = False,
    BATCH_SIZE: int = 2000,
    policy_iteration: bool = False,
    return_Q_values: bool = False,
) -> Tuple[Float32[Array, "nr_states"], UInt8[Array, "nr_states"]]:
    """
    Robust value iteration for interval MDPs — optimized variant.

    Optimizations over RVI_JAX:
      1. argsort moved inside JIT: XLA fuses the gather V[successors] with the sort and
         parallelises both across all vmapped states, instead of NumPy doing it single-threaded
         on the host and materialising a full intermediate array before each call.
      2. V kept on the JAX device across iterations: eliminates the NumPy↔XLA round-trip
         on the full value array every iteration; .at[].set() replaces device_get + numpy write.
      3. Static arrays (successors, prob bounds, sort_indices) transferred to device once
         before the loop instead of slicing in NumPy and device_put-ing on every iteration.
    """

    start_time = time.time()

    phase1_initial_it = 10
    phase1_increment_it = 10
    phase1_max_it = 100
    fix_policy_above_value = 2  # >1 means this feature is disabled

    #####

    def compute_lower_val(
        prob_lb: Float32[Array, "nr_successors"],
        prob_ub: Float32[Array, "nr_successors"],
        successor_values: Float32[Array, "nr_successors"],
        sort: UInt8[Array, "nr_successors"],
    ) -> Float32:
        budget = 1.0 - jnp.sum(prob_lb)
        sorted_lb = prob_lb[sort]
        sorted_ub = prob_ub[sort]
        extra_probs = jnp.minimum(sorted_ub - sorted_lb, budget)
        cumsum = jnp.cumsum(extra_probs)
        extra_probs = jnp.minimum(extra_probs, jnp.maximum(0.0, budget - cumsum + extra_probs))
        probs = sorted_lb + extra_probs
        lower_val = probs @ successor_values[sort]
        return jnp.clip(lower_val, 0.0, 1.0)

    vmap_compute_lower_val = jax.jit(jax.vmap(compute_lower_val, in_axes=(0, 0, 0, 0), out_axes=0))

    def state_policy_improvement(
        successors_slice: UInt8[Array, "nr_actions nr_successors"],
        prob_lb_slice: Float32[Array, "nr_actions nr_successors"],
        prob_ub_slice: Float32[Array, "nr_actions nr_successors"],
        V: Float32[Array, "nr_states"],
    ) -> Tuple[Float32, UInt8]:
        successor_values = V[successors_slice]                            # (nr_actions, nr_successors)
        sort_indices = jnp.argsort(successor_values, axis=-1)             # inside JIT: XLA fuses gather+sort
        lower_vals = vmap_compute_lower_val(prob_lb_slice, prob_ub_slice, successor_values, sort_indices)
        return jnp.max(lower_vals), jnp.argmax(lower_vals)

    vmap_state_policy_improvement = jax.jit(
        jax.vmap(state_policy_improvement, in_axes=(0, 0, 0, None), out_axes=(0, 0))
    )

    def state_policy_evaluation(
        successors_slice: UInt8[Array, "nr_successors"],
        prob_lb_slice: Float32[Array, "nr_successors"],
        prob_ub_slice: Float32[Array, "nr_successors"],
        V: Float32[Array, "nr_states"],
    ) -> Float32:
        successor_values = V[successors_slice]
        sort_indices = jnp.argsort(successor_values, axis=-1)             # inside JIT: XLA fuses gather+sort
        return compute_lower_val(prob_lb_slice, prob_ub_slice, successor_values, sort_indices)

    vmap_state_policy_evaluation = jax.jit(
        jax.vmap(state_policy_evaluation, in_axes=(0, 0, 0, None), out_axes=0)
    )

    #####
    # Array construction (unchanged from original)
    total_actions = np.array([len(imdp.A_id[s]) for s in imdp.states if s in imdp.A_id])
    max_actions = np.max(total_actions) if len(total_actions) > 0 else 0
    max_successors = max([imdp.S_id[s].shape[1] + 1 for s in imdp.states if s in imdp.S_id])

    if policy_iteration:
        logger.info('=== Run robust policy iteration (RVI_JAX2) ===')
    else:
        logger.info('=== Run robust value iteration (RVI_JAX2) ===')

    logger.info('- Number of states: %d', len(imdp.states))
    logger.info('- Total number of choices: %d', np.sum(total_actions))
    logger.info('- Max number of actions per state: %d', max_actions)
    logger.info('- Max number of successor states per action: %d', max_successors)

    full_successors_array = np.full((len(imdp.states), max_actions, max_successors), -1, dtype=np.int32)
    full_prob_lb_array = np.zeros((len(imdp.states), max_actions, max_successors), dtype=args.floatprecision)
    full_prob_ub_array = np.zeros((len(imdp.states), max_actions, max_successors), dtype=args.floatprecision)

    for s in imdp.states:
        if s not in imdp.A_id:
            continue
        successors = imdp.S_id[s]
        num_actions, num_successors = successors.shape
        full_successors_array[s, :num_actions, :num_successors] = successors
        full_prob_lb_array[s, :num_actions, :num_successors] = imdp.P_full[s][:, :, 0]
        full_prob_ub_array[s, :num_actions, :num_successors] = imdp.P_full[s][:, :, 1]
        full_successors_array[s, :num_actions, max_successors - 1] = imdp.absorbing_state
        full_prob_lb_array[s, :num_actions, max_successors - 1] = imdp.P_absorbing[s][:, 0]
        full_prob_ub_array[s, :num_actions, max_successors - 1] = imdp.P_absorbing[s][:, 1]

    logger.info('- Padding and array construction done')

    # Fix 3: Transfer static arrays to device once before the loop.
    d_successors = jax.device_put(full_successors_array, args.rvi_device)
    d_prob_lb = jax.device_put(full_prob_lb_array, args.rvi_device)
    d_prob_ub = jax.device_put(full_prob_ub_array, args.rvi_device)
    logger.info('- Static arrays transferred to device')

    #####

    logger.info('- Set states to update...')
    states_with_enabled_actions = np.array(
        [True if s in imdp.A_id and len(imdp.A_id[s]) > 0 else False for s in imdp.states]
    )
    update_mask = (
        ~imdp.goal_regions
        & ~imdp.critical_regions
        & (imdp.states != imdp.absorbing_state)
        & states_with_enabled_actions
    )
    states_to_update = imdp.states[update_mask]
    states_not_to_update = imdp.states[~update_mask]

    # Fix 2: Initialize V on device; use JAX functional updates to avoid NumPy round-trips.
    V_init = np.zeros(imdp.nr_states, dtype=args.floatprecision)
    if len(imdp.goal_regions) > 0:
        goal_indices = np.flatnonzero(imdp.goal_regions)  # indices of goal states (excludes absorbing)
        V_init[goal_indices] = 1.0
    V = jax.device_put(V_init, args.rvi_device)


    policy = np.zeros(imdp.nr_states, dtype=np.int32)
    policy[states_not_to_update] = -1

    logger.info(f'- IMDP defined (took {time.time() - start_time:.3f}s); start robust dynamic programming...')

    pbar = tqdm(desc='Iteration', total=None, unit='it', dynamic_ncols=True, leave=True)

    if RND_SWEEPS:
        states_to_update = np.random.permutation(states_to_update)
        state_batches = [states_to_update[i:i + BATCH_SIZE] for i in range(0, len(states_to_update), BATCH_SIZE)]
    else:
        state_batches = [states_to_update]

    if not policy_iteration:
        # Value iteration
        for iteration in range(max_iterations):
            pbar.update(1)
            postfix_dict = {}
            if s0 is not None:
                postfix_dict[f'v[{s0}]'] = f'{float(V[s0]):.6f}'
                postfix_dict[f'v_avg'] = f'{float(jnp.mean(V[states_to_update])):.6f}'
            pbar.set_postfix(postfix_dict)

            V_old = V  # Fix 2: JAX arrays are immutable; V.at[].set() creates a new array below

            for state_batch in state_batches:
                V_batch, policy_batch = vmap_state_policy_improvement(
                    d_successors[state_batch],
                    d_prob_lb[state_batch],
                    d_prob_ub[state_batch],
                    V,
                )
                V = V.at[state_batch].set(V_batch)
                policy[state_batch] = np.asarray(jax.device_get(policy_batch), dtype=np.int32)

            # Convergence check: one cheap scalar device_get per iteration
            if float(jnp.max(jnp.abs(V - V_old))) < epsilon:
                pbar.write(f'Converged after {iteration + 1} iterations')
                break

    else:
        partial_convergence_reached = False
        sat_policy = False
        delta = float('inf')

        # Policy iteration
        for iteration in range(max_iterations):
            pbar.update(1)

            # Policy evaluation inner loop
            i = 0
            while True:
                postfix_dict = {}
                if s0 is not None:
                    postfix_dict['eval_it'] = i
                    postfix_dict[f'v[{s0}]'] = f'{float(V[s0]):.6f}'
                    postfix_dict[f'v_avg'] = f'{float(jnp.mean(V[states_to_update])):.6f}'
                    postfix_dict['max(v-v_old)'] = f'{delta:.6f}'

                    if float(V[s0]) > fix_policy_above_value:
                        sat_policy = True
                    else:
                        sat_policy = False
                pbar.set_postfix(postfix_dict)

                V_old = V  # Fix 2: immutable reference; updated array created below

                for state_batch in state_batches:
                    policy_actions = policy[state_batch]
                    V_eval = vmap_state_policy_evaluation(
                        d_successors[state_batch, policy_actions],
                        d_prob_lb[state_batch, policy_actions],
                        d_prob_ub[state_batch, policy_actions],
                        V,
                    )
                    V = V.at[state_batch].set(V_eval)

                delta = float(jnp.max(jnp.abs(V - V_old)))
                if delta < epsilon or (
                    not partial_convergence_reached
                    and i > min(phase1_initial_it + iteration * phase1_increment_it, phase1_max_it)
                ):
                    break

                i += 1

            # Policy improvement
            V_before_improvement = V

            if not sat_policy:
                for state_batch in state_batches:
                    V_batch, policy_batch = vmap_state_policy_improvement(
                        d_successors[state_batch],
                        d_prob_lb[state_batch],
                        d_prob_ub[state_batch],
                        V,
                    )
                    V = V.at[state_batch].set(V_batch)
                    policy[state_batch] = np.asarray(jax.device_get(policy_batch), dtype=np.int32)

            if float(jnp.max(V - V_before_improvement)) < epsilon:
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

    # Extract policy labels (policy stays on NumPy throughout)
    policy_labels = np.full_like(policy, fill_value=-1)
    for s in imdp.states:
        policy_labels[s] = imdp.A_id[s][int(policy[s])] if policy[s] != -1 and s in imdp.A_id else -1

    # Fix 2: bring V back to NumPy to match original return type
    return np.asarray(V), policy_labels
