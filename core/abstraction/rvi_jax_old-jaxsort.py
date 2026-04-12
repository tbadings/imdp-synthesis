import numpy as np
from tqdm import tqdm
from copy import copy, deepcopy
import jax
import jax.numpy as jnp
import time
import argparse
from typing import Optional, Tuple
from jaxtyping import Array, UInt8, Bool, Float32, PyTree

from core.abstraction.imdp import IMDP
from core.utils import jit_compile_count

def RVI_JAX(
    args: argparse.Namespace, 
    imdp: IMDP, 
    s0: Optional[int] = None, 
    max_iterations: int = 1000, 
    epsilon: float = 1e-6, 
    RND_SWEEPS: bool = False, 
    BATCH_SIZE: int = 2000, 
    policy_iteration: bool = False,
    return_Q_values: bool = False
) -> Tuple[Float32[Array, "nr_states"], Bool, UInt8[Array, "nr_states"], Float32[Array, "nr_states p"]]:

    """
    Robust value iteration for interval MDPs.
    
    :param args: Argument namespace
    :param imdp: Instance of IMDP class
    :param s0: Initial state for tracking
    :param max_iterations: Maximum number of iterations
    :param epsilon: Convergence threshold
    :param RND_SWEEPS: Whether to use random state sweeps
    :param BATCH_SIZE: Batch size for state updates
    :param policy_iteration: Whether to use policy iteration instead of value iteration
    :param return_Q_values: Whether to return Q-values for all state-action pairs
    :return: Tuple of (values, Q-values, policy_labels, policy_inputs)
    """

    iterations_phase1 = 100

    #####

    def compute_lower_val(
        prob_lb: Float32[Array, "nr_successors"], 
        prob_ub: Float32[Array, "nr_successors"], 
        successor_values: Float32[Array, "nr_successors"]
    ) -> Float32:

        """
        Compute the robust value for a given action based on the probability intervals and successor values.

        :param prob_lb: Lower bounds of transition probabilities for the successor states
        :param prob_ub: Upper bounds of transition probabilities for the successor states
        :param successor_values: Values of the successor states
        :return: The robust value for the action
        """
        
        # Budget is the total probability mass we can assign to the successors
        budget = 1.0 - jnp.sum(prob_lb)
        
        # Sort the values for these successor states
        sort = jnp.argsort(successor_values)
        sorted_lb = prob_lb[sort]
        sorted_ub = prob_ub[sort]
        
        # Vectorized computation of extra probabilities
        extra_probs = jnp.minimum(sorted_ub - sorted_lb, budget)
        cumsum = jnp.cumsum(extra_probs)
        extra_probs = jnp.minimum(extra_probs, jnp.maximum(0.0, budget - cumsum + extra_probs))
        
        probs = sorted_lb + extra_probs
        lower_val = probs @ successor_values[sort]
        
        return lower_val

    vmap_compute_lower_val = jax.jit(jax.vmap(compute_lower_val, in_axes=(0, 0, 0), out_axes=0))

    def state_policy_improvement(
        successors_slice: UInt8[Array, "nr_actions nr_successors"],
        prob_lb_slice: Float32[Array, "nr_actions nr_successors"],
        prob_ub_slice: Float32[Array, "nr_actions nr_successors"],
        V: Float32[Array, "nr_states"]
    ) -> Tuple[Float32, UInt8]:

        """
        Perform policy improvement for a given state by computing the robust values for all actions.

        :param successors_slice: Slice of successor states for all actions
        :param prob_lb_slice: Slice of lower bounds of transition probabilities for all actions
        :param prob_ub_slice: Slice of upper bounds of transition probabilities for all actions
        :param V: Current value function
        :return: Tuple of (maximum robust value, index of the action with maximum robust value)
        """

        # Retrieve the values for the successor states, including absorbing state
        successor_values = V[successors_slice]

        # Compute lower value for all actions in parallel using JAX vectorization
        lower_vals = vmap_compute_lower_val(prob_lb_slice, prob_ub_slice, successor_values)

        return jnp.max(lower_vals), jnp.argmax(lower_vals)

    vmap_state_policy_improvement = jax.jit(jax.vmap(state_policy_improvement, in_axes=(0, 0, 0, None), out_axes=(0, 0)))

    def state_policy_evaluation(
        successors_slice: UInt8[Array, "nr_actions nr_successors"],
        prob_lb_slice: Float32[Array, "nr_actions nr_successors"],
        prob_ub_slice: Float32[Array, "nr_actions nr_successors"],
        V: Float32[Array, "nr_states"]
    ) -> Float32:

        """
        Perform policy evaluation for a given state by computing the robust value for the action specified by the current policy.

        :param successors_slice: Slice of successor states for the action specified by the current policy
        :param prob_lb_slice: Slice of lower bounds of transition probabilities for the action specified by the current policy
        :param prob_ub_slice: Slice of upper bounds of transition probabilities for the action specified by the current policy
        :param V: Current value function
        :return: The robust value for the action specified by the current policy
        """

        # Retrieve the values for the successor states, including absorbing state
        successor_values = V[successors_slice]

        # Compute lower value for the action specified by the current policy
        lower_val = compute_lower_val(prob_lb_slice, prob_ub_slice, successor_values)

        return lower_val

    vmap_state_policy_evaluation = jax.jit(jax.vmap(state_policy_evaluation, in_axes=(0, 0, 0, None), out_axes=(0)))

    #####
    # Padding the probability intervals and successor values for JAX vectorization
    total_actions = np.array([len(imdp.A_id[s]) for s in imdp.states if s in imdp.A_id])
    max_actions = np.max(total_actions) if len(total_actions) > 0 else 0
    max_successors = max([imdp.S_id[s].shape[1] + 1 for s in imdp.states if s in imdp.S_id]) # +1 for absorbing state

    if policy_iteration:
        print(f'=== Run robust policy iteration ===')
    else:
        print(f'=== Run robust value iteration ===')

    print(f'- Number of states: {len(imdp.states)}')
    print(f'- Total number of choices: {np.sum(total_actions)} (total number of state-action pairs)')
    print(f'- Max number of actions per state: {max_actions}')
    print(f'- Max number of successor states per action: {max_successors}')

    # Filling the following arrays is faster with NumPy
    JAX_successors_array = np.full((len(imdp.states), max_actions, max_successors), -1, dtype=np.int32)
    JAX_prob_lb_array = np.zeros((len(imdp.states), max_actions, max_successors), dtype=args.floatprecision)
    JAX_prob_ub_array = np.zeros((len(imdp.states), max_actions, max_successors), dtype=args.floatprecision)

    for s in imdp.states:
        if s not in imdp.A_id:
            continue
        # Fill in the dense array
        successors = imdp.S_id[s]
        num_actions, num_successors = successors.shape
        JAX_successors_array[s, :num_actions, :num_successors] = successors
        JAX_prob_lb_array[s, :num_actions, :num_successors] = imdp.P_full[s][:, :, 0]
        JAX_prob_ub_array[s, :num_actions, :num_successors] = imdp.P_full[s][:, :, 1]
        # Add the absorbing state as a successor in the final column (max_successors-1) for all actions
        JAX_successors_array[s, :num_actions, max_successors-1] = imdp.absorbing_state
        JAX_prob_lb_array[s, :num_actions, max_successors-1] = imdp.P_absorbing[s][:, 0]
        JAX_prob_ub_array[s, :num_actions, max_successors-1] = imdp.P_absorbing[s][:, 1]

    print(f'- Padding and array construction done')

    #####

    print('- Set states to update...')
    states_with_enabled_actions = np.array([True if s in imdp.A_id and len(imdp.A_id[s]) > 0 else False for s in imdp.states])
    update_mask = ~imdp.goal_regions & ~imdp.critical_regions & (imdp.states != imdp.absorbing_state) & states_with_enabled_actions
    states_to_update = imdp.states[update_mask]
    states_not_to_update = imdp.states[~update_mask]
    
    # Initialize value function and policy
    V = np.zeros(imdp.nr_states, dtype=args.floatprecision)
    if len(imdp.goal_regions) > 0:
        V[:-1][imdp.goal_regions] = 1.0 # [:-1] to exclude the absorbing state

    policy = np.zeros(imdp.nr_states, dtype=np.int32)
    policy[states_not_to_update] = -1  # Mark states that we do not update with a special action index (e.g., -1)
    
    pbar = tqdm(range(max_iterations), desc='Iteration')

    if RND_SWEEPS:
        # Shuffle and batch states_to_update
        states_to_update = np.random.permutation(states_to_update)
        state_batches = [states_to_update[i:i + BATCH_SIZE] for i in range(0, len(states_to_update), BATCH_SIZE)]
    else:
        state_batches = [states_to_update]

    JAX_successors_array = jax.device_put(JAX_successors_array, args.rvi_device)
    JAX_prob_lb_array = jax.device_put(JAX_prob_lb_array, args.rvi_device)
    JAX_prob_ub_array = jax.device_put(JAX_prob_ub_array, args.rvi_device)

    if not policy_iteration:
        # Value iteration
        for iteration in pbar:
            postfix_dict = {}
            if s0 is not None:
                postfix_dict[f'v[{s0}]'] = f'{V[s0]:.6f}'
                postfix_dict[f'v_avg'] = f'{np.mean(V[states_to_update]):.6f}'
            pbar.set_postfix(postfix_dict)
            
            V_old = V.copy()
                
            # Policy evaluation + improvement
            for state_batch in state_batches:
                V_batch, policy_batch = vmap_state_policy_improvement(
                                            JAX_successors_array[state_batch], 
                                            JAX_prob_lb_array[state_batch], 
                                            JAX_prob_ub_array[state_batch], 
                                            V)
                V_batch, policy_batch = jax.device_get((V_batch, policy_batch))
                V[state_batch] = np.asarray(V_batch, dtype=args.floatprecision)
                policy[state_batch] = np.asarray(policy_batch, dtype=np.int32)
            
            # Check convergence
            if np.max(np.abs(V - V_old)) < epsilon:
                print(f'Converged after {iteration + 1} iterations')
                break

    else:
        bool = False

        # Policy iteration
        for iteration in pbar:
            postfix_dict = {}
            if s0 is not None:
                postfix_dict[f'v[{s0}]'] = f'{V[s0]:.6f}'
                postfix_dict[f'v_avg'] = f'{np.mean(V[states_to_update]):.6f}'
            pbar.set_postfix(postfix_dict)

            # Policy evaluation
            i = 0
            t = time.time()
            while True: # TODO: Remove this hardcoding
                # print(f'- Policy evaluation iteration {i + 1}...')
                V_old = V.copy()
                
                # Policy evaluation only
                for state_batch in state_batches:
                    V_eval = vmap_state_policy_evaluation(
                                                JAX_successors_array[state_batch, policy[state_batch]], 
                                                JAX_prob_lb_array[state_batch, policy[state_batch]], 
                                                JAX_prob_ub_array[state_batch, policy[state_batch]], 
                                                V)
                    V[state_batch] = np.asarray(jax.device_get(V_eval), dtype=args.floatprecision)
                
                if np.max(np.abs(V - V_old)) < epsilon or (bool is False and i > iterations_phase1):
                    break

                i += 1
            # print(f'- Policy evaluation took: {time.time() - t:.3f} sec')

            # Policy evaluation + improvement
            t = time.time()
            policy_old = policy.copy()

            for state_batch in state_batches:
                V_batch, policy_batch = vmap_state_policy_improvement(
                                            JAX_successors_array[state_batch], 
                                            JAX_prob_lb_array[state_batch], 
                                            JAX_prob_ub_array[state_batch], 
                                            V)
                V_batch, policy_batch = jax.device_get((V_batch, policy_batch))
                V[state_batch] = np.asarray(V_batch, dtype=args.floatprecision)
                policy[state_batch] = np.asarray(policy_batch, dtype=np.int32)
            
            # Check convergence
            if np.all(policy == policy_old):
                if bool:
                    print(f'Converged after {iteration + 1} iterations')
                    break
                else:
                    print(f'Partial convergence after {iteration + 1} iterations. Decrease epsilon to refine values...')
                    bool = True

    # Extract policy inputs from policy
    policy_labels = np.full_like(policy, fill_value=-1)
    for s in imdp.states:
        policy_labels[s] = imdp.A_id[s][int(policy[s])] if policy[s] != -1 and s in imdp.A_id else -1

    policy_inputs = imdp.actions_inputs[policy_labels]

    return V, policy_labels, policy_inputs