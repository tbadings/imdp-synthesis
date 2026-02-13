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

class IMDP:
    """
    Class to construct the IMDP abstraction.
    """

    def __init__(self, partition, states, actions_inputs, x0, goal_regions, critical_regions, P_full, P_id, P_absorbing):
        '''
        Generate the IMDP abstraction

        :param partition:
        :param actions_inputs:
        :param states:
        :param x0:
        :param goal_regions:
        :param critical_regions:
        :param P_full:
        :param P_id:
        :param P_absorbing:
        '''
        self.actions_inputs = actions_inputs
        self.states = states

        self.goal_regions = goal_regions
        self.critical_regions = critical_regions
        self.P_full = P_full
        self.P_id = P_id
        self.P_absorbing = P_absorbing

        # Define initial state
        self.s_init = partition.x2state(x0)[0]

        # Define absorbing state
        self.absorbing_state = np.max(self.states) + 1

        # Number of states
        self.nr_states = len(self.states) + 1

        # Create map from action index to action labels
        self.A_idx2lab = {}
        for s in self.states:
            self.A_idx2lab[s] = {idx: a for idx, a in enumerate(self.P_id[s].keys())} 

def RVI(imdp, s0=None, max_iterations=1000, epsilon=1e-6):
    """
    Robust value iteration for interval MDPs.
    
    :param imdp: Instance of IMDP class
    :param max_iterations: Maximum number of iterations
    :param epsilon: Convergence threshold
    :return: Tuple of (lower_bounds, upper_bounds) for all states
    """
    n_states = imdp.nr_states
    V = np.zeros(n_states)
    
    # Mark goal and absorbing states
    for s in imdp.goal_regions:
        V[s] = 1
    
    pbar = tqdm(range(max_iterations), desc='Iteration')
    for iteration in pbar:
        postfix_dict = {}
        if s0 is not None:
            postfix_dict[f'v[{s0}]'] = f'{V[s0]:.6f}'
        pbar.set_postfix(postfix_dict)
        
        V_old = V.copy()
        for s in imdp.states:
            if s in imdp.goal_regions or s in imdp.critical_regions or s == imdp.absorbing_state:
                continue
            
            if len(imdp.P_id[s]) == 0:
                V[s] = 0
                continue
            
            lower_vals = []

            for a_idx, (_, successors) in enumerate(imdp.P_id[s].items()):
                # Add absorbing state as successor
                successors_plus_abs = np.append(successors, imdp.absorbing_state)
                probabilities_plus_abs = np.vstack((imdp.P_full[s][a_idx], imdp.P_absorbing[s][a_idx]))
                prob_lb = probabilities_plus_abs[:, 0]
                prob_ub = probabilities_plus_abs[:, 1]

                # Retrieve the values for the successor states, including absorbing state
                successor_values = V[successors_plus_abs]

                # Budget is the total probability mass we can assign to the successors, which is 1 minus the sum 
                # of the lower bounds of the probability intervals for all successors (including absorbing state).
                budget = 1 - np.sum(prob_lb)

                # Sort the values for these successor states
                sort = np.argsort(successor_values)

                lower_val = 0

                for pos in sort:
                    # The extra probability mass we can assign to this successor is the difference between the 
                    # upper and lower bound of the probability interval, but we cannot exceed the remaining budget.
                    extra_prob = min(prob_ub[pos] - prob_lb[pos], budget)
                    prob = prob_lb[pos] + extra_prob
                    lower_val += prob * successor_values[pos]

                    # Decrease budget
                    budget -= extra_prob

                lower_vals.append(lower_val)
                
            
            V[s] = max(lower_vals) if lower_vals else 0
        
        # Check convergence
        if np.max(np.abs(V - V_old)) < epsilon:
            print(f'Converged after {iteration + 1} iterations')
            break
    
    return V

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
    
    P_id_plusAbs = {}
    P_full_plusAbs = {}

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

    ######

    # Append the absorbing state to the IMDP object
    for s in imdp.states:
        P_id_plusAbs[s] = {}
        P_full_plusAbs[s] = {}
        for a_idx, (a_label, successors) in enumerate(imdp.P_id[s].items()):
            P_id_plusAbs[s][a_label] = np.append(successors, imdp.absorbing_state)
            P_full_plusAbs[s][a_idx] = np.vstack((imdp.P_full[s][a_idx], imdp.P_absorbing[s][a_idx]))

    #####
    # Padding the probability intervals and successor values for JAX vectorization
    max_actions = max([len(P_id_plusAbs[s]) for s in imdp.states])
    max_successors = max([len(P_id_plusAbs[s][a]) + 1 for s in imdp.states for a in P_id_plusAbs[s].keys()])

    print(f'- Max actions per state: {max_actions}')
    print(f'- Max successors per action: {max_successors}')

    # Filling the following arrays is faster with NumPy
    JAX_successors_array = np.full((len(imdp.states), max_actions, max_successors), -1, dtype=np.int32)
    JAX_prob_lb_array = np.zeros((len(imdp.states), max_actions, max_successors), dtype=np.float32)
    JAX_prob_ub_array = np.zeros((len(imdp.states), max_actions, max_successors), dtype=np.float32)

    for s in imdp.states:
        for a_idx, (a_label, successors) in enumerate(P_id_plusAbs[s].items()):
            JAX_successors_array[s, a_idx, :len(successors)] = successors
            JAX_prob_lb_array[s, a_idx, :len(successors)] = P_full_plusAbs[s][a_idx][:, 0]
            JAX_prob_ub_array[s, a_idx, :len(successors)] = P_full_plusAbs[s][a_idx][:, 1]

    print(f'- Padding and array construction done')

    #####

    states_to_update = np.array([s for s in imdp.states 
                            if s not in imdp.goal_regions and 
                               s not in imdp.critical_regions and 
                               s != imdp.absorbing_state and
                               len(imdp.P_id[s]) > 0])
    
    states_not_to_update = np.array([s for s in imdp.states 
                            if s in imdp.goal_regions or 
                               s in imdp.critical_regions or 
                               s == imdp.absorbing_state or
                               len(imdp.P_id[s]) == 0])
    
    # Initialize value function and policy
    V = np.zeros(imdp.nr_states, dtype=np.float32)
    if len(imdp.goal_regions) > 0:
        V[imdp.goal_regions] = 1.0

    policy = np.zeros(imdp.nr_states, dtype=int)
    policy[states_not_to_update] = -1  # Mark states that we do not update with a special action index (e.g., -1)
    
    pbar = tqdm(range(max_iterations), desc='Iteration')

    if RND_SWEEPS:
        # Shuffle and batch states_to_update
        states_to_update = np.random.permutation(states_to_update)
        state_batches = [states_to_update[i:i + BATCH_SIZE] for i in range(0, len(states_to_update), BATCH_SIZE)]
    else:
        state_batches = [states_to_update]

    if not policy_iteration:
        # Value iteration
        for iteration in pbar:
            postfix_dict = {}
            if s0 is not None:
                postfix_dict[f'v[{s0}]'] = f'{V[s0]:.6f}'
            pbar.set_postfix(postfix_dict)
            
            V_old = V.copy()
                
            # Policy evaluation + improvement
            for state_batch in state_batches:
                V_batch, policy_batch = vmap_state_policy_improvement(
                                            JAX_successors_array[state_batch], 
                                            JAX_prob_lb_array[state_batch], 
                                            JAX_prob_ub_array[state_batch], 
                                            V)
                V[state_batch] = np.array(V_batch, dtype=np.float32)
                policy[state_batch] = np.array(policy_batch, dtype=int)
            
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
            pbar.set_postfix(postfix_dict)

            # Policy evaluation
            i = 0
            t = time.time()
            while True:
                V_old = V.copy()
                
                # Policy evaluation only
                for state_batch in state_batches:
                    V[state_batch] = vmap_state_policy_evaluation(
                                                JAX_successors_array[state_batch, policy[state_batch]], 
                                                JAX_prob_lb_array[state_batch, policy[state_batch]], 
                                                JAX_prob_ub_array[state_batch, policy[state_batch]], 
                                                V)
                
                if np.max(np.abs(V - V_old)) < epsilon or (bool is False and i > iterations_phase1):
                    break

                i += 1
            # print(f'- Policy evaluation took: {time.time() - t:.3f} sec')

            # Policy evaluation + improvement
            t = time.time()
            policy_old = policy.copy()

            for state_batch in state_batches:
                V[state_batch], policy[state_batch] = vmap_state_policy_improvement(
                                            JAX_successors_array[state_batch], 
                                            JAX_prob_lb_array[state_batch], 
                                            JAX_prob_ub_array[state_batch], 
                                            V)
                
            # print(f'- Policy improvement took: {time.time() - t:.3f} sec')
            
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
        policy_labels[s] = imdp.A_idx2lab[s][int(policy[s])] if policy[s] != -1 else -1

    policy_inputs = imdp.actions_inputs[policy_labels]

    # Extract Q-values
    Q = {}
    if return_Q_values:
        for s in states_to_update:
            Q[s] = {}
            for a_idx, (a_label, successors) in enumerate(P_id_plusAbs[s].items()):
                prob_lb = P_full_plusAbs[s][a_idx][:, 0]
                prob_ub = P_full_plusAbs[s][a_idx][:, 1]
                successor_values = V[successors]
                Q[s][a_label] = compute_lower_val(prob_lb, prob_ub, successor_values)

    return V, Q, policy_labels, policy_inputs