import numpy as np
import jax.numpy as jnp

class SVMDP:
    """
    Class to construct the SVMDP abstraction.
    """

    def __init__(self, partition, states, x0, goal_regions, critical_regions, P_full, S_idx_lb, S_idx_ub, box_to_ids, A_id, P_absorbing):
        '''
        Generate the SVMDP abstraction

        :param partition:
        :param states:
        :param x0:
        :param goal_regions:
        :param critical_regions:
        :param P_full: Per-(state, action, noise cell) probability mass, shape [S, A, nc]
        :param S_idx_lb: Lower grid-index bound of each forward-reachable box, shape [S, A, nc, D]
        :param S_idx_ub: Upper grid-index bound of each forward-reachable box, shape [S, A, nc, D]
        :param box_to_ids: Single-box -> successor state IDs function (shape [D] -> [prod(max_span)])
            with the static partition/grid data bound (see successor_ids.make_box_to_ids). The DP
            recomposes successor IDs on the fly from the boxes instead of storing them materialised.
        :param A_id: Single shared list of enabled action ids (every state has all actions
            enabled). The action index chosen by the DP maps to a label through this list.
        :param P_absorbing:
        '''

        self.states = states

        self.goal_regions = goal_regions
        self.critical_regions = critical_regions
        self.P_full = P_full
        self.S_idx_lb = S_idx_lb
        self.S_idx_ub = S_idx_ub
        self.box_to_ids = box_to_ids
        self.A_id = A_id
        self.P_absorbing = P_absorbing

        # Define initial state
        self.s_init = partition.x2state(x0)[0]

        # Define absorbing state
        self.absorbing_state = np.max(self.states) + 1

        # Number of states
        self.nr_states = len(self.states) + 1