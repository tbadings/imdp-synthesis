import numpy as np
import jax.numpy as jnp

class SVMDP:
    """
    Class to construct the SVMDP abstraction.
    """

    def __init__(self, partition, states, x0, goal_regions, critical_regions, interval_lb, interval_ub, box_probs,
                 slots, max_slice, union_span, grid, A_id, P_absorbing):
        '''
        Generate the SVMDP abstraction

        :param partition:
        :param states:
        :param x0:
        :param goal_regions:
        :param critical_regions:
        :param interval_lb: Lower grid indices of the merged successor intervals per (state, action), dimension
            after dimension, shape [S, A, sum(slots)] (see forward_reachability.RectangularForward)
        :param interval_ub: Upper grid indices of the merged successor intervals, shape [S, A, sum(slots)]
        :param box_probs: Probability of each successor box (combination of one interval per dimension, C order
            over the dimensions), shape [S, A, prod(slots)]
        :param slots: Number of stored successor intervals per dimension (tuple of D ints)
        :param max_slice: Largest span of a successor interval per dimension (tuple of D ints)
        :param union_span: Largest span of the union of a pair's successor intervals per dimension (tuple of D ints)
        :param grid: TiledGrid of the partition (see successor_ids), on which the DP keeps the state values
        :param A_id: Single shared list of enabled action ids (every state has all actions
            enabled). The action index chosen by the DP maps to a label through this list.
        :param P_absorbing:
        '''

        self.states = states

        self.goal_regions = goal_regions
        self.critical_regions = critical_regions
        self.interval_lb = interval_lb
        self.interval_ub = interval_ub
        self.box_probs = box_probs
        self.slots = slots
        self.max_slice = max_slice
        self.union_span = union_span
        self.grid = grid
        self.A_id = A_id
        self.P_absorbing = P_absorbing

        # Define initial state
        self.s_init = partition.x2state(x0)[0]

        # Define absorbing state
        self.absorbing_state = np.max(self.states) + 1

        # Number of states
        self.nr_states = len(self.states) + 1
