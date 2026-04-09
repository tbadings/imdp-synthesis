import numpy as np
import jax.numpy as jnp

class IMDP:
    """
    Class to construct the IMDP abstraction.
    """

    def __init__(self, partition, states, actions_inputs, x0, goal_regions, critical_regions, P_full, S_id, A_id, P_absorbing):
        '''
        Generate the IMDP abstraction

        :param partition:
        :param actions_inputs:
        :param states:
        :param x0:
        :param goal_regions:
        :param critical_regions:
        :param P_full:
        :param S_id:
        :param A_id:
        :param P_absorbing:
        '''
        self.actions_inputs = actions_inputs
        self.states = states

        self.goal_regions = goal_regions
        self.critical_regions = critical_regions
        self.P_full = P_full
        self.S_id = S_id
        self.A_id = A_id
        self.P_absorbing = P_absorbing

        # Define initial state
        self.s_init = partition.x2state(x0)[0]

        # Define absorbing state
        self.absorbing_state = np.max(self.states) + 1

        # Number of states
        self.nr_states = len(self.states) + 1