import numpy as np


class SVMDP:
    """
    Set-valued Markov Decision Process (SVMDP) abstraction.

    Transition structure for a state s under action a:

        1. Probabilistic branching: noise cell c is drawn with probability noise_probs[c].
           The residual mass noise_remainder goes directly to the absorbing state.
        2. Nondeterministic branching: given noise cell c, the next state is an adversarial
           choice from the successor set  Succ(s, a, c)  (see get_successors).

    Bellman equation (reach-avoid, adversary minimises):

        V(s) = max_{a in A_id[s]}  sum_c  noise_probs[c] * min_{s' in Succ(s,a,c)} V(s')
                                   + noise_remainder * V(absorbing_state)

    Goal / critical / absorbing states carry fixed values 1, 0 and 0 respectively.
    """

    def __init__(
        self,
        partition,
        states,
        x0,
        goal_regions,
        critical_regions,
        A_id,
        actions,
        noise_probs,
        noise_remainder,
    ):
        """
        :param partition:       RectangularPartition or SparsePartition instance.
        :param states:          1-D integer array of state indices in the partition.
        :param x0:              Initial continuous state.
        :param goal_regions:    Boolean array (len = nr_states-1); True for goal cells.
        :param critical_regions: Boolean array (len = nr_states-1); True for unsafe cells.
        :param A_id:            Dict  s -> array of enabled action indices for state s.
        :param actions:         RectangularForward instance; must have noise_frs_cell_ids
                                attribute of shape (nr_states, num_actions, num_noise_cells),
                                dtype object, each element a 1-D int32 array of flat
                                partition cell IDs.
        :param noise_probs:     1-D array of length num_noise_cells with noise cell probs.
        :param noise_remainder: Scalar probability mass outside all noise cells.
        """
        self.states = states
        self.goal_regions = goal_regions
        self.critical_regions = critical_regions

        self.s_init = partition.x2state(x0)[0]
        self.absorbing_state = int(np.max(states)) + 1
        self.nr_states = len(states) + 1

        self.A_id = A_id
        self.noise_probs = np.asarray(noise_probs, dtype=np.float64)  # (num_noise_cells,)
        self.noise_remainder = float(noise_remainder)
        self.num_noise_cells = len(self.noise_probs)

        # RectangularForward — holds noise_frs_cell_ids[s, a, c]
        self.actions = actions

        # Sorted flat linear keys and corresponding partition state IDs (numpy, for solver).
        # These mirror partition.region_linear_idx / region_linear_state but as plain numpy.
        self._linear_idx = np.asarray(partition.region_linear_idx)
        self._linear_state = np.asarray(partition.region_linear_state)

    # ------------------------------------------------------------------
    # Successor-set query
    # ------------------------------------------------------------------

    def get_successors(self, s, a, c):
        """
        Return the unique partition state IDs reachable from state s under action a
        when noise cell c is realised.

        Flat cell IDs in noise_frs_cell_ids[s, a, c] that do not correspond to any
        active partition cell are mapped to absorbing_state.

        :param s: Partition state index (int).
        :param a: Action index (int, as stored in A_id).
        :param c: Noise cell index (int, 0 .. num_noise_cells-1).
        :returns: 1-D int32 numpy array of successor state IDs (unique, sorted).
        """
        flat_ids = self.actions.noise_frs_cell_ids[s, a, c]
        if len(flat_ids) == 0:
            return np.array([self.absorbing_state], dtype=np.int32)

        pos = np.searchsorted(self._linear_idx, flat_ids)
        valid = (pos < len(self._linear_idx)) & (self._linear_idx[pos] == flat_ids)
        state_ids = np.where(valid, self._linear_state[pos], self.absorbing_state)
        return np.unique(state_ids).astype(np.int32)

    def get_successors_batch(self, s, a):
        """
        Return successor sets for all noise cells at once for a given (s, a) pair.

        :param s: Partition state index (int).
        :param a: Action index (int).
        :returns: List of num_noise_cells arrays, each a 1-D int32 array of unique
                  successor state IDs (one entry per noise cell).
        """
        return [self.get_successors(s, a, c) for c in range(self.num_noise_cells)]
