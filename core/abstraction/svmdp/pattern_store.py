'''
Compressed storage of the forward reachable sets (--compress_frs).

For many benchmarks, the successor intervals of a state-action pair, taken relative to the state's own grid cell,
and the probabilities of its successor boxes repeat across the partition (e.g., linear dynamics on a grid whose
cell widths align with the inputs). PatternStore keeps every distinct (relative intervals, probabilities) pattern
once, plus one pattern ID per state-action pair. Its views interval_lb, interval_ub and box_probs rebuild the
exact arrays of RectangularForward (same values and dtypes) for the rows the dynamic program reads.
'''

import numpy as np


class PatternStore:
    '''
    Distinct successor patterns and one pattern ID per (state, action).

    :param cells: Grid cell of every state, shape [S, D] (int)
    :param slots: Number of stored intervals per dimension (tuple of D ints)
    :param num_actions: Number of actions per state
    :param idx_dtype: dtype of the reconstructed interval bounds (int8 or int16)
    :param prob_dtype: dtype of the reconstructed box probabilities
    '''

    def __init__(self, cells, slots, num_actions, idx_dtype, prob_dtype):
        self.cells = np.asarray(cells, dtype=np.int32)
        self.slot_dim = np.repeat(np.arange(len(slots)), slots)
        self.idx_dtype = np.dtype(idx_dtype)
        self.prob_dtype = np.dtype(prob_dtype)
        self.ids = np.zeros((len(self.cells), num_actions), dtype=np.uint32)
        self._lookup = {}
        self._rel_lb, self._rel_ub, self._probs = [], [], []
        self.rel_lb = self.rel_ub = self.probs = None

    def add(self, start, end, lb, ub, probs):
        '''Store the patterns of states start, ..., end - 1 (arrays of shape [B, A, ...] as RectangularForward computes).'''
        base = self.cells[start:end][:, None, self.slot_dim]
        rel_lb = (np.asarray(lb, dtype=np.int32) - base).astype(np.int16)
        rel_ub = (np.asarray(ub, dtype=np.int32) - base).astype(np.int16)
        probs = np.asarray(probs, dtype=self.prob_dtype)
        B, A = rel_lb.shape[:2]
        rows = np.concatenate([rel_lb.reshape(B * A, -1).view(np.uint8), rel_ub.reshape(B * A, -1).view(np.uint8),
                               probs.reshape(B * A, -1).view(np.uint8)], axis=1)
        rows = np.ascontiguousarray(rows).view(np.dtype((np.void, rows.shape[1]))).ravel()
        unique, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
        flat_lb, flat_ub, flat_p = rel_lb.reshape(B * A, -1), rel_ub.reshape(B * A, -1), probs.reshape(B * A, -1)
        global_ids = np.empty(len(unique), dtype=np.uint32)
        for i, (key, row) in enumerate(zip(unique.tolist(), first)):
            pattern = self._lookup.get(key)
            if pattern is None:
                pattern = self._lookup[key] = len(self._rel_lb)
                self._rel_lb.append(flat_lb[row])
                self._rel_ub.append(flat_ub[row])
                self._probs.append(flat_p[row])
            global_ids[i] = pattern
        self.ids[start:end] = global_ids[inverse.reshape(-1)].reshape(B, A)

    def finalize(self):
        '''Turn the collected patterns into arrays; returns the views (interval_lb, interval_ub, box_probs).'''
        self.rel_lb = np.stack(self._rel_lb).astype(np.int32)
        self.rel_ub = np.stack(self._rel_ub).astype(np.int32)
        self.probs = np.stack(self._probs)
        self._lookup, self._rel_lb, self._rel_ub, self._probs = None, None, None, None
        return _Bounds(self, self.rel_lb), _Bounds(self, self.rel_ub), _Probs(self)

    @property
    def num_patterns(self):
        return len(self.rel_lb) if self.rel_lb is not None else len(self._rel_lb)

    @property
    def nbytes(self):
        return self.ids.nbytes + self.cells.nbytes + sum(
            a.nbytes for a in (self.rel_lb, self.rel_ub, self.probs) if a is not None)


class _Bounds:
    '''Read-only view of interval_lb or interval_ub: X[states] (shape [B, A, sum(slots)]) or X[states, actions].'''

    def __init__(self, store, table):
        self.store, self.table = store, table

    def __getitem__(self, key):
        store = self.store
        if isinstance(key, tuple):  # X[states, actions]: one action per state
            return (self.table[store.ids[key]] + store.cells[key[0]][:, store.slot_dim]).astype(store.idx_dtype)
        # X[states]: all actions
        return (self.table[store.ids[key]] + store.cells[key][:, None, store.slot_dim]).astype(store.idx_dtype)


class _Probs:
    '''Read-only view of box_probs: X[states] (shape [B, A, prod(slots)]) or X[states, actions].'''

    def __init__(self, store):
        self.store = store

    def __getitem__(self, key):
        return self.store.probs[self.store.ids[key]]
