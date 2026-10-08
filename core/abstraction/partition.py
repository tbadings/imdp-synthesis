import itertools
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

EPS = 1e-3

logger = logging.getLogger(__name__)


@jax.jit
def meshgrid_jax(points, size):
    '''
    Set rectangular grid.

    :param points: Center points per dimension (list or arrays).
    :param size: Number of cells per dimension (list of ints).
    :return: Grid as 2D array.
    '''

    meshgrid = jnp.asarray(jnp.meshgrid(*points, indexing='ij'))
    grid = jnp.reshape(meshgrid, (len(size), -1)).T

    return grid


def define_grid_jax(low, high, size):
    '''
    Define a grid of the specified size, covering the box [low,high].

    :param low: Lower bound of the box to cover (array).
    :param high: Upper bound of the box to cover (array).
    :param size: Number of cells per dimension (list of ints).
    :return:
    '''
    points = [np.linspace(low[i], high[i], size[i]) for i in range(len(size))]
    grid = meshgrid_jax(points, size)

    return grid


@jax.jit
def _cells_contained_in_any_box(cell_lbs, cell_ubs, box_lbs, box_ubs):
    """Return which cells are fully contained in at least one axis-aligned box."""
    # XLA fuses these broadcasts with both reductions, so the logical (cells, boxes, dimensions)
    # array is not materialized. This avoids the sequential fori_loop overhead, especially for the
    # common single-goal case, while still doing one pass over the cell bounds.
    contained_per_box = jnp.all(
        (cell_lbs[:, None, :] >= box_lbs[None, :, :])
        & (cell_ubs[:, None, :] <= box_ubs[None, :, :]),
        axis=2,
    )
    return jnp.any(contained_per_box, axis=1)


@jax.jit
def _cells_overlapping_any_box(cell_lbs, cell_ubs, box_lbs, box_ubs):
    """Return which cells overlap at least one axis-aligned box."""
    overlapping_per_box = jnp.all(
        (cell_ubs[:, None, :] >= box_lbs[None, :, :])
        & (cell_lbs[:, None, :] <= box_ubs[None, :, :]),
        axis=2,
    )
    return jnp.any(overlapping_per_box, axis=1)


def _compute_linear_strides(number_per_dim):
    # Row-major strides to map an n-D grid index [i0, ..., in-1] to a unique 1-D key.
    number_per_dim = np.asarray(number_per_dim, dtype=np.int64)
    strides = np.ones_like(number_per_dim, dtype=np.int64)
    if len(number_per_dim) > 1:
        strides[:-1] = np.cumprod(number_per_dim[1:][::-1], dtype=np.int64)[::-1]
    return strides


def _linear_key_dtype(number_per_dim):
    '''
    Integer dtype able to hold every linear key of this grid. The key indexes the *nominal* grid, so it
    scales with prod(number_per_dim) however few cells are actually kept: a sparse partition over a
    fine grid still needs the wide dtype. int32 keys that silently wrap collide two distinct cells onto
    one key, which searchsorted then resolves to the wrong state without erroring.
    '''
    max_key = int(np.prod(np.asarray(number_per_dim, dtype=np.int64))) - 1
    grid = 'x'.join(str(n) for n in np.asarray(number_per_dim).tolist())
    if max_key <= np.iinfo(np.int32).max:
        logger.info(f'- Partition cell keys: int32 (dense grid {grid} = {max_key + 1:,} cells)')
        return np.int32

    # JAX truncates int64 to int32 unless it is allowed to keep it (see core/jax_config.py), and only
    # warns while doing so. Fail loudly rather than build an index that quietly aliases cells.
    if jnp.zeros(1, dtype=jnp.int64).dtype != jnp.int64:
        raise ValueError(
            f"Grid {np.asarray(number_per_dim).tolist()} needs int64 cell keys (max key {max_key} exceeds "
            f"int32), but JAX is configured to truncate them to int32. Call core.jax_config."
            f"configure_jax first, or set jax.config.update('jax_explicit_x64_dtypes', 'allow')."
        )
    logger.info(f'- Partition cell keys: int64 (dense grid {grid} = {max_key + 1:,} cells, exceeds int32)')
    return np.int64


def _build_sparse_region_index(centers):
    centers_np = np.asarray(centers, dtype=np.int64)
    # Lookup table for Python-side indexing (x2state / grid_idx2state) without allocating dense tensors.
    region_idx_dict = {tuple(c.tolist()): i for i, c in enumerate(centers_np)}
    return region_idx_dict


def _compute_goal_regions(goal_regions, lower_bounds, upper_bounds, region_idxs, size):
    '''Boolean mask + index list of partition cells fully contained in any goal region.'''
    t = time.time()
    if len(goal_regions) > 0:
        goals = jnp.asarray(goal_regions, dtype=lower_bounds.dtype)
        # The previous vertex/halfspace check expanded the full goal width by EPS, which moves
        # each face outward by EPS / 2. Comparing cell bounds directly is equivalent for boxes and
        # avoids materialising all 2**dimension vertices of every cell.
        goal_regions_bools = _cells_contained_in_any_box(
            lower_bounds,
            upper_bounds,
            goals[:, 0, :] - EPS / 2,
            goals[:, 1, :] + EPS / 2,
        )
        goal_regions_idxs = region_idxs[goal_regions_bools]
    else:
        goal_regions_bools = jnp.full(size, False, dtype=bool)
        goal_regions_idxs = jnp.array([], dtype=int)
    logger.debug(f'- Goal regions defined (took {(time.time() - t):.3f} sec.)')

    goal = {
        'bools': goal_regions_bools,
        'idxs': goal_regions_idxs.tolist(),  # TODO: Set should be more efficient here
    }
    logger.debug(f"-- Number of goal regions: {len(goal['idxs'])}")
    return goal


def _compute_critical_regions(critical_regions, lower_bounds, upper_bounds, region_idxs, size):
    '''Boolean mask + index list of partition cells overlapping any critical (unsafe) region.'''
    t = time.time()
    if len(critical_regions) > 0:
        critical = jnp.asarray(critical_regions, dtype=lower_bounds.dtype)
        # Preserve the previous unsafe-set tolerance: shrink every critical box by EPS on each face
        # and mark partition cells whose closed bounds still overlap it.
        critical_regions_bools = _cells_overlapping_any_box(
            lower_bounds,
            upper_bounds,
            critical[:, 0, :] + EPS,
            critical[:, 1, :] - EPS,
        )
        critical_regions_idxs = region_idxs[critical_regions_bools]
    else:
        critical_regions_bools = jnp.full(size, False, dtype=bool)
        critical_regions_idxs = jnp.array([], dtype=int)
    logger.debug(f'- Critical regions defined (took {(time.time() - t):.3f} sec.)')

    critical = {
        'bools': critical_regions_bools,
        'idxs': critical_regions_idxs.tolist(),  # TODO: Set should be more efficient here
    }
    logger.debug(f"-- Number of critical regions: {len(critical['idxs'])}")
    return critical


class _HyperrectangularPartition(object):
    """
    Base class for a partitioning of a state space into hyperrectangular regions (cells).

    Each cell is defined by its center and bounds. Subclasses only decide *which* cells
    are kept and *what* action vectors each cell carries (via `_cells_and_actions`); the index maps,
    bounds, goal/critical detection, and coordinate lookup are all common and built here.
    """

    # Set by each subclass: True for the dense grid, False for the RL-pruned sparse grid. Read by
    # downstream code (forward_reachability / successor_ids / probability_intervals) to pick fast paths.
    rectangular: bool = None

    def __init__(self, model):
        t_total = time.time()

        self.dimension = model.n

        # Retrieve necessary data from the model object
        self.number_per_dim = model.partition['number_per_dim']
        partition_boundary = model.partition['boundary']
        self.boundary_lb = partition_boundary[0]
        self.boundary_ub = partition_boundary[1]

        # From the partition boundary, determine where the first grid centers are placed
        self.cell_width = (partition_boundary[1] - partition_boundary[0]) / self.number_per_dim
        lb_center = partition_boundary[0] + self.cell_width * 0.5

        # Subclass-specific: unit-cube grid indices of the kept cells and their action vectors.
        # actions is (1, num_actions, dim) for the dense grid (broadcast downstream) or
        # (num_states, num_actions, dim) for the sparse grid.
        t = time.time()
        centers_unit, actions = self._cells_and_actions(model)
        centers = jnp.array(centers_unit, dtype=int)

        # Build the Python-side lookup table (used by x2state / grid_idx2state).
        self.region_idx_dict = _build_sparse_region_index(centers)

        # JAX-friendly sparse index map via linearized coordinates and searchsorted.
        # We store sorted linear keys and aligned state IDs so lookup can stay inside JIT.
        idx_np_dtype = _linear_key_dtype(self.number_per_dim)
        idx_jnp_dtype = jnp.int64 if idx_np_dtype == np.int64 else jnp.int32
        region_linear_strides = _compute_linear_strides(self.number_per_dim).astype(idx_np_dtype)
        centers_np = np.asarray(centers, dtype=idx_np_dtype)
        # Linear key for each kept cell index tuple.
        region_linear_idx = np.sum(centers_np * region_linear_strides, axis=1, dtype=idx_np_dtype)
        order = np.argsort(region_linear_idx)

        # Sorted keys -> sorted state IDs (parallel arrays used by jnp.searchsorted). Keys must go
        # through jnp.asarray: jnp.array() canonicalizes an int64 host array to int32 *before* applying
        # dtype, so it would hand back int64-typed keys whose values have already wrapped.
        self.region_linear_idx = jnp.asarray(region_linear_idx[order], dtype=idx_jnp_dtype)
        self.region_linear_state = jnp.asarray(np.arange(len(centers_np), dtype=np.int32)[order], dtype=jnp.int32)
        self.region_linear_strides = jnp.asarray(region_linear_strides, dtype=idx_jnp_dtype)

        # Unit-cube index tuple for each state (used to map a state ID back to grid coordinates).
        self.region_idx_inv = centers

        # Now scale the unit-cube partition to physical coordinates.
        centers = centers_unit * self.cell_width + lb_center

        region_idxs = jnp.arange(len(centers))
        lower_bounds = centers - self.cell_width / 2
        upper_bounds = centers + self.cell_width / 2

        logger.debug(f'- Grid cells defined (took {(time.time() - t):.3f} sec.)')

        self.regions = {
            'centers': jnp.array(centers, dtype=float),
            'idxs': region_idxs,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'actions': actions,
        }
        self.size = len(centers)

        # Sentinel state id used by JAX kernels for coordinates that are not present.
        self.missing_state = int(self.size)

        # Also store the partition bounds per dimension
        elems_per_dim = [jnp.arange(num) for num in self.number_per_dim]
        centers_per_dim = [elems_per_dim[i] * self.cell_width[i] + lb_center[i] for i in range(self.dimension)]
        lower_bounds_per_dim = [jnp.array(centers_per_dim[i] - self.cell_width[i] / 2) for i in range(self.dimension)]
        upper_bounds_per_dim = [jnp.array(centers_per_dim[i] + self.cell_width[i] / 2) for i in range(self.dimension)]

        self.regions_per_dim = {
            'centers': centers_per_dim,
            'idxs': elems_per_dim,
            'lower_bounds': lower_bounds_per_dim,
            'upper_bounds': upper_bounds_per_dim,
        }

        self.goal = _compute_goal_regions(
            model.goal, self.regions['lower_bounds'], self.regions['upper_bounds'], region_idxs, self.size
        )
        self.critical = _compute_critical_regions(
            model.critical, self.regions['lower_bounds'], self.regions['upper_bounds'], region_idxs, self.size
        )

        logger.debug(f'Partitioning took {(time.time() - t_total):.3f} sec.')

        logger.info(f"(Number of states: {len(self.regions['idxs'])})")
        logger.info(f"(Number of actions: {actions.shape[1]})")

    def _cells_and_actions(self, model):
        '''Return (centers_unit, actions): the kept unit-cube grid indices and their action vectors.'''
        raise NotImplementedError

    def x2state(self, x):
        '''
        Return the state ID for a given point x in the continuous state space.

        :param x: Point in the continuous state space.
        :return: (state ID, exists). exists is False (and the ID is the missing-state sentinel) when x
            is outside the partition boundary or falls in a cell absent from the (sparse) partition.
        '''
        # Discard points outside of partition
        in_partition = np.all((x >= self.boundary_lb) * (x <= self.boundary_ub))
        if not in_partition:
            return self.size, False

        # Normalize points to unit-cube grid indices
        x_norm = np.array(((x - self.boundary_lb) / (self.boundary_ub - self.boundary_lb) * self.number_per_dim) // 1, dtype=int)
        return self.grid_idx2state(x_norm)

    def grid_idx2state(self, grid_idx):
        '''Map a unit-cube grid index tuple to its (state ID, exists).'''
        state = self.region_idx_dict.get(tuple(np.asarray(grid_idx, dtype=int).tolist()), -1)
        if state == -1:
            return self.size, False
        return int(state), True


class DensePartition(_HyperrectangularPartition):
    """Dense partition: every cell of the nominal grid is kept, all sharing one action set."""

    rectangular = True

    def __init__(self, model, verbose=False):
        t = time.time()
        logger.info('Define dense partition...')
        super().__init__(model)

        logger.info(f"Time to build dense partition: %.3f seconds" % (time.time() - t))
        print('')

    def _cells_and_actions(self, model):
        # Full dense grid where each region is a unit cube.
        lb_unit = jnp.zeros(self.dimension, dtype=int)
        ub_unit = jnp.array(self.number_per_dim - 1, dtype=int)
        centers_unit = define_grid_jax(lb_unit, ub_unit, self.number_per_dim)

        # Discrete action grid: Cartesian product of the per-dimension action grids.
        discrete_actions_per_dimension = [
            np.linspace(model.uMin[i], model.uMax[i], num=model.num_actions[i])
            for i in range(len(model.num_actions))
        ]
        action_array = jnp.array(list(itertools.product(*discrete_actions_per_dimension)))
        # Store as (1, num_actions, action_dim) — broadcast in forward_reachability to avoid
        # materializing a (num_states, num_actions, action_dim) array for large rectangular partitions.
        actions = action_array[None, :, :].astype(float)
        return centers_unit, actions


class SparsePartition(_HyperrectangularPartition):
    """Sparse partition: only the RL-explored active cells are kept, each with its own action set."""

    rectangular = False

    def __init__(self, model, active_states, active_actions, verbose=False):
        '''
        :param model: Benchmark model
        :param active_states: Grid cells of the active states, shape [num_states, state_dim]
        :param active_actions: Discrete actions of the active states, shape [num_states, num_actions, input_dim],
            where row i holds the actions of active_states[i] (see core.rl.find_active). Kept, without a copy,
            as regions['actions'].
        '''
        t = time.time()
        logger.info('=== Define sparse partition ===')
        self._active_states = active_states
        self._active_actions = active_actions
        super().__init__(model)
        # Only needed to build the partition (the actions live on as regions['actions'])
        del self._active_states, self._active_actions

        logger.info(f"Time to build sparse partition: %.3f seconds" % (time.time() - t))
        print('')

    def _cells_and_actions(self, model):
        # The sparse partition is defined directly by the RL-explored active states.
        centers_unit = jnp.array(self._active_states, dtype=int)

        # Every state has the same number of actions; row i belongs to state i. The array is used as is: a
        # copy would double its memory (states x actions x input dim).
        actions = self._active_actions
        assert actions.ndim == 3 and len(actions) == len(centers_unit), (
            f"Expected actions of shape [{len(centers_unit)}, num_actions, input_dim], got {actions.shape}.")
        return centers_unit, actions
