import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

from .polytope import hyperrectangles_isdisjoint_multi

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
def center2halfspace(center, cell_width):
    '''
    From given centers and cell widths, compute the halfspace inequalities Ax <= b.

    :param center:
    :param cell_width:
    :return:
    '''

    A1 = jnp.identity(len(center))
    A2 = -jnp.identity(len(center))

    b1 = center + cell_width / 2
    b2 = -(center - cell_width / 2)

    A = jnp.concatenate((A1, A2))
    b = jnp.concatenate((b1, b2))

    return A, b


# Vectorized function over different polytopes
from .polytope import points_in_polytope

vmap_points_in_polytope = jax.jit(jax.vmap(points_in_polytope, in_axes=(0, 0, None), out_axes=0))

from .polytope import any_points_in_polytope

vmap_any_points_in_polytope = jax.jit(jax.vmap(any_points_in_polytope, in_axes=(0, 0, None), out_axes=0))


@jax.jit
def check_if_region_in_goal(goals_A, goals_b, points):
    # Vectorized over all goal regions
    points_contained = vmap_points_in_polytope(goals_A, goals_b, points)

    # Check for every goal region if all points are contained in the polytope
    all_points_contained = jnp.all(points_contained, axis=1)

    # If any goal region is contained in the polytope, then set current polytope as goal
    return jnp.any(all_points_contained)


# Vectorized function over different sets of points
vmap_check_if_region_in_goal = jax.jit(jax.vmap(check_if_region_in_goal, in_axes=(None, None, 0), out_axes=0))


@jax.jit
def get_vertices_from_bounds(lb, ub):
    # Stack lower and upper bounds in one array
    stacked = jnp.vstack((lb, ub))

    # Get all vertices (by taking combinations of lower and upper bounds)
    vertices = meshgrid_jax(stacked.T, lb)

    return vertices


def _compute_linear_strides(number_per_dim):
    # Row-major strides to map an n-D grid index [i0, ..., in-1] to a unique 1-D key.
    number_per_dim = np.asarray(number_per_dim, dtype=np.int64)
    strides = np.ones_like(number_per_dim, dtype=np.int64)
    if len(number_per_dim) > 1:
        strides[:-1] = np.cumprod(number_per_dim[1:][::-1], dtype=np.int64)[::-1]
    return strides


def _build_sparse_region_index(centers):
    centers_np = np.asarray(centers, dtype=np.int64)
    # Membership and lookup tables for Python-side indexing without allocating dense tensors.
    region_idx_set = {tuple(c.tolist()) for c in centers_np}
    region_idx_dict = {tuple(c.tolist()): i for i, c in enumerate(centers_np)}
    return region_idx_set, region_idx_dict


class RectangularPartition(object):
    """
    Represents a rectangular partitioning of a state space into hyperrectangular regions.

    This class is used to define and manage a partition of a state space, where the space is divided
    into hyperrectangular regions (or cells). Each cell is defined by its center, bounds, and vertices,
    and the entirety of these regions form a structured grid within the state space.
    """

    def __init__(self, model, verbose=False):
        logger.info('Define rectangular partition...')
        t_total = time.time()

        self.dimension = model.n

        # Retrieve necessary data from the model object
        self.number_per_dim = model.partition['number_per_dim']
        partition_boundary = model.partition['boundary']
        self.boundary_lb = partition_boundary[0]
        self.boundary_ub = partition_boundary[1]
        goal_regions = model.goal
        critical_regions = model.critical

        # Set partition as being (hyper)rectangular and nonsparse
        self.rectangular = True

        t = time.time()
        # From the partition boundary, determine where the first grid centers are placed
        self.cell_width = (partition_boundary[1] - partition_boundary[0]) / self.number_per_dim
        lb_center = partition_boundary[0] + self.cell_width * 0.5
        ub_center = partition_boundary[1] - self.cell_width * 0.5

        # First define a grid where each region is a unit cube
        lb_unit = jnp.zeros(len(lb_center), dtype=int)
        ub_unit = jnp.array(self.number_per_dim - 1, dtype=int)
        centers_unit = define_grid_jax(lb_unit, ub_unit, self.number_per_dim)

        # Build sparse index structures for state lookup.
        centers = jnp.array(centers_unit, dtype=int)
        self.region_idx_set, self.region_idx_dict = _build_sparse_region_index(centers)

        # JAX-friendly sparse index map via linearized coordinates and searchsorted.
        # We store sorted linear keys and aligned state IDs so lookup can stay inside JIT.
        idx_np_dtype = np.int64 if jax.config.read('jax_enable_x64') else np.int32
        idx_jnp_dtype = jnp.int64 if jax.config.read('jax_enable_x64') else jnp.int32
        region_linear_strides = _compute_linear_strides(self.number_per_dim).astype(idx_np_dtype)
        centers_np = np.asarray(centers, dtype=idx_np_dtype)
        # Linear key for each kept cell index tuple.
        region_linear_idx = np.sum(centers_np * region_linear_strides, axis=1, dtype=idx_np_dtype)
        order = np.argsort(region_linear_idx)
        
        # Sorted keys -> sorted state IDs (parallel arrays used by jnp.searchsorted).
        self.region_linear_idx = jnp.array(region_linear_idx[order], dtype=idx_jnp_dtype)
        self.region_linear_state = jnp.array(np.arange(len(centers_np), dtype=np.int32)[order], dtype=jnp.int32)
        self.region_linear_strides = jnp.array(region_linear_strides, dtype=idx_jnp_dtype)
        
        # Define list with each element containing its index elements
        self.region_idx_inv = centers

        # Now scale the unit-cube partition appropriately
        centers = centers_unit * self.cell_width + lb_center

        region_idxs = jnp.arange(len(centers))
        lower_bounds = centers - self.cell_width / 2
        upper_bounds = centers + self.cell_width / 2

        # Determine the vertices of all partition elements
        vmap_get_vertices_from_bounds = jax.jit(jax.vmap(get_vertices_from_bounds, in_axes=(0, 0), out_axes=0))
        all_vertices = vmap_get_vertices_from_bounds(lower_bounds, upper_bounds)
        logger.debug(f'- Grid points defined (took {(time.time() - t):.3f} sec.)')

        t = time.time()
        # Determine halfspace (Ax <= b) inequalities
        vmap_center2halfspace = jax.jit(jax.vmap(center2halfspace, in_axes=(0, None), out_axes=(0, 0)))
        all_A, all_b = vmap_center2halfspace(centers, self.cell_width)
        logger.debug(f'- Halfspace inequalities (Ax <= b) defined (took {(time.time() - t):.3f} sec.)')

        self.regions = {
            'centers': jnp.array(centers, dtype=float),
            'idxs': region_idxs,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'all_vertices': all_vertices,
            'A': all_A,
            'b': all_b
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

        t = time.time()
        if len(goal_regions) > 0:
            # Compute halfspace representation of the goal regions
            goal_centers = np.zeros((len(goal_regions), len(self.number_per_dim)))
            goal_widths = np.zeros((len(goal_regions), len(self.number_per_dim)))
            for i, goal in enumerate(goal_regions):
                goal_centers[i] = (goal[1] + goal[0]) / 2
                goal_widths[i] = (goal[1] - goal[0]) + EPS

            goal_centers = jnp.array(goal_centers, dtype=float)
            goal_widths = jnp.array(goal_widths, dtype=float)

            vmap_center2halfspace = jax.jit(jax.vmap(center2halfspace, in_axes=(0, 0), out_axes=(0, 0)))
            goals_A, goals_b = vmap_center2halfspace(goal_centers, goal_widths)

            # Determine goal regions
            goal_regions_bools = vmap_check_if_region_in_goal(goals_A, goals_b, all_vertices)
            goal_regions_idxs = region_idxs[goal_regions_bools]
        else:
            goal_regions_bools = jnp.full(self.size, False, dtype=bool)
            goal_regions_idxs = jnp.array([], dtype=int)
        logger.debug(f'- Goal regions defined (took {(time.time() - t):.3f} sec.)')

        self.goal = {
            'bools': goal_regions_bools,
            'idxs': goal_regions_idxs.tolist(), # TODO: Set should be more efficient here
        }
        logger.debug(f"-- Number of goal regions: {len(self.goal['idxs'])}")

        t = time.time()
        if len(critical_regions) > 0:
            # Check which regions (hyperrectangles) are *not* disjoint from the critical regions (also hyperrectangles)
            critical_lbs = critical_regions[:, 0, :]
            critical_ubs = critical_regions[:, 1, :]

            vfun = jax.jit(jax.vmap(hyperrectangles_isdisjoint_multi, in_axes=(0, 0, None, None), out_axes=0))
            critical_regions_bools = ~vfun(self.regions['lower_bounds'], self.regions['upper_bounds'],
                                           critical_lbs + EPS, critical_ubs - EPS)
            critical_regions_idxs = region_idxs[critical_regions_bools]
        else:
            critical_regions_bools = jnp.full(self.size, False, dtype=bool)
            critical_regions_idxs = jnp.array([], dtype=int)
        logger.debug(f'- Critical regions defined (took {(time.time() - t):.3f} sec.)')

        self.critical = {
            'bools': critical_regions_bools,
            'idxs': critical_regions_idxs.tolist(), # TODO: Set should be more efficient here
        }
        logger.debug(f"-- Number of critical regions: {len(self.critical['idxs'])}")

        logger.debug(f'Partitioning took {(time.time() - t_total):.3f} sec.')

        logger.info(f"(Number of states: {len(self.regions['idxs'])})")
        return

    def x2state(self, x):
        '''
        Return the state ID for a given point x in the continuous state space.

        :param x: Point in the continuous state space.
        :return: State ID.
        '''
        # Discard points outside of partition
        in_partition = np.all((x >= self.boundary_lb) * (x <= self.boundary_ub))

        if in_partition:
            # Normalize points
            x_norm = np.array(((x - self.boundary_lb) / (self.boundary_ub - self.boundary_lb) * self.number_per_dim) // 1, dtype=int)
            state = self.region_idx_dict.get(tuple(x_norm.tolist()), -1)
            if state == -1:
                return self.size, False
            return int(state), True

        else:
            return self.size, False

    def grid_idx2state(self, grid_idx):
        state = self.region_idx_dict.get(tuple(np.asarray(grid_idx, dtype=int).tolist()), -1)
        if state == -1:
            return self.size, False
        return int(state), True


class SparsePartition(object):

    """
    Represents a rectangular partitioning of a state space into hyperrectangular regions.

    This class is used to define and manage a partition of a state space, where the space is divided
    into hyperrectangular regions (or cells). Each cell is defined by its center, bounds, and vertices,
    and the entirety of these regions form a structured grid within the state space.
    """

    def __init__(self, model, remove_cells=10, verbose=False):
        logger.info('Define non-rectangular (sparse) partition...')
        t_total = time.time()

        self.dimension = model.n

        # Retrieve necessary data from the model object
        self.number_per_dim = model.partition['number_per_dim']
        partition_boundary = model.partition['boundary']
        self.boundary_lb = partition_boundary[0]
        self.boundary_ub = partition_boundary[1]
        goal_regions = model.goal
        critical_regions = model.critical

        # Set partition as being nonsparse
        self.rectangular = False

        t = time.time()
        # From the partition boundary, determine where the first grid centers are placed
        self.cell_width = (partition_boundary[1] - partition_boundary[0]) / self.number_per_dim
        lb_center = partition_boundary[0] + self.cell_width * 0.5
        ub_center = partition_boundary[1] - self.cell_width * 0.5

        # First define a grid where each region is a unit cube
        lb_unit = jnp.zeros(len(lb_center), dtype=int)
        ub_unit = jnp.array(self.number_per_dim - 1, dtype=int)
        centers_unit = define_grid_jax(lb_unit, ub_unit, self.number_per_dim)

        # ====== #
        # Remove cells to make partition sparse.
        # If model-provided exclusion boxes are present, remove all cells whose centers
        # lie in those boxes. Otherwise, fall back to random cell removal.
        centers_unit = np.array(centers_unit)
        sparse_exclude = model.partition.get('sparse_exclude', None)
        if sparse_exclude is not None and len(sparse_exclude) > 0:
            sparse_exclude = np.asarray(sparse_exclude, dtype=float)
            centers_cont = centers_unit * np.asarray(self.cell_width) + np.asarray(lb_center)
            keep_mask = np.ones(len(centers_unit), dtype=bool)

            for box in sparse_exclude:
                lb = box[0]
                ub = box[1]
                in_box = np.all((centers_cont >= lb) & (centers_cont <= ub), axis=1)
                keep_mask &= ~in_box

            centers_unit = jnp.array(centers_unit[keep_mask])
        else:
            remove_cells_eff = min(remove_cells, len(centers_unit))
            remove_idxs = np.random.choice(len(centers_unit), size=remove_cells_eff, replace=False)
            centers_unit = jnp.array(np.delete(centers_unit, remove_idxs, axis=0))

        # ====== #

        # Build sparse index structures for state lookup.
        centers = jnp.array(centers_unit, dtype=int)
        self.region_idx_set, self.region_idx_dict = _build_sparse_region_index(centers)

        # JAX-friendly sparse index map via linearized coordinates and searchsorted.
        # We store sorted linear keys and aligned state IDs so lookup can stay inside JIT.
        idx_np_dtype = np.int64 if jax.config.read('jax_enable_x64') else np.int32
        idx_jnp_dtype = jnp.int64 if jax.config.read('jax_enable_x64') else jnp.int32
        region_linear_strides = _compute_linear_strides(self.number_per_dim).astype(idx_np_dtype)
        centers_np = np.asarray(centers, dtype=idx_np_dtype)
        
        # Linear key for each kept cell index tuple.
        region_linear_idx = np.sum(centers_np * region_linear_strides, axis=1, dtype=idx_np_dtype)
        order = np.argsort(region_linear_idx)
        
        # Sorted keys -> sorted state IDs (parallel arrays used by jnp.searchsorted).
        self.region_linear_idx = jnp.array(region_linear_idx[order], dtype=idx_jnp_dtype)
        self.region_linear_state = jnp.array(np.arange(len(centers_np), dtype=np.int32)[order], dtype=jnp.int32)
        self.region_linear_strides = jnp.array(region_linear_strides, dtype=idx_jnp_dtype)
        # Define list with each element containing its index elements
        self.region_idx_inv = centers

        # Now scale the unit-cube partition appropriately
        centers = centers_unit * self.cell_width + lb_center

        region_idxs = jnp.arange(len(centers))
        lower_bounds = centers - self.cell_width / 2
        upper_bounds = centers + self.cell_width / 2

        # Determine the vertices of all partition elements
        vmap_get_vertices_from_bounds = jax.jit(jax.vmap(get_vertices_from_bounds, in_axes=(0, 0), out_axes=0))
        all_vertices = vmap_get_vertices_from_bounds(lower_bounds, upper_bounds)
        logger.debug(f'- Grid points defined (took {(time.time() - t):.3f} sec.)')

        t = time.time()
        # Determine halfspace (Ax <= b) inequalities
        vmap_center2halfspace = jax.jit(jax.vmap(center2halfspace, in_axes=(0, None), out_axes=(0, 0)))
        all_A, all_b = vmap_center2halfspace(centers, self.cell_width)
        logger.debug(f'- Halfspace inequalities (Ax <= b) defined (took {(time.time() - t):.3f} sec.)')

        self.regions = {
            'centers': jnp.array(centers, dtype=float),
            'idxs': region_idxs,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'all_vertices': all_vertices,
            'A': all_A,
            'b': all_b
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

        t = time.time()
        if len(goal_regions) > 0:
            # Compute halfspace representation of the goal regions
            goal_centers = np.zeros((len(goal_regions), len(self.number_per_dim)))
            goal_widths = np.zeros((len(goal_regions), len(self.number_per_dim)))
            for i, goal in enumerate(goal_regions):
                goal_centers[i] = (goal[1] + goal[0]) / 2
                goal_widths[i] = (goal[1] - goal[0]) + EPS

            goal_centers = jnp.array(goal_centers, dtype=float)
            goal_widths = jnp.array(goal_widths, dtype=float)

            vmap_center2halfspace = jax.jit(jax.vmap(center2halfspace, in_axes=(0, 0), out_axes=(0, 0)))
            goals_A, goals_b = vmap_center2halfspace(goal_centers, goal_widths)

            # Determine goal regions
            goal_regions_bools = vmap_check_if_region_in_goal(goals_A, goals_b, all_vertices)
            goal_regions_idxs = region_idxs[goal_regions_bools]
        else:
            goal_regions_bools = jnp.full(self.size, False, dtype=bool)
            goal_regions_idxs = jnp.array([], dtype=int)
        logger.debug(f'- Goal regions defined (took {(time.time() - t):.3f} sec.)')

        self.goal = {
            'bools': goal_regions_bools,
            'idxs': goal_regions_idxs.tolist(), # TODO: Set should be more efficient here
        }
        logger.debug(f"-- Number of goal regions: {len(self.goal['idxs'])}")

        t = time.time()
        if len(critical_regions) > 0:
            # Check which regions (hyperrectangles) are *not* disjoint from the critical regions (also hyperrectangles)
            critical_lbs = critical_regions[:, 0, :]
            critical_ubs = critical_regions[:, 1, :]

            vfun = jax.jit(jax.vmap(hyperrectangles_isdisjoint_multi, in_axes=(0, 0, None, None), out_axes=0))
            critical_regions_bools = ~vfun(self.regions['lower_bounds'], self.regions['upper_bounds'],
                                           critical_lbs + EPS, critical_ubs - EPS)
            critical_regions_idxs = region_idxs[critical_regions_bools]
        else:
            critical_regions_bools = jnp.full(self.size, False, dtype=bool)
            critical_regions_idxs = jnp.array([], dtype=int)
        logger.debug(f'- Critical regions defined (took {(time.time() - t):.3f} sec.)')

        self.critical = {
            'bools': critical_regions_bools,
            'idxs': critical_regions_idxs.tolist(), # TODO: Set should be more efficient here
        }
        logger.debug(f"-- Number of critical regions: {len(self.critical['idxs'])}")

        logger.debug(f'Partitioning took {(time.time() - t_total):.3f} sec.')

        logger.info(f"(Number of states: {len(self.regions['idxs'])})")
        return

    def x2state(self, x):
        '''
        Return the state ID for a given point x in the continuous state space.

        :param x: Point in the continuous state space.
        :return: State ID.
        '''
        # Discard points outside of partition
        in_partition = np.all((x >= self.boundary_lb) * (x <= self.boundary_ub))

        if in_partition:
            # Normalize points
            x_norm = np.array(((x - self.boundary_lb) / (self.boundary_ub - self.boundary_lb) * self.number_per_dim) // 1, dtype=int)
            state = self.region_idx_dict.get(tuple(x_norm.tolist()), -1)
            # state == -1 means the cell was removed (sparse partition)
            if state == -1:
                return self.size, False
            return int(state), True

        else:
            return self.size, False

    def grid_idx2state(self, grid_idx):
        state = self.region_idx_dict.get(tuple(np.asarray(grid_idx, dtype=int).tolist()), -1)
        if state == -1:
            return self.size, False
        return int(state), True
