import jax
import jax.numpy as jnp


def points_in_polytope(A, b, points):
    ''' Check if polytope defined by Ax <= b contains given list of points '''

    # Check matrix inequality
    bools = (jnp.matmul(A, points.T).T <= b)

    # A point is contained if every constraint is satisfied
    points_contained = jnp.all(bools, axis=1)

    return points_contained


def any_points_in_polytope(A, b, points):
    ''' Check if polytope defined by Ax <= b contains given list of points '''

    # Check matrix inequality
    bools = (jnp.matmul(A, points.T).T <= b)

    # A point is contained if every constraint is satisfied
    points_contained = jnp.min(bools, axis=1)  # jnp.all(bools, axis=1)

    return jnp.max(points_contained)  # jnp.any(points_contained)


def all_points_in_polytope(A, b, points):
    ''' Check if polytope defined by Ax <= b contains given list of points '''

    # Check matrix inequality
    bools = (jnp.matmul(A, points.T).T <= b)

    # A point is contained if every constraint is satisfied
    # points_contained = jnp.all(bools, axis=1)
    # return jnp.all(points_contained)

    return jnp.min(bools)


def num_points_in_polytope(A, b, points):
    ''' Check if polytope defined by Ax <= b contains given list of points '''

    # Check matrix inequality
    bools = (jnp.matmul(A, points.T).T < b)

    # A point is contained if every constraint is satisfied
    points_contained = jnp.min(bools, axis=1)  # jnp.all(bools, axis=1)

    return jnp.sum(points_contained)


def hyperrectangles_isdisjoint(lb1, ub1, lb2, ub2):
    '''
    Check if two hyperrectangles are disjoint.

    Algorithm:
    ``H1 \cap H2 are not disjoint iff |c_2 - c_1| \leq r_1 + r_2, where \leq is taken component-wise.``
    '''

    # Compute both centers
    center1 = (ub1 + lb1) / 2
    center2 = (ub2 + lb2) / 2

    # Compute both radii
    radius1 = (ub1 - lb1) / 2
    radius2 = (ub2 - lb2) / 2

    center_diff = center2 - center1
    empty_intersection = jnp.any(jnp.abs(center_diff) > radius1 + radius2)

    return empty_intersection


vmap_hyperrectangles_isdisjoint = jax.jit(jax.vmap(hyperrectangles_isdisjoint, in_axes=(None, None, 0, 0), out_axes=0))


def hyperrectangles_isdisjoint_multi(lb1, ub1, lbs2, ubs2):
    '''
    Check if one hyperrectangle (lb1, ub1) is disjoint from multiple other hyperrectangles given by (lbs2, ubs2)

    Algorithm:
    ``H1 \cap H2 are not disjoint iff |c_2 - c_1| \leq r_1 + r_2, where \leq is taken component-wise.``
    '''

    empty_intersections = vmap_hyperrectangles_isdisjoint(lb1, ub1, lbs2, ubs2)
    empty_intersection = jnp.all(empty_intersections)

    return empty_intersection
