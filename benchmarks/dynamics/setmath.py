import jax
import jax.numpy as jnp


@jax.jit
def box(x_min, x_max):
    '''
    Generate the box [x_min, x_max].

    :param x_min:
    :param x_max:
    :return: Box.
    '''
    # Check if x_min < x_max and if not, flip around.
    return jnp.minimum(x_min, x_max), jnp.maximum(x_min, x_max)

@jax.jit
def tuple2box(values):
    '''
    Compute the min and max of a set of values.

    :param values:
    :return: min and max of the values.
    '''
    return jnp.array([jnp.minimum(values[0], values[1]), jnp.maximum(values[0], values[1])])


@jax.jit
def box2vertices(x_min, x_max):
    '''
    Convert box [x_min, x_max] in R^n to its 2^n vertices and return in an array of shape (2^n,n).
    :param x_min:
    :param x_max:
    :return: Vertices of the box.
    '''
    
    x_min, x_max = box(x_min, x_max)
    n = x_min.shape[0]
    num = 1 << n
    idx = jnp.arange(num, dtype=jnp.uint32)
    shifts = jnp.arange(n, dtype=jnp.uint32)
    mask = ((idx[:, None] >> shifts[None, :]) & jnp.uint32(1)).astype(bool)
    return jnp.where(mask, x_max, x_min)


@jax.jit
def mult(X, Y):
    '''
    Multiply two intervals X and Y.

    :param X:
    :param Y:
    :return: X*Y (sorted)
    '''
    # Multiply two boxes with each other
    x_min, x_max = X
    y_min, y_max = Y

    x_min, x_max = box(x_min, x_max)
    y_min, y_max = box(y_min, y_max)

    # Multiply all combinations
    x_min_y_min = x_min * y_min
    x_max_y_min = x_max * y_min
    x_min_y_max = x_min * y_max
    x_max_y_max = x_max * y_max

    Z = jnp.vstack((x_min_y_min, x_max_y_min, x_min_y_max, x_max_y_max))
    z_min = jnp.min(Z, axis=0)
    z_max = jnp.max(Z, axis=0)

    return z_min, z_max


@jax.jit
def sin(x_min, x_max):
    '''
    Compute min/max of sin(x) over the interval [x_min, x_max].

    :param x_min:
    :param x_max:
    :return: min/max of sin(x).
    '''
    x_min, x_max = box(x_min, x_max)

    # Shift such that x_min is always in [0,2*pi]
    mod = x_min // (jnp.pi * 2)
    x_min -= mod * 2 * jnp.pi
    x_max -= mod * 2 * jnp.pi

    # If (0.5+2k)*pi, for any k, is in the interval, then the maximum is 1
    # This is the case if x_min < pi/2 and x_max > pi/2, or (because we know x_min <= 2pi) if x_max > 3pi
    Q = (x_min < 0.5 * jnp.pi) * (x_max > 0.5 * jnp.pi) + (x_max > 2.5 * jnp.pi)
    y_max = 1 * Q + jnp.maximum(jnp.sin(x_min), jnp.sin(x_max)) * ~Q

    # If (1.5+2k)*pi is in the interval, then the minimum is -1
    # This is the case if x_min < 3pi/2 and x_max > 3pi/2, or (because we know x_min <= 2pi) if x_max > 7pi/2
    Q = (x_min < 1.5 * jnp.pi) * (x_max > 1.5 * jnp.pi) + (x_max > 3.5 * jnp.pi)
    y_min = -1 * Q + jnp.minimum(jnp.sin(x_min), jnp.sin(x_max)) * ~Q

    return y_min, y_max


@jax.jit
def cos(x_min, x_max):
    '''
    Compute min/max of cos(x) over the interval [x_min, x_max].

    :param x_min:
    :param x_max:
    :return: min/max of cos(x).
    '''
    x_min, x_max = box(x_min, x_max)

    # Shift such that x_min is always in [0,2*pi]
    mod = x_min // (jnp.pi * 2)
    x_min -= mod * 2 * jnp.pi
    x_max -= mod * 2 * jnp.pi

    # If (1+2k)*pi, for any k, is in the interval, then the minimum is -1
    # This is the case if x_min < pi and x_max > pi, or (because we know x_min <= 2pi) if x_max > 3pi
    Q = (x_min < jnp.pi) * (x_max > jnp.pi) + (x_max > 3 * jnp.pi)
    y_min = -1 * Q + jnp.minimum(jnp.cos(x_min), jnp.cos(x_max)) * ~Q

    # If 2*pi is in the interval, then the maximum is 1
    Q = (x_max > 2 * jnp.pi)
    y_max = 1 * Q + jnp.maximum(jnp.cos(x_min), jnp.cos(x_max)) * ~Q

    return y_min, y_max
