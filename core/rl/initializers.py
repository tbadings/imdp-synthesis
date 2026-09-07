"""Portable orthogonal initialization for the Dense layers used by PPO.

The versioned sampler uses SHAKE-256 and Marsaglia's polar transform. Decimal
log/sqrt avoid platform libm differences when generating Gaussian samples.
Householder QR uses float64, explicit reduction order, and separate elementary
operations: no BLAS, LAPACK, parallel reductions, or fused multiply-adds.

For the same key data, shape, gain and dtype, this targets identical weights on
IEEE-754 machines (including M3/M5). It preserves Gaussian-QR orthogonal
initialization's distribution and gains, but changes the seed-to-weights mapping.
It does not make subsequent JAX training arithmetic portable or deterministic.
The host callback runs only during parameter initialization and is slower than
native QR. Neither JAX's global precision nor the process RNG state is changed.
"""

from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from functools import partial
import hashlib
import math
import operator
import struct

import jax
import jax.numpy as jnp
import numpy as np


def _uniform_stream(seed):
    counter = 0
    while True:
        block = hashlib.shake_256(seed + struct.pack("<Q", counter)).digest(4096)
        for (bits,) in struct.iter_unpack("<Q", block):
            # An exact dyadic value in [-1, 1), using 53 random bits.
            yield (bits >> 11) * 2.0**-52 - 1.0
        counter += 1


def _normal_matrix(key_data, shape):
    key_bytes = np.asarray(key_data, dtype="<u4").tobytes(order="C")
    seed = b"imdp-deterministic-orthogonal-v1\0" + key_bytes
    uniforms = _uniform_stream(seed)
    values = np.empty(math.prod(shape), dtype=np.float64)
    index = 0
    # Supply a fresh context, so caller changes to decimal precision/traps cannot
    # affect weights. Decimal's ln and sqrt are correctly rounded.
    with localcontext(Context(prec=50, rounding=ROUND_HALF_EVEN)):
        while index < values.size:
            u, v = next(uniforms), next(uniforms)
            radius_squared = u * u + v * v
            if not 0.0 < radius_squared < 1.0:
                continue
            radius_decimal = Decimal.from_float(radius_squared)
            factor = float((-2 * radius_decimal.ln() / radius_decimal).sqrt())
            values[index] = u * factor
            index += 1
            if index < values.size:
                values[index] = v * factor
                index += 1
    return values.reshape(shape)


def _norm(vector):
    total = 0.0
    for value in vector:
        value = float(value)
        total = total + value * value
    return math.sqrt(total)


def _reflect_in_place(matrix, vector):
    # Explicit row order pins each dot product's accumulation. NumPy ufuncs
    # operate elementwise only; separate calls prevent multiply-add fusion.
    projection = np.zeros(matrix.shape[1], dtype=np.float64)
    for row, coefficient in zip(matrix, vector):
        projection += coefficient * row
    matrix -= (2.0 * vector[:, None]) * projection[None, :]


def _fixed_qr(matrix):
    """Reduced Q for a tall matrix, with the positive-R-diagonal convention."""
    work = np.array(matrix, dtype=np.float64, copy=True)
    rows, columns = work.shape
    reflectors = []
    signs = np.ones(columns, dtype=np.float64)
    for column in range(columns):
        vector = work[column:, column].copy()
        length = _norm(vector)
        if length == 0.0:
            reflectors.append(None)
            continue
        diagonal = -math.copysign(length, float(vector[0]))
        vector[0] -= diagonal
        vector /= _norm(vector)
        _reflect_in_place(work[column:, column:], vector)
        signs[column] = math.copysign(1.0, diagonal)
        reflectors.append(vector)

    q = np.eye(rows, columns, dtype=np.float64)
    for column in range(columns - 1, -1, -1):
        vector = reflectors[column]
        if vector is not None:
            _reflect_in_place(q[column:, :], vector)
    return q * signs


def _host_orthogonal(key_data, *, shape, dtype, scale):
    rows, columns = shape
    matrix = _normal_matrix(key_data, (max(rows, columns), min(rows, columns)))
    q = _fixed_qr(matrix)
    if rows < columns:
        q = q.T
    return np.asarray(scale * q, dtype=dtype, order="C")


def deterministic_orthogonal(scale=1.0):
    """Flax kernel initializer for nonempty 2-D float32/float64 Dense weights.

    Columns are orthogonal for tall matrices; rows are orthogonal for wide
    matrices, with norm ``scale``. Supports eager, jitted and vmapped init via a
    pure host callback. Float64 output requires JAX x64 mode; internal host
    arithmetic is always float64, regardless of JAX's precision setting.
    """
    scale = float(scale)
    if not math.isfinite(scale):
        raise ValueError("deterministic_orthogonal requires a finite scale")

    def init(key, shape, dtype=jnp.float32):
        shape = tuple(operator.index(dimension) for dimension in shape)
        if len(shape) != 2 or any(dimension <= 0 for dimension in shape):
            raise ValueError("deterministic_orthogonal requires a nonempty 2-D shape")
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("deterministic_orthogonal supports float32 and float64 only")
        dtype = np.dtype(jax.dtypes.canonicalize_dtype(dtype))
        callback = partial(_host_orthogonal, shape=shape, dtype=dtype, scale=scale)
        return jax.pure_callback(
            callback,
            jax.ShapeDtypeStruct(shape, dtype),
            jax.random.key_data(key),
            vmap_method="sequential",
        )

    return init
