import itertools
import time

import jax.numpy as jnp
import numpy as np


def parse_linear_model(base_model):
    '''
    Parse linear dynamical model

    :param base_model: Input model.
    :return: Model object
    '''

    print('Parse linear dynamical model...')
    t = time.time()

    # If independent_dimensions is not defined, then assume all dimensions are dependent
    if model.independent_dimensions_x is None:
        model.independent_dimensions_x = [jnp.arange(model.n)]
    if model.independent_dimensions_u is None:
        model.independent_dimensions_u = [jnp.arange(model.p)]

    base_model.partition['boundary'] = jnp.array(base_model.partition['boundary']).astype(float)
    base_model.partition['number_per_dim'] = jnp.array(base_model.partition['number_per_dim']).astype(int)

    # Control limitations
    base_model.uMin = jnp.array(base_model.uMin).astype(float)
    base_model.uMax = jnp.array(base_model.uMax).astype(float)

    lump = base_model.lump

    if lump == 0:
        model = make_fully_actuated(base_model,
                                    manualDimension='auto')
    else:
        model = make_fully_actuated(base_model,
                                    manualDimension=lump)

    # Determine vertices of the control input space
    stacked = np.vstack((model.uMin, model.uMax))
    model.uVertices = jnp.array(list(itertools.product(*stacked.T)))

    # Determine inverse A matrix
    model.A_inv = np.linalg.inv(model.A)

    # Determine pseudo-inverse B matrix
    model.B_pinv = np.linalg.pinv(model.B)

    # Retreive system dimensions
    model.p = np.size(model.B, 1)  # Nr of inputs

    # Determine what the equilibrium point of the linear system is
    uAvg = (model.uMin + model.uMax) / 2
    if np.linalg.matrix_rank(np.eye(model.n) - model.A) == model.n:
        model.equilibrium = jnp.array(np.linalg.inv(np.eye(model.n) - model.A) @ \
                                      (model.B @ uAvg + model.q), dtype=float)

    # Convert from np to jnp
    model.A = jnp.array(model.A, dtype=float)
    model.B = jnp.array(model.B, dtype=float)
    model.A_inv = jnp.array(model.A_inv, dtype=float)
    model.B_pinv = jnp.array(model.B_pinv, dtype=float)
    model.q = jnp.array(model.q, dtype=float)
    model.uMin = jnp.array(model.uMin, dtype=float)
    model.uMax = jnp.array(model.uMax, dtype=float)
    model.noise['cov'] = jnp.array(model.noise['cov'])
    model.noise['cov_diag'] = jnp.array(model.noise['cov_diag'])

    print(f'- Model parsing done (took {(time.time() - t):.3f} sec.)')
    print('')
    return model


def parse_nonlinear_model(model):
    '''
    Parse nonlinear dynamical model

    :param base_model: Input model.
    :return: Model object
    '''

    print('Parse nonlinear dynamical model...')
    t = time.time()

    # If independent_dimensions is not defined, then assume all dimensions are dependent
    if model.independent_dimensions_x is None:
        model.independent_dimensions_x = [jnp.arange(model.n)]
    if model.independent_dimensions_u is None:
        model.independent_dimensions_u = [jnp.arange(model.p)]

    model.partition['boundary'] = jnp.array(model.partition['boundary']).astype(float)
    model.partition['number_per_dim'] = jnp.array(model.partition['number_per_dim']).astype(int)

    # Control limitations
    model.uMin = jnp.array(model.uMin, dtype=float)
    model.uMax = jnp.array(model.uMax, dtype=float)

    # Convert from np to jnp
    model.noise['cov'] = jnp.array(model.noise['cov'])
    model.noise['cov_diag'] = jnp.array(model.noise['cov_diag'])

    # Determine vertices of the control input space
    stacked = np.vstack((model.uMin, model.uMax))
    model.uVertices = jnp.array(list(itertools.product(*stacked.T)))

    print(f'- Model parsing done (took {(time.time() - t):.3f} sec.)')
    print('')
    return model


def make_fully_actuated(model, manualDimension='auto'):
    '''
    Given a parsed model, render it fully actuated. Only available for linear models.

    :param model: Parsed model object.
    :param manualDimension: Integer or 'auto'. If 'auto', then the dimension is determined automatically. If an integer, than that number of of time steps is lumped together used.
    :return: Model object.
    '''

    if manualDimension == 'auto':
        # Determine dimension for actuation transformation
        dim = int(np.size(model.A, 1) / np.size(model.B, 1))
    else:
        # Group a manual number of time steps
        dim = int(manualDimension)

    # Determine fully actuated system matrices and parameters
    A_hat = np.linalg.matrix_power(model.A, (dim))
    B_hat = np.concatenate([np.linalg.matrix_power(model.A, (dim - i)) \
                            @ model.B for i in range(1, dim + 1)], 1)

    q_hat = sum([np.linalg.matrix_power(model.A, (dim - i)) @ model.q
                 for i in range(1, dim + 1)])

    w_sigma_hat = sum([np.array(np.linalg.matrix_power(model.A, (dim - i))
                                @ model.noise['cov'] @
                                np.linalg.matrix_power(model.A.T, (dim - i))
                                ) for i in range(1, dim + 1)])

    # Overwrite original system matrices
    model.A = A_hat
    model.B = B_hat
    model.q = q_hat

    # Update control dimension
    model.p = np.size(model.B, 1)  # Nr of inputs

    model.noise['cov'] = w_sigma_hat

    # Redefine sampling time of model
    model.tau *= dim

    model.uMin = jnp.repeat(model.uMin, dim)
    model.uMax = jnp.repeat(model.uMax, dim)

    return model