import argparse
import logging

import jax
import numpy as np

logger = logging.getLogger(__name__)


def configure_jax(args: argparse.Namespace) -> None:
    jax.config.update("jax_default_matmul_precision", "high")
    args.floatprecision = np.float32

    if args.gpu:
        jax.config.update('jax_platform_name', 'gpu')
        logger.info('Requested to run on GPU')
    else:
        jax.config.update('jax_platform_name', 'cpu')
        logger.info('Requested to run on CPU')

    if args.gpu_rvi:
        args.rvi_device = jax.devices('gpu')[0]
        logger.info('Requested to run RVI on GPU')
    else:
        args.rvi_device = jax.devices('cpu')[0]
        logger.info('Requested to run RVI on CPU')

    logger.info('JAX backend in use: %s', args.rvi_device.platform)
    logger.debug('JAX devices (%s): %s', args.rvi_device.platform, jax.devices(args.rvi_device.platform))

    # In debug mode, configure jax to use Float64 (for more accurate computations)
    if args.debug:
        from jax import config

        config.update("jax_enable_x64", True)
