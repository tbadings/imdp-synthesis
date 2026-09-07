import argparse
import logging

import jax
import numpy as np

logger = logging.getLogger(__name__)


def configure_jax(args: argparse.Namespace) -> None:
    # Use 'highest' precision for float32 matmuls (disables TF32 on Ampere/Ada/Hopper GPUs)
    # to ensure identical IEEE-754 precision on both CPU and GPU.
    jax.config.update("jax_default_matmul_precision", "highest")

    # Keep explicitly requested int64 arrays at int64 (default is to warn and truncate to int32). Only
    # dtypes asked for by name are affected, so defaults stay 32-bit and floats stay float32. Partition
    # cell keys need this on fine grids, where int32 keys would wrap and alias distinct cells.
    jax.config.update("jax_explicit_x64_dtypes", "allow")

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

    # RL exploration / policy training device (defaults to CPU for cross-hardware determinism)
    rl_device_choice = getattr(args, 'rl_device', 'cpu')
    if rl_device_choice == 'gpu' and args.gpu:
        try:
            args.rl_device_obj = jax.devices('gpu')[0]
            logger.info('Requested to run RL on GPU')
        except Exception:
            args.rl_device_obj = jax.devices('cpu')[0]
            logger.warning('GPU requested for RL but unavailable; falling back to CPU')
    else:
        args.rl_device_obj = jax.devices('cpu')[0]
        logger.info('Requested to run RL on CPU (deterministic)')

    logger.info('JAX backend in use: %s | RL device: %s', args.rvi_device.platform, args.rl_device_obj.platform)
    logger.debug('JAX devices (%s): %s', args.rvi_device.platform, jax.devices(args.rvi_device.platform))

    # In debug mode, configure jax to use Float64 (for more accurate computations)
    if args.debug:
        from jax import config

        config.update("jax_enable_x64", True)

    # Report the widths in use, after the x64 switch above has settled. Integer widths chosen further
    # down (partition cell keys, FRS merge keys) log themselves where they are picked.
    x64_enabled = getattr(jax.config, 'jax_enable_x64', False)
    matmul_prec = getattr(jax.config, 'jax_default_matmul_precision', 'highest')
    explicit_x64 = getattr(jax.config, 'jax_explicit_x64_dtypes', None)
    explicit_x64_str = explicit_x64.name.lower() if hasattr(explicit_x64, 'name') else str(explicit_x64).lower()

    logger.info('Float precision: %s | JAX x64: %s | matmul precision: %s | explicit int64 dtypes: %s',
                np.dtype(args.floatprecision).name,
                'enabled' if x64_enabled else 'disabled',
                matmul_prec,
                explicit_x64_str)
