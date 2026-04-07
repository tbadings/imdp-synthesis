import jax

# jax.config.update("jax_default_matmul_precision", "high")

jax.config.update('jax_platform_name', 'gpu')

print('=== JAX STATUS ===')
print(f'Devices available: {jax.devices()}')
from jax.extend.backend import get_backend

print(f'Jax runs on: {get_backend().platform}')
print('==================\n')