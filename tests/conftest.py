import os

# must be set before jax is imported anywhere
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
