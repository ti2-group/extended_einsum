import jax.numpy as jnp
import numpy as np

import extended_einsum.interface as xe

np.random.seed(0)
x = jnp.array(np.random.randn(4, 4))
y = xe.exp(x)

z = y.materialize()
assert jnp.allclose(z, np.exp(x))
