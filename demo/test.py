import jax.numpy as jnp
import torch

from extended_einsum.backend import get_backend_of_array

a = torch.randn(3, 4, 5)
b = a.numpy()
c = jnp.array(b)

print(get_backend_of_array(a))
print(get_backend_of_array(b))
print(get_backend_of_array(c))
