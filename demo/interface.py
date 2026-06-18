import jax.numpy as jnp
import torch

import extended_einsum.interface as xe

x = jnp.array(torch.randn(2, 2).abs() + 0.1)
w = jnp.array(torch.randn(2, 3).abs() + 0.1)

intermediate_1 = xe.log(x)
intermediate_2 = xe.log(intermediate_1)
intermediate_3 = xe.exp(intermediate_1)
result = intermediate_2 + intermediate_3

program, inputs = xe.extract_program(result)
print(program)

einsum_expr = xe.einsum("ij,jk->ik", x, w)
einsum_program, einsum_inputs = xe.extract_program(einsum_expr)
print(einsum_program)
