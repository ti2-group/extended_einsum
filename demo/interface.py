import torch

import extended_einsum.interface as xe

x = xe.array(torch.randn(2, 2).abs() + 0.1, format="dense")
w = xe.array(torch.randn(2, 3).abs() + 0.1, format="dense")

intermediate_1 = xe.log(x)
intermediate_2 = xe.log(intermediate_1)
intermediate_3 = xe.exp(intermediate_1)
result = intermediate_2 + intermediate_3

program, inputs = xe.extract_program(result, stability_mode="scaled")
print(program)
print(result.materialize(stability_mode="none"))

einsum_expr = xe.einsum("ij,jk->ik", x, w)
einsum_program, einsum_inputs = xe.extract_program(einsum_expr, stability_mode="scaled")
print(einsum_program)
print(einsum_expr.materialize(stability_mode="none"))
