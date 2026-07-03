import torch

import extended_einsum.interface as xe
from extended_einsum.backend_translation import BackendCompiler, run_program, translate_to_backend_program
from extended_einsum.backends.registry import BACKEND_TO_COMPILER, BACKEND_TO_FUNCTIONS

x = xe.array(torch.randn(2, 2).abs() + 0.1, format="dense")
w = xe.array(torch.randn(2, 3).abs() + 0.1, format="dense")

intermediate_1 = xe.softmax(x)
intermediate_2 = xe.log(intermediate_1)
intermediate_3 = xe.exp(intermediate_1)
result = xe.einsum("ij,jk->ik", intermediate_2, intermediate_3)

# print(result.materialize(stability_mode="unstable"))
# print(result.materialize(stability_mode="scaled"))
# print(result.materialize(stability_mode="logspace"))

rich_program, input_arguments = xe.extract_program(result, stability_mode="scaled")
backend_program = translate_to_backend_program(rich_program, BACKEND_TO_FUNCTIONS[result.backend])
compiler: BackendCompiler[torch.Tensor] = BACKEND_TO_COMPILER[result.backend]
backend_code = compiler.compile(backend_program, [argument.backend_array for argument in input_arguments])

x_new = torch.randn(2, 2).abs() + 0.1
w_new = torch.randn(2, 3).abs() + 0.1
# result_tensor = backend_code([x_new, w_new])
run_program(backend_program, [x_new, w_new])
