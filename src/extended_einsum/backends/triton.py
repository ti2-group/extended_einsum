"""Compiled-only Triton backend. No eager operations or Torch fallback."""

from dataclasses import dataclass
from importlib.util import find_spec
from types import SimpleNamespace

import torch

from extended_einsum.compiled import UnsupportedProgramError
from extended_einsum.inputs import CategoricalInput
from extended_einsum.language.rich_operators import OperatorEinsum, OperatorExp, OperatorLog, OperatorSelect, OperatorTake

MODES = ("logspace_max", "logspace_min", "scaled_max", "scaled_sum", "scaled_min")


@dataclass(frozen=True)
class Selection:
    backend: str
    reason: str
    options: dict


def _reject(message):
    raise UnsupportedProgramError(f"Triton: {message}")


def _tensors(value):
    return value.tensors if isinstance(value, CategoricalInput) else (value,)


def _spec(value):
    return (type(value), tuple((tuple(t.shape), t.dtype, t.device, t.stride()) for t in _tensors(value)), (value.logits, value.log_values) if isinstance(value, CategoricalInput) else ())


def _equation(equation):
    # Einsum labels are arbitrary: recognize by incidence and axis order.
    labels = {}
    return "".join(labels.setdefault(c, str(len(labels))) if c.isalpha() else c for c in equation.replace(" ", ""))


class TritonBackend:
    """Compile supported RichProgram patterns into the fused tree kernels.

    Initial subset: complete binary trees with a leaf projection, equal units,
    even/odd sibling takes, select(root=0), then log. Input may be a dense
    positive tensor, exp(dense log values), or exp(CategoricalInput). Weights
    are linear-space parameter inputs. Routing inputs must be declared in
    constant_inputs; changed routing requires recompilation.
    """

    def compile(self, program, inputs, *, precision="tf32x3", constant_inputs=()):
        if program.stability_mode not in MODES:
            _reject(f"unsupported stability mode {program.stability_mode!r}")
        if precision != "tf32":
            _reject("regular-tree inference currently requires precision='tf32'")
        if len(inputs) != program.n_inputs or not program.instructions:
            _reject("input count or empty program")
        if any(f != "dense" for f in program.tensor_formats):
            _reject("only dense programs are supported")
        # Validate IR rather than trusting stale metadata after preprocessing.
        for i, instruction in enumerate(program.instructions, program.n_inputs):
            args = instruction.argument_ssa_ids
            if any(a < 0 or a >= i for a in args):
                _reject(f"invalid SSA arguments at instruction {i}")
            try:
                instruction.operator.check_inputs([SimpleNamespace(shape=program.shapes[a]) for a in args])
                shape = instruction.operator.propagate_shapes([program.shapes[a] for a in args])
            except (ValueError, IndexError, TypeError) as exc:
                _reject(f"invalid shapes at instruction {i}: {exc}")
            if tuple(shape) != tuple(program.shapes[i]):
                _reject(f"inconsistent output shape at instruction {i}")
        used, constants, weights, routes = set(), frozenset(constant_inputs), [], []
        if not constants <= set(range(program.n_inputs)) or constants & program.parameter_indices:
            _reject("constant_inputs must identify non-parameter inputs")

        def node(ssa, cls):
            if ssa < program.n_inputs:
                _reject(f"expected {cls.__name__} at SSA {ssa}")
            ins = program.instructions[ssa - program.n_inputs]
            if not isinstance(ins.operator, cls):
                _reject(f"unsupported {ins.operator.name!r} at SSA {ssa}; expected {cls.__name__}")
            used.add(ssa)
            return ins

        end = node(program.output_ssa, OperatorLog)
        root = node(end.argument_ssa_ids[0], OperatorSelect)
        if root.operator.axis != 0 or root.operator.index != 0:
            _reject("expected select(root=0, axis=0)")
        current = root.argument_ssa_ids[0]
        while True:
            ins = node(current, OperatorEinsum)
            args = ins.argument_ssa_ids
            expected = "fbi,foi->fbo" if len(args) == 2 else "fbi,fbi,foi->fbo"
            if len(args) not in (2, 3) or _equation(ins.operator.format_string) != _equation(expected):
                _reject(f"unsupported contraction at SSA {current}")
            weight = args[-1]
            if weight not in program.parameter_indices or weight >= program.n_inputs:
                _reject("projection weights must be parameter inputs")
            weights.append(weight)
            if len(args) == 2:
                data = args[0]
                break
            left, right = (node(a, OperatorTake) for a in args[:2])
            if left.operator.axis != 0 or right.operator.axis != 0 or left.argument_ssa_ids[0] != right.argument_ssa_ids[0]:
                _reject("expected sibling takes from the same fold axis")
            pair = (left.argument_ssa_ids[1], right.argument_ssa_ids[1])
            if any(a not in constants for a in pair):
                _reject("sibling routing must be declared in constant_inputs")
            routes.append((pair, program.shapes[current][0]))
            current = left.argument_ssa_ids[0]
        input_log = data >= program.n_inputs
        if input_log:
            exp = node(data, OperatorExp)
            data = exp.argument_ssa_ids[0]
        if data >= program.n_inputs:
            _reject("leaf projection must consume an input or exp(input)")
        if used != set(range(program.n_inputs, program.n_inputs + len(program.instructions))):
            _reject("program contains operations outside the supported tree pattern")
        weights.reverse()
        if set(weights) != program.parameter_indices or len(set(weights)) != len(weights):
            _reject("shared or unused parameters are unsupported")
        if set(range(program.n_inputs)) != {data, *weights, *constants}:
            _reject("unused or unsupported program inputs")
        if constants != {i for pair, _ in routes for i in pair}:
            _reject("constant_inputs must contain exactly the sibling routing inputs")
        values = inputs[data]
        if isinstance(values, CategoricalInput) and (not input_log or values.logits or not values.log_values):
            _reject("categorical bindings require exp(input) of log-valued tables")
        if len(values.shape) != 3:
            _reject("expected input[fold,batch,unit]")
        f, b, u = values.shape
        if f < 2 or f & (f - 1) or len(weights) != f.bit_length() or min(b, u) < 1:
            _reject("requires a complete binary tree with a leaf projection and equal units")
        if any(tuple(inputs[w].shape) != (f >> i, u, u) for i, w in enumerate(weights)):
            _reject("weights must have shape [fold,output,input] at each level")
        if tuple(program.shapes[program.output_ssa]) != (b, u):
            _reject("unsupported tree output shape")
        for i, value in enumerate(inputs):
            if tuple(value.shape) != tuple(program.shapes[i]):
                _reject(f"input {i} differs from the program shape")
        tensors = [t for v in inputs for t in _tensors(v)]
        if any(not isinstance(t, torch.Tensor) for t in tensors):
            _reject("requires Torch tensor storage")
        device = values.device
        if device.type != "cuda" or any(t.device != device for t in tensors):
            _reject("requires inputs on one CUDA device")
        if torch.cuda.get_device_capability(device)[0] < 8:
            _reject("requires SM80 or newer")
        if any(t.layout != torch.strided or not t.is_contiguous() or t.numel() == 0 or t.numel() >= 2**31 for t in tensors) or f * b * u >= 2**31:
            _reject("requires nonempty contiguous storage with 32-bit indexing")
        if any(t.dtype != torch.float32 for i in (data, *weights) for t in ((inputs[i].tables,) if isinstance(inputs[i], CategoricalInput) else (inputs[i],))):
            _reject("requires real FP32 values and weights")
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
            _reject("regular-tree compilation is inference only; use the custom training API for backward")
        for pair, folds in routes:
            for side, i in enumerate(pair):
                route = inputs[i]
                if route.dtype not in (torch.int32, torch.int64) or not torch.equal(route, torch.arange(side, 2 * folds, 2, device=device, dtype=route.dtype)):
                    _reject("only even/odd sibling routing is supported")
        for i in constants:
            if inputs[i].is_inference():
                _reject("constant routing needs version counters; construct it outside inference_mode")
        if not find_spec("triton"):
            raise ModuleNotFoundError("Install extended-einsum[cuda] for the Triton compiled backend")
        from extended_einsum.backends.triton_circuit import TritonCircuit

        options = dict(block_size=32, num_warps=16, pipeline_stages=3) if u >= 256 else dict(block_size=16, num_warps=8, pipeline_stages=3 if program.stability_mode.startswith("scaled") else 1)
        kernel = TritonCircuit(values, [inputs[w] for w in weights], stability=program.stability_mode, input_log=input_log, **options)
        return _Executable(inputs, constants, data, weights, kernel, options)


class _Executable:
    def __init__(self, inputs, constants, data, weights, kernel, options):
        self.specs = tuple(_spec(v) for v in inputs)
        self.constants = {i: (inputs[i], inputs[i]._version) for i in constants}
        self.data, self.weights, self.kernel = data, tuple(weights), kernel
        self.selection = Selection("triton", "Compiled complete binary-tree pattern", options)

    def __call__(self, inputs):
        try:
            metadata = tuple(_spec(v) for v in inputs)
        except (AttributeError, RuntimeError, TypeError):
            _reject("input storage type/layout changed; recompile the program")
        if len(inputs) != len(self.specs) or metadata != self.specs:
            _reject("input metadata changed; recompile for the new specialization")
        for i, (value, version) in self.constants.items():
            if inputs[i] is not value or value._version != version:
                _reject("constant routing changed; recompile the program")
        if torch.is_grad_enabled() and any(t.requires_grad for v in inputs for t in _tensors(v)):
            _reject("regular-tree execution is inference only")
        return self.kernel(inputs[self.data], [inputs[i] for i in self.weights])
