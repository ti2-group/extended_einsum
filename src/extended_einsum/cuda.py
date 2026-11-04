"""Optional CUDA compilers with explicit stability and evidence-aware selection.

Importing this module does not import Helion or Triton. CPU and differentiable
programs keep using the regular Torch backend. See docs/custom-kernels.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

import torch

from extended_einsum.backend_translation import translate_to_backend_program
from extended_einsum.backend_translation.runtime import run_program
from extended_einsum.backends.helion_categorical import CategoricalInput
from extended_einsum.backends.torch import TorchBackendFunctions
from extended_einsum.language.rich_instruction import RichInstruction
from extended_einsum.language.rich_operators import OperatorEinsum, OperatorExp, OperatorLog, OperatorSelect, OperatorTake
from extended_einsum.language.rich_program import RichProgram

MODES = ("logspace_max", "logspace_min", "scaled_max", "scaled_sum", "scaled_min")


@dataclass(frozen=True)
class Selection:
    backend: str
    reason: str
    options: dict


def _cuda_ready(inputs):
    tensors = [t for value in inputs for t in (value.tensors if isinstance(value, CategoricalInput) else (value,))]
    return bool(tensors) and all(t.device.type == "cuda" and t.device == tensors[0].device for t in tensors)


def _gradients(inputs):
    return any(x.requires_grad for x in inputs)


class CompiledProgram:
    def __init__(self, program, inputs, selection, runner=None):
        self.selection = selection
        self._runner = runner
        self._program = translate_to_backend_program(program, TorchBackendFunctions())
        self._specs = tuple((tuple(x.shape), x.dtype, x.device) for x in inputs)

    def __call__(self, inputs):
        if tuple((tuple(x.shape), x.dtype, x.device) for x in inputs) != self._specs:
            raise ValueError("Input metadata differs from compilation")
        if self._runner is not None:
            if torch.is_grad_enabled() and _gradients(inputs):
                # Custom inference kernels must never silently discard gradients.
                return self.reference(inputs)
            return self._runner(inputs)
        return self.reference(inputs)

    def reference(self, inputs):
        values = [x.materialize() if isinstance(x, CategoricalInput) else x for x in inputs]
        return run_program(self._program, values)


def compile_cuda(program, inputs, *, batch_axes, backend="auto", precision="tf32x3", **options):
    """Compile an arbitrary supported RichProgram; preserve its stability mode.

    Auto uses Helion subtrees for CUDA inference and Torch for CPU/autograd or
    unsupported schedules. Explicit ``backend='helion'`` surfaces compiler errors.
    ``precision='tf32'`` opts into reduced mantissa precision. No global Torch
    precision settings are changed. Compilation and preparation are not cached.
    """
    if backend not in ("auto", "torch", "helion"):
        raise ValueError("backend must be auto, torch, or helion")
    if precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("Unsupported precision")
    reason = "Explicit Torch backend"
    eligible = _cuda_ready(inputs) and not _gradients(inputs) and find_spec("helion") is not None
    if backend == "helion" or (backend == "auto" and eligible):
        from extended_einsum.backends.helion import compile_helion

        config = dict(schedule="subtree", subtree_depth=2, block_size=16, reduction_block_size=32, dot_precision=precision, grid_order="samples")
        config.update(options)
        try:
            runner = compile_helion(program, inputs, batch_axes=batch_axes, **config)
            return CompiledProgram(program, inputs, Selection("helion", "Supported CUDA inference schedule", config), runner)
        except (ValueError, NotImplementedError) as exc:
            if backend != "auto":
                raise
            reason = f"Torch fallback: {exc}"
    elif backend == "auto":
        reason = "Torch fallback: CPU, autograd, mixed devices, or missing optional Helion dependency"
    return CompiledProgram(program, inputs, Selection("torch", reason, {}))


def _regular_program(binding, weights, mode):
    f, b, u = binding.shape
    inputs, axes = [binding], [1]
    weight_ids, routing = [], []
    for level, w in enumerate(weights):
        weight_ids.append(len(inputs))
        inputs.append(w)
        axes.append(None)
        if level:
            folds = f >> level
            routing.append((len(inputs), len(inputs) + 1))
            inputs.extend([torch.arange(0, 2 * folds, 2, device=binding.device), torch.arange(1, 2 * folds, 2, device=binding.device)])
            axes.extend([None, None])
    shapes = [tuple(x.shape) for x in inputs]
    instructions = []

    def append(op, args, shape):
        result = len(inputs) + len(instructions)
        instructions.append(RichInstruction(op, tuple(args)))
        shapes.append(shape)
        return result

    current = append(OperatorExp(), (0,), binding.shape)
    current = append(OperatorEinsum("fbi,foi->fbo"), (current, weight_ids[0]), binding.shape)
    for level, (left, right) in enumerate(routing, 1):
        shape = (f >> level, b, u)
        a = append(OperatorTake(0), (current, left), shape)
        z = append(OperatorTake(0), (current, right), shape)
        current = append(OperatorEinsum("fbi,fbi,foi->fbo"), (a, z, weight_ids[level]), shape)
    current = append(OperatorSelect(0, 0), (current,), (b, u))
    append(OperatorLog(), (current,), (b, u))
    return RichProgram(instructions, len(inputs), mode, ["dense"] * len(shapes), shapes, frozenset(weight_ids)), inputs, axes


class CompiledCircuit:
    """Regular complete binary tree; selected backend and options are inspectable."""

    def __init__(self, binding, weights, *, stability="logspace_max", backend="auto", precision="tf32x3"):
        if stability not in MODES or backend not in ("auto", "torch", "helion", "triton"):
            raise ValueError("Unsupported stability/backend")
        if precision not in ("tf32", "tf32x3", "ieee"):
            raise ValueError("Unsupported precision")
        f, b, u = binding.shape
        if f < 2 or f & (f - 1) or len(weights) != f.bit_length():
            raise ValueError("Requires complete binary tree weights, including a leaf projection")
        if any(tuple(w.shape) != (f >> i, u, u) for i, w in enumerate(weights)):
            raise ValueError("Expected weights[depth][fold,output,input]")
        if binding.logits or not binding.log_values:
            raise ValueError("Requires log-valued categorical probabilities")
        program, inputs, axes = _regular_program(binding, weights, stability)
        self._weight_ids = sorted(program.parameter_indices)
        self._inputs = inputs
        self._specs = tuple((tuple(x.shape), x.dtype, x.device) for x in inputs)
        self._triton = None
        selected = backend
        options = {}
        reason = "Explicit backend"
        eligible = _cuda_ready(inputs) and not _gradients(inputs)
        ampere = eligible and torch.cuda.get_device_capability(binding.device) == (8, 6)
        # Measured A6000 profile. Other devices retain conservative generic paths.
        if backend == "auto":
            selected = "torch"
            reason = "Torch fallback for autograd, CPU, or an unprofiled device/shape"
            if ampere and precision == "tf32" and u in (64, 128, 256) and b >= 128 and find_spec("triton"):
                selected = "triton"
                reason = "A6000 Triton profile; Helion remains an explicit comparison backend"
        if selected == "triton":
            if precision != "tf32":
                raise ValueError("Regular Triton kernel requires explicit precision='tf32'; use Helion/Torch for stricter precision")
            from extended_einsum.backends.triton_circuit import TritonCircuit

            options = dict(block_size=32, num_warps=16, pipeline_stages=3) if u >= 256 else dict(block_size=16, num_warps=8, pipeline_stages=3 if stability.startswith("scaled") else 1)
            self._triton = TritonCircuit(binding, weights, stability=stability, **options)
            self._compiled = CompiledProgram(program, inputs, Selection("torch", "Autograd reference", {}))
            self.selection = Selection(selected, reason, options)
        else:
            if selected == "helion":
                options = dict(block_size=32 if u >= 256 else 16, num_warps=16 if u >= 256 else 8, pipeline_stages=3)
            self._compiled = compile_cuda(program, inputs, batch_axes=axes, backend=selected, precision=precision, **options)
            self.selection = Selection(self._compiled.selection.backend, reason if backend == "auto" else self._compiled.selection.reason, options)

    def __call__(self, binding, weights):
        inputs = list(self._inputs)
        inputs[0] = binding
        if len(weights) != len(self._weight_ids):
            raise ValueError("Weight count differs")
        for i, w in zip(self._weight_ids, weights, strict=True):
            inputs[i] = w
        if self._triton is not None and not (torch.is_grad_enabled() and _gradients(inputs)):
            return self._triton(binding, weights)
        return self._compiled(inputs)


def compile_circuit(binding, weights, **options):
    """Select a regular-tree inference kernel; includes categorical lookup each call."""
    return CompiledCircuit(binding, weights, **options)
