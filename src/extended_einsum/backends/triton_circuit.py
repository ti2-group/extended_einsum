"""Register-based Triton kernels for regular positive binary circuits.

Inference kernels. Log modes retain log-valued frontiers;
scaled modes carry normalized values and a separate per-row log scale.
"""

from __future__ import annotations

import os

import torch
import triton as tr
import triton.language as tl

from extended_einsum.backends.triton_inputs import _categorical_values

_USE_RECIPROCAL = tl.constexpr(os.environ.get("XE_TRITON_NORMALIZATION", "division") == "reciprocal")


@tr.jit
def _round(x):
    return tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", constraints="=f,f", args=[x], dtype=tl.float32, is_pure=True, pack=1)


@tr.jit
def _norm(x, scale, U: tl.constexpr, P: tl.constexpr, MODE: tl.constexpr):
    cols = tl.arange(0, P)
    if MODE == "scaled_sum":
        n = tl.sum(x, 1)
    elif MODE == "scaled_min":
        n = tl.min(tl.where(cols[None, :] < U, x, float("inf")), 1)
    else:
        n = tl.max(x, 1)
    safe = tl.where(n > 0, n, 1.0)
    if not _USE_RECIPROCAL:
        return x / safe[:, None], scale + tl.log(safe)
    # Keep the reciprocal and log argument normal even for subnormal norms.
    exponent = tl.where(safe < 5.421010862427522e-20, 64, tl.where(safe > 1.8446744073709552e19, -64, 0))
    adjust = tl.where(exponent == 64, 1.8446744073709552e19, tl.where(exponent == -64, 5.421010862427522e-20, 1.0))
    bounded = safe * adjust
    reciprocal = tl.div_rn(1.0, bounded)
    return (x * adjust[:, None]) * reciprocal[:, None], scale + tl.log(bounded) - exponent.to(tl.float32) * 0.6931471805599453


@tr.jit
def _load_weight(W, root, indices, cols, U: tl.constexpr, STORAGE: tl.constexpr, ONLINE: tl.constexpr):
    if STORAGE == "transposed":
        offset = (root * U + indices[:, None]) * U + cols[None, :]
    elif STORAGE == "panels":
        offset = root * U * U + (indices[:, None] // 32) * U * 32 + cols[None, :] * 32 + indices[:, None] % 32
    elif STORAGE == "paired_panels":
        offset = (root // 2) * 2 * U * U + (indices[:, None] // 32) * 2 * U * 32 + (root % 2) * U * 32 + cols[None, :] * 32 + indices[:, None] % 32
    elif STORAGE == "interleaved":
        # Pairs of sibling matrices, contiguous [input, output, sibling].
        offset = (root // 2) * (2 * U * U) + (indices[:, None] * U + cols[None, :]) * 2 + root % 2
    else:
        offset = (root * U + cols[None, :]) * U + indices[:, None]
    return tl.load(W + offset, (cols < U)[None, :] & (indices < U)[:, None], other=float("-inf") if ONLINE else 0.0)


@tr.jit
def _tl_pair(
    Pixels,
    Tables,
    Variables,
    Banks,
    Prev,
    Scales,
    Weights,
    root,
    rows,
    B: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BANKS: tl.constexpr,
    U: tl.constexpr,
    P: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    FIRST: tl.constexpr,
    MODE: tl.constexpr,
    LOOKUP: tl.constexpr,
    STAGES: tl.constexpr,
    STORAGE: tl.constexpr,
    ONLINE: tl.constexpr,
    TRANSPOSE_FRONTIER: tl.constexpr = False,
):
    xl, sl = _tl_prepare(
        Pixels, Tables, Variables, Banks, Prev, Scales, Weights, root * 2, rows, B, V, C, BANKS, U, P, M, K, 1, FIRST, MODE, LOOKUP, STAGES, STORAGE, ONLINE, False, TRANSPOSE_FRONTIER
    )
    xr, sr = _tl_prepare(
        Pixels, Tables, Variables, Banks, Prev, Scales, Weights, root * 2 + 1, rows, B, V, C, BANKS, U, P, M, K, 1, FIRST, MODE, LOOKUP, STAGES, STORAGE, ONLINE, False, TRANSPOSE_FRONTIER
    )
    cols = tl.arange(0, P)
    kk = tl.arange(0, K)
    al = tl.full((M, P), 0.0, tl.float32)
    ar = tl.full((M, P), 0.0, tl.float32)
    for panel in tl.range(tr.cdiv(U, K), num_stages=STAGES):
        indices = panel * K + kk
        gather = tl.broadcast_to(tl.minimum(indices, P - 1)[None, :], (M, K))
        a = tl.where(indices[None, :] < U, tl.gather(xl, gather, 1), 0.0)
        w = _load_weight(Weights[0], root * 2, indices, cols, U, STORAGE[0], False)
        al = tl.dot(_round(a), _round(w), al, input_precision="tf32")
        a = tl.where(indices[None, :] < U, tl.gather(xr, gather, 1), 0.0)
        w = _load_weight(Weights[0], root * 2 + 1, indices, cols, U, STORAGE[0], False)
        ar = tl.dot(_round(a), _round(w), ar, input_precision="tf32")
    al = tl.where(cols[None, :] < U, al, 0.0)
    ar = tl.where(cols[None, :] < U, ar, 0.0)
    if MODE.startswith("scaled"):
        al, sl = _norm(al, sl, U, P, MODE)
        ar, sr = _norm(ar, sr, U, P, MODE)
    else:
        al = tl.log(al) + sl[:, None]
        ar = tl.log(ar) + sr[:, None]
    return al, sl, ar, sr


@tr.jit
def _tl_prepare(
    Pixels,
    Tables,
    Variables,
    Banks,
    Prev,
    Scales,
    Weights,
    root,
    rows,
    B: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BANKS: tl.constexpr,
    U: tl.constexpr,
    P: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    FIRST: tl.constexpr,
    MODE: tl.constexpr,
    LOOKUP: tl.constexpr,
    STAGES: tl.constexpr,
    STORAGE: tl.constexpr = ("standard", "standard", "standard", "standard"),
    ONLINE: tl.constexpr = False,
    PAIR: tl.constexpr = False,
    TRANSPOSE_FRONTIER: tl.constexpr = False,
):
    cols = tl.arange(0, P)
    if H == 2 and PAIR:
        left, ls, right, rs = _tl_pair(
            Pixels, Tables, Variables, Banks, Prev, Scales, Weights, root, rows, B, V, C, BANKS, U, P, M, K, FIRST, MODE, LOOKUP, STAGES, STORAGE, ONLINE, TRANSPOSE_FRONTIER
        )
        if MODE.startswith("scaled"):
            if MODE == "scaled_min":
                x, shift = _norm(left * right, ls + rs, U, P, MODE)
            else:
                x, shift = left * right, ls + rs
        else:
            x = left + right
    elif H > 1:
        left, ls = _tl_node(
            Pixels, Tables, Variables, Banks, Prev, Scales, Weights, root * 2, rows, B, V, C, BANKS, U, P, M, K, H - 1, FIRST, MODE, LOOKUP, STAGES, STORAGE, ONLINE, PAIR, TRANSPOSE_FRONTIER
        )
        right, rs = _tl_node(
            Pixels, Tables, Variables, Banks, Prev, Scales, Weights, root * 2 + 1, rows, B, V, C, BANKS, U, P, M, K, H - 1, FIRST, MODE, LOOKUP, STAGES, STORAGE, ONLINE, PAIR, TRANSPOSE_FRONTIER
        )
        if MODE.startswith("scaled"):
            if MODE == "scaled_min":
                x, shift = _norm(left * right, ls + rs, U, P, MODE)
            else:
                x, shift = left * right, ls + rs
        else:
            x = left + right
    elif FIRST:
        if LOOKUP == 2:
            x = tl.load(Prev + (root * B + rows[:, None]) * U + cols[None, :], (rows < B)[:, None] & (cols < U)[None, :], other=0.0)
        else:
            x = _categorical_values(Pixels, Tables, Variables, Banks, root, rows, cols, B, V, C, BANKS, U, LOOKUP)
        if MODE.startswith("scaled"):
            shift = tl.max(tl.where(cols[None, :] < U, x, float("-inf")), 1)
            shift = tl.where(shift == float("-inf"), 0.0, shift)
            x = tl.where(cols[None, :] < U, tl.exp(x - shift[:, None]), 0.0)
            if MODE != "scaled_max":
                x, shift = _norm(x, shift, U, P, MODE)
    else:
        if TRANSPOSE_FRONTIER:
            lo = root * 2 * B * U + cols[None, :] * B + rows[:, None]
            ro = (root * 2 + 1) * B * U + cols[None, :] * B + rows[:, None]
        else:
            lo = ((root * 2) * B + rows[:, None]) * U + cols[None, :]
            ro = ((root * 2 + 1) * B + rows[:, None]) * U + cols[None, :]
        left = tl.load(Prev + lo, (rows < B)[:, None] & (cols < U)[None, :], other=0.0)
        right = tl.load(Prev + ro, (rows < B)[:, None] & (cols < U)[None, :], other=0.0)
        if MODE.startswith("scaled"):
            ls = tl.load(Scales + root * 2 * B + rows, rows < B, other=0.0)
            rs = tl.load(Scales + (root * 2 + 1) * B + rows, rows < B, other=0.0)
            if MODE == "scaled_min":
                x, shift = _norm(left * right, ls + rs, U, P, MODE)
            else:
                x, shift = left * right, ls + rs
        else:
            x = left + right
    if not MODE.startswith("scaled"):
        shift = tl.max(tl.where(cols[None, :] < U, x, float("-inf")), 1)
        if MODE == "logspace_min":
            shift = tl.min(tl.where(cols[None, :] < U, x, float("inf")), 1)
        shift = tl.where(tl.abs(shift) == float("inf"), 0.0, shift)
        x = tl.where(cols[None, :] < U, tl.exp(x - shift[:, None]), 0.0)
    return x, shift


@tr.jit
def _tl_node(
    Pixels,
    Tables,
    Variables,
    Banks,
    Prev,
    Scales,
    Weights,
    root,
    rows,
    B: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BANKS: tl.constexpr,
    U: tl.constexpr,
    P: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    FIRST: tl.constexpr,
    MODE: tl.constexpr,
    LOOKUP: tl.constexpr,
    STAGES: tl.constexpr,
    STORAGE: tl.constexpr = ("standard", "standard", "standard", "standard"),
    ONLINE: tl.constexpr = False,
    PAIR: tl.constexpr = False,
    TRANSPOSE_FRONTIER: tl.constexpr = False,
):
    x, shift = _tl_prepare(Pixels, Tables, Variables, Banks, Prev, Scales, Weights, root, rows, B, V, C, BANKS, U, P, M, K, H, FIRST, MODE, LOOKUP, STAGES, STORAGE, ONLINE, PAIR, TRANSPOSE_FRONTIER)
    cols = tl.arange(0, P)
    acc = tl.full((M, P), 0.0, tl.float32)
    maximum = tl.full((P,), float("-inf"), tl.float32)
    denominator = tl.full((P,), 0.0, tl.float32)
    kk = tl.arange(0, K)
    for panel in tl.range(tr.cdiv(U, K), num_stages=STAGES):
        indices = panel * K + kk
        a = tl.gather(x, tl.broadcast_to(tl.minimum(indices, P - 1)[None, :], (M, K)), 1)
        a = tl.where(indices[None, :] < U, a, 0.0)
        w = _load_weight(Weights[H - 1], root, indices, cols, U, STORAGE[H - 1], ONLINE)
        if ONLINE:
            newmax = tl.maximum(maximum, tl.max(w, 0))
            newmax = tl.where(cols < U, newmax, 0.0)
            alpha = tl.exp(maximum - newmax)
            w = tl.exp(w - newmax[None, :])
            denominator = denominator * alpha + tl.sum(w, 0)
            acc = acc * alpha[None, :]
            maximum = newmax
        acc = tl.dot(_round(a), _round(w), acc, input_precision="tf32")
    if ONLINE:
        acc = acc / tl.where(denominator > 0, denominator, 1.0)[None, :]
    acc = tl.where(cols[None, :] < U, acc, 0.0)
    if MODE.startswith("scaled"):
        return _norm(acc, shift, U, P, MODE)
    return tl.log(acc) + shift[:, None], shift


@tr.jit
def _tl_kernel(
    Pixels,
    Tables,
    Variables,
    Banks,
    Prev,
    Scales,
    Weights,
    Out,
    OutScale,
    B: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BANKS: tl.constexpr,
    U: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    FIRST: tl.constexpr,
    LAST: tl.constexpr,
    MODE: tl.constexpr,
    LOOKUP: tl.constexpr,
    STAGES: tl.constexpr,
    ORDER: tl.constexpr,
    STORAGE: tl.constexpr = ("standard", "standard", "standard", "standard"),
    ONLINE: tl.constexpr = False,
    PAIR: tl.constexpr = False,
    TRANSPOSE_FRONTIER: tl.constexpr = False,
):
    if ORDER == "samples":
        tile, root = tl.program_id(0), tl.program_id(1)
    else:
        root, tile = tl.program_id(0), tl.program_id(1)
    rows = tile * M + tl.arange(0, M)
    P: tl.constexpr = max(16, tr.next_power_of_2(U))
    x, s = _tl_node(Pixels, Tables, Variables, Banks, Prev, Scales, Weights, root, rows, B, V, C, BANKS, U, P, M, K, H, FIRST, MODE, LOOKUP, STAGES, STORAGE, ONLINE, PAIR, TRANSPOSE_FRONTIER)
    if MODE.startswith("scaled"):
        if LAST:
            x = tl.log(x) + s[:, None]
        else:
            tl.store(OutScale + root * B + rows, s, rows < B)
    cols = tl.arange(0, P)
    if TRANSPOSE_FRONTIER:
        offset = root * B * U + cols[None, :] * B + rows[:, None]
    else:
        offset = (root * B + rows[:, None]) * U + cols[None, :]
    tl.store(Out + offset, x, (rows < B)[:, None] & (cols < U)[None, :])


class TritonCircuit:
    """Fused categorical lookup and two-level register subtrees (TF32)."""

    def __init__(self, binding, weights, *, stability="logspace_max", block_size=32, num_warps=16, pipeline_stages=3, reduction_block_size=32, online_weights=False):
        self.shape, self.device = binding.shape, binding.device
        f, _, u = self.shape
        if self.device.type != "cuda" or binding.logits or not binding.log_values:
            raise ValueError("Requires CUDA log-valued categorical probabilities")
        if f < 2 or f & (f - 1):
            raise ValueError("Requires a complete binary tree")
        if stability not in ("logspace_max", "logspace_min", "scaled_max", "scaled_sum", "scaled_min"):
            raise ValueError("Unsupported stability mode")
        if f * binding.shape[1] * u >= 2**31:
            raise ValueError("Node storage exceeds 32-bit indexing")
        self.mode, self.m, self.warps, self.stages = stability, block_size, num_warps, pipeline_stages
        self.k = reduction_block_size
        self.online = online_weights
        self.specs = tuple((x.shape, x.dtype) for x in binding.tensors)
        self.compiled = []
        self._check_weights(weights)

    def _check_weights(self, weights):
        f, _, u = self.shape
        if len(weights) != f.bit_length() or any(
            tuple(w.shape) != (f >> i, u, u) or w.device != self.device or w.dtype != torch.float32 or not w.is_contiguous() or w.numel() >= 2**31 for i, w in enumerate(weights)
        ):
            raise ValueError("Weights must be contiguous FP32 [fold,output,input] tensors per level")

    def __call__(self, binding, weights):
        if binding.shape != self.shape or binding.device != self.device or binding.logits or not binding.log_values or tuple((x.shape, x.dtype) for x in binding.tensors) != self.specs:
            raise ValueError("Evidence metadata differs from construction")
        self._check_weights(weights)
        if torch.is_grad_enabled() and (binding.requires_grad or any(w.requires_grad for w in weights)):
            raise ValueError("Triton circuit is inference only")
        f, b, u = self.shape
        prev = scales = binding.tables
        kernels = []
        with torch.cuda.device(self.device):
            for level in range(0, len(weights), 2):
                h = min(2, len(weights) - level)
                roots = f >> (level + h - 1)
                out = torch.empty((roots, b, u), device=self.device, dtype=torch.float32)
                outscale = torch.empty((roots, b), device=self.device, dtype=torch.float32) if self.mode.startswith("scaled") else out
                kernels.append(
                    _tl_kernel[(tr.cdiv(b, self.m), roots)](
                        *binding.tensors,
                        prev,
                        scales,
                        tuple(weights[level : level + h]),
                        out,
                        outscale,
                        b,
                        binding.pixels.shape[1],
                        binding.tables.shape[1],
                        binding.tables.shape[0],
                        u,
                        self.m,
                        max(16, min(self.k, tr.next_power_of_2(u))),
                        h,
                        level == 0,
                        level + h == len(weights),
                        self.mode,
                        0,
                        self.stages,
                        "samples",
                        STORAGE=("standard",) * h,
                        ONLINE=self.online,
                        PAIR=False,
                        TRANSPOSE_FRONTIER=False,
                        num_warps=self.warps,
                        num_stages=self.stages,
                    )
                )
                prev, scales = out, outscale
        self.compiled = kernels
        return prev[0]
