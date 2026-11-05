"""Custom forward, backward and Equation (19) updates for unshared tree PCs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import wraps

import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable

from extended_einsum.backends.triton_inputs import _categorical_backward, _leaves, _rna


@tr.jit
def _input(Nodes, Routes, g, b, u, B: tl.constexpr, U: tl.constexpr, ARITY: tl.constexpr):
    node = tl.load(Routes + g * 4)
    left = tl.load(Routes + g * 4 + 1)
    right = tl.load(Routes + g * 4 + 2)
    z = tl.load(Nodes + (left * B + b[:, None]) * U + u[None, :], (b < B)[:, None] & (u < U)[None, :], other=float("-inf"))
    if ARITY == 2:
        z += tl.load(Nodes + (right * B + b[:, None]) * U + u[None, :], (b < B)[:, None] & (u < U)[None, :], other=float("-inf"))
    return node, z


@tr.jit
def _forward(
    Nodes,
    LogW,
    Routes,
    CachedX,
    Shifts,
    F: tl.constexpr,
    CACHE: tl.constexpr,
    B: tl.constexpr,
    U: tl.constexpr,
    OUT: tl.constexpr,
    ARITY: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    K: tl.constexpr,
    STAGES: tl.constexpr,
    PRECISION: tl.constexpr,
):
    g = tl.program_id(1)
    b = tl.program_id(0) * M + tl.arange(0, M)
    u = tl.arange(0, P)
    node, z = _input(Nodes, Routes, g, b, u, B, U, ARITY)
    shift = tl.max(z, 1)
    shift = tl.where(shift == float("-inf"), 0.0, shift)
    # Scale both MMA operands by 2**32 so probabilities at Anemone's
    # float32 floor stay normal at tensor-core input and output boundaries.
    x = tl.exp(z - shift[:, None] + 22.18070977791825)
    if CACHE:
        tl.store(CachedX + ((node - F) * B + b[:, None]) * U + u[None, :], x, (b < B)[:, None] & (u < U)[None, :])
        tl.store(Shifts + (node - F) * B + b, shift, b < B)
    offset = tl.load(Routes + g * 4 + 3)
    acc = tl.full((M, P), 0.0, tl.float32)
    kk = tl.arange(0, K)
    for panel in tl.range(tr.cdiv(U, K), num_stages=STAGES):
        k = panel * K + kk
        a = tl.gather(x, tl.broadcast_to(tl.minimum(k, P - 1)[None, :], (M, K)), 1)
        a = tl.where(k[None, :] < U, a, 0.0)
        w = tl.load(LogW + offset + u[None, :] * U + k[:, None], (u < OUT)[None, :] & (k < U)[:, None], other=float("-inf"))
        w = tl.exp(w + 22.18070977791825)
        acc = tl.dot(_rna(a, PRECISION), _rna(w, PRECISION), acc, input_precision=PRECISION)
    y = (tl.log(acc) - 44.3614195558365) + shift[:, None]
    tl.store(Nodes + (node * B + b[:, None]) * U + u[None, :], y, (b < B)[:, None] & (u < OUT)[None, :])


@tr.jit
def _root(Nodes, Loss, B: tl.constexpr, U: tl.constexpr, ROOT: tl.constexpr, P: tl.constexpr):
    b = tl.arange(0, P)
    y = tl.load(Nodes + (ROOT * B + b) * U, b < B, other=0.0)
    tl.store(Loss, -tl.sum(y, 0) / B)


@tr.jit
def _seed(Grad, Flow, Model, B: tl.constexpr, U: tl.constexpr, ROOT: tl.constexpr, P: tl.constexpr):
    i = tl.program_id(0) * P + tl.arange(0, P)
    v = tl.where(i % U == 0, -tl.load(Grad) / B, 0.0)
    tl.store(Flow + ROOT * B * U + i, v, i < B * U)
    if tl.program_id(0) == 0:
        u = tl.arange(0, P)
        tl.store(Model + ROOT * U + u, tl.where(u == 0, 1.0, 0.0), u < U)


@tr.jit
def _seed_cached(Grad, Flow, Model, Nodes, Shifts, B: tl.constexpr, U: tl.constexpr, F: tl.constexpr, ROOT: tl.constexpr, P: tl.constexpr):
    i = tl.program_id(0) * P + tl.arange(0, P)
    b = i // U
    valid = (b < B) & (i % U == 0)
    y = tl.load(Nodes + ROOT * B * U + i, valid, other=0.0)
    shift = tl.load(Shifts + (ROOT - F) * B + b, valid, other=0.0)
    den = tl.exp(y - shift + 44.3614195558365)
    q = tl.where(i % U == 0, (-tl.load(Grad) / B) / den, 0.0)
    tl.store(Flow + ROOT * B * U + i, q, i < B * U)
    if tl.program_id(0) == 0:
        u = tl.arange(0, P)
        tl.store(Model + ROOT * U + u, tl.where(u == 0, 1.0, 0.0), u < U)


@tr.jit
def _clock(Step, Bias, MOMENTUM: tl.constexpr):
    t = tl.load(Step) + 1
    tl.store(Step, t)
    tl.store(Bias, 1.0 - tl.exp(t.to(tl.float32) * tl.log(MOMENTUM)) if MOMENTUM > 0.0 else 1.0)


@tr.jit
def _prepare_backward(Nodes, Flow, Routes, X, Q, B: tl.constexpr, U: tl.constexpr, OUT: tl.constexpr, ARITY: tl.constexpr, M: tl.constexpr, P: tl.constexpr):
    g = tl.program_id(1)
    b = tl.program_id(0) * M + tl.arange(0, M)
    u = tl.arange(0, P)
    node, z = _input(Nodes, Routes, g, b, u, B, U, ARITY)
    shift = tl.max(z, 1)
    shift = tl.where(shift == float("-inf"), 0.0, shift)
    x = tl.exp(z - shift[:, None] + 22.18070977791825)
    y = tl.load(Nodes + (node * B + b[:, None]) * U + u[None, :], (b < B)[:, None] & (u < OUT)[None, :], other=0.0)
    flow = tl.load(Flow + (node * B + b[:, None]) * U + u[None, :], (b < B)[:, None] & (u < OUT)[None, :], other=0.0)
    # Multiply by flow before division: exp(shift-y) alone can overflow near
    # the float32 floor even though this mean-loss derivative is finite.
    denom = tl.exp(y - shift[:, None] + 44.3614195558365)
    q = tl.where((u < OUT)[None, :], flow / tl.where(denom > 0.0, denom, 1.0), 0.0)
    tl.store(X + (g * B + b[:, None]) * U + u[None, :], x, (b < B)[:, None] & (u < U)[None, :])
    tl.store(Q + (g * B + b[:, None]) * U + u[None, :], q, (b < B)[:, None] & (u < U)[None, :])


@tr.jit
def _children(
    X,
    Q,
    LogW,
    Routes,
    Flow,
    Model,
    Nodes,
    Shifts,
    F: tl.constexpr,
    CACHE: tl.constexpr,
    B: tl.constexpr,
    U: tl.constexpr,
    OUT: tl.constexpr,
    ARITY: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    STAGES: tl.constexpr,
    PRECISION: tl.constexpr,
    MODEL: tl.constexpr,
):
    tile = tl.program_id(0)
    g = tl.program_id(1)
    nt = tl.program_id(2)
    b = tile * M + tl.arange(0, M)
    n = nt * N + tl.arange(0, N)
    kk = tl.arange(0, K)
    offset = tl.load(Routes + g * 4 + 3)
    node = tl.load(Routes + g * 4)
    qi = node if CACHE else g
    xi = node - F if CACHE else g
    acc = tl.full((M, N), 0.0, tl.float32)
    model_acc = tl.full((N,), 0.0, tl.float32)
    for panel in tl.range(tr.cdiv(OUT, K), num_stages=STAGES):
        k = panel * K + kk
        q = tl.load(Q + (qi * B + b[:, None]) * U + k[None, :], (b < B)[:, None] & (k < OUT)[None, :], other=0.0)
        w = tl.exp(tl.load(LogW + offset + k[:, None] * U + n[None, :], (k < OUT)[:, None] & (n < U)[None, :], other=float("-inf")) + 22.18070977791825)
        acc = tl.dot(_rna(q, PRECISION), _rna(w, PRECISION), acc, input_precision=PRECISION)
        if MODEL:
            if tile == 0:
                mf = tl.load(Model + node * U + k, k < OUT, other=0.0)
                model_acc += tl.sum(mf[:, None] * (w * 2.3283064365386963e-10), 0)
    x = tl.load(X + (xi * B + b[:, None]) * U + n[None, :], (b < B)[:, None] & (n < U)[None, :], other=0.0)
    result = acc * x
    left = tl.load(Routes + g * 4 + 1)
    if CACHE:
        if left >= F:
            sy = tl.load(Shifts + (left - F) * B + b, b < B, other=0.0)
            y = tl.load(Nodes + (left * B + b[:, None]) * U + n[None, :], (b < B)[:, None] & (n < U)[None, :], other=0.0)
            den = tl.exp(y - sy[:, None] + 44.3614195558365)
            out_left = result / tl.where(den > 0.0, den, 1.0)
        else:
            out_left = result
    else:
        out_left = result
    tl.store(Flow + (left * B + b[:, None]) * U + n[None, :], out_left, (b < B)[:, None] & (n < U)[None, :])
    if ARITY == 2:
        right = tl.load(Routes + g * 4 + 2)
        if CACHE:
            if right >= F:
                sy = tl.load(Shifts + (right - F) * B + b, b < B, other=0.0)
                y = tl.load(Nodes + (right * B + b[:, None]) * U + n[None, :], (b < B)[:, None] & (n < U)[None, :], other=0.0)
                den = tl.exp(y - sy[:, None] + 44.3614195558365)
                out_right = result / tl.where(den > 0.0, den, 1.0)
            else:
                out_right = result
        else:
            out_right = result
        tl.store(Flow + (right * B + b[:, None]) * U + n[None, :], out_right, (b < B)[:, None] & (n < U)[None, :])
    if MODEL:
        if tile == 0:
            tl.store(Model + left * U + n, model_acc, n < U)
            if ARITY == 2:
                tl.store(Model + right * U + n, model_acc, n < U)


@tr.jit
def _weight_backward(
    X,
    Q,
    LogW,
    Routes,
    GradW,
    Model,
    Momentum,
    Bias,
    F: tl.constexpr,
    CACHE: tl.constexpr,
    B: tl.constexpr,
    U: tl.constexpr,
    OUT: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    K: tl.constexpr,
    STAGES: tl.constexpr,
    PRECISION: tl.constexpr,
    UPDATE: tl.constexpr,
    ALPHA: tl.constexpr,
    MOM: tl.constexpr,
):
    g = tl.program_id(1)
    o = tl.program_id(0) * M + tl.arange(0, M)
    u = tl.arange(0, P)
    kk = tl.arange(0, K)
    node = tl.load(Routes + g * 4)
    qi = node if CACHE else g
    xi = node - F if CACHE else g
    acc = tl.full((M, P), 0.0, tl.float32)
    for panel in tl.range(tr.cdiv(B, K), num_stages=STAGES):
        b = panel * K + kk
        q = tl.load(Q + (qi * B + b[None, :]) * U + o[:, None], (b < B)[None, :] & (o < OUT)[:, None], other=0.0)
        x = tl.load(X + (xi * B + b[:, None]) * U + u[None, :], (b < B)[:, None] & (u < U)[None, :], other=0.0)
        acc = tl.dot(_rna(q, PRECISION), _rna(x, PRECISION), acc, input_precision=PRECISION)
    offset = tl.load(Routes + g * 4 + 3)
    pos = offset + o[:, None] * U + u[None, :]
    mask = (o < OUT)[:, None] & (u < U)[None, :]
    w = tl.exp(tl.load(LogW + pos, mask, other=float("-inf")) + 22.18070977791825)
    grad = acc * w
    if UPDATE:
        node = tl.load(Routes + g * 4)
        mf = tl.load(Model + node * U + o, o < OUT, other=0.0)
        data = -grad
        if MOM > 0.0:
            running = MOM * tl.load(Momentum + pos, mask, other=0.0) + (1.0 - MOM) * data
            tl.store(Momentum + pos, running, mask)
            data = running / tl.load(Bias)
        numerator = tl.maximum((1.0 - ALPHA) * mf[:, None] * (w * 2.3283064365386963e-10) + ALPHA * data, 1.1754943508222875e-38)
        numerator = tl.where((u < U)[None, :], numerator, 0.0)
        logp = tl.log(numerator) - tl.log(tl.sum(numerator, 1))[:, None]
        tl.store(LogW + pos, logp, mask)
    else:
        tl.store(GradW + pos, grad, mask)


@dataclass(frozen=True)
class Group:
    routes: torch.Tensor
    owner: int
    outputs: int
    arity: int
    count: int
    depth: int


def make_plan(source):
    groups = defaultdict(list)
    depths = {i: 0 for i in range(source.num_variables)}
    for node, projection in enumerate(source.projections, source.num_variables):
        owner, offset, outputs, children = projection.owner, projection.offset, projection.outputs, projection.children
        depth = max(depths[t] for t in children) + 1
        depths[node] = depth
        groups[(depth, owner, outputs, len(children))].append((node, children[0], children[-1], offset))
    result = []
    for (depth, owner, outputs, arity), routes in sorted(groups.items()):
        result.append(Group(torch.tensor(routes, dtype=torch.int32, device=source.device), owner, outputs, arity, len(routes), depth))
    return tuple(result), len(depths), len(depths) - 1


class _CircuitLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runner, images, *parameters):
        loss, state = runner.forward_values(images)
        ctx.runner = runner
        ctx.save_for_backward(images, runner.source.variable_ids, *state, *parameters)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        images, variables, nodes, cached_x, shifts, *parameters = ctx.saved_tensors
        result = ctx.runner.backward_values(images, (nodes, cached_x, shifts), grad, update=False, parameters=parameters, variables=variables)
        return None, None, *result


def _device_guard(fn):
    @wraps(fn)
    def guarded(self, *args, **kwargs):
        with torch.cuda.device(self.source.device):
            return fn(self, *args, **kwargs)

    return guarded


class TritonAnemone:
    """Custom loss/autograd and fused EM training over existing model parameters.

    ``loss(images).backward()`` provides true log-parameter gradients. ``step``
    fuses gradient production with Equation (19), avoiding parameter-sized
    gradient/model-flow buffers. Optimizer state is separate from the source
    optimizer; source parameters are updated in place. Steps are counted on GPU,
    so CUDA graph replay advances momentum bias correction correctly.
    With ``cache_inputs=True``, the forward saves its GEMM inputs; the reverse
    pass produces child coefficients directly, eliminating backward preparation
    launches. Log-valued node frontiers are retained in both variants.
    """

    def __init__(
        self,
        source,
        batch_size,
        *,
        alpha=0.4,
        momentum=0.9,
        precision="tf32",
        block_size=32,
        num_warps=8,
        pipeline_stages=2,
        child_units=64,
        weight_rows=16,
        weight_warps=8,
        categorical_units=32,
        categorical_warps=8,
        child_stages=None,
        weight_stages=None,
        categorical_stages=None,
        cache_inputs=False,
    ):
        if source.device.type != "cuda":
            raise ValueError("Requires CUDA tree parameters")
        if precision not in ("tf32", "tf32x3", "ieee"):
            raise ValueError("Unsupported precision")
        self.cache_inputs = cache_inputs
        self.source = source
        self.parameters = tuple(p.detach() for p in source.log_parameters()) if source.parameterization == "logits" else tuple(source.log_parameters())
        if any(p.dtype != torch.float32 or not p.is_contiguous() for p in self.parameters):
            raise ValueError("Requires contiguous FP32 log parameters")
        self.groups, self.nodes, self.root = make_plan(source)
        if (source.num_variables + len(source.projections)) * batch_size * source.units >= 2**31:
            raise ValueError("Node storage exceeds 32-bit indexing")
        self.b = batch_size
        self.u = source.units
        self.f = source.num_variables
        self.c = self.parameters[0].shape[1]
        self.precision = precision
        self.m = block_size
        self.warps = num_warps
        self.stages = pipeline_stages
        self.bs = child_stages if child_stages is not None else pipeline_stages
        self.ws = weight_stages if weight_stages is not None else pipeline_stages
        self.cs = categorical_stages if categorical_stages is not None else pipeline_stages
        self.n = child_units
        self.wm = weight_rows
        self.ww = weight_warps
        self.cn = categorical_units
        self.cw = categorical_warps
        self.alpha = alpha
        self.momentum = momentum
        self.running = tuple(torch.zeros_like(p) for p in self.parameters)
        self.step_count = torch.zeros((), dtype=torch.int32, device=source.device)
        self.bias = torch.ones((), device=source.device)
        self.one = torch.ones((), device=source.device)
        self.compiled = []

    def check_images(self, images):
        self.source.check_images(images, self.b)

    @_device_guard
    def forward_values(self, images):
        self.check_images(images)
        u, b = self.u, self.b
        p = max(16, tr.next_power_of_2(u))
        nodes = torch.empty((self.nodes, b, u), device=images.device)
        cached_x = torch.empty((self.nodes - self.f, b, u), device=images.device) if self.cache_inputs else nodes
        shifts = torch.empty((self.nodes - self.f, b), device=images.device) if self.cache_inputs else nodes
        kernels = [_leaves[(tr.cdiv(b, 16), self.f)](images, self.source.variable_ids, self.parameters[0], nodes, b, images.shape[1], self.f, self.c, u, 16, p, num_warps=4)]
        for group in self.groups:
            kernels.append(
                _forward[(tr.cdiv(b, self.m), group.count)](
                    nodes,
                    self.parameters[group.owner + 1],
                    group.routes,
                    cached_x,
                    shifts,
                    self.f,
                    self.cache_inputs,
                    b,
                    u,
                    group.outputs,
                    group.arity,
                    self.m,
                    p,
                    32,
                    self.stages,
                    self.precision,
                    num_warps=self.warps,
                    num_stages=self.stages,
                )
            )
        loss = torch.empty((), device=images.device)
        kernels.append(_root[(1,)](nodes, loss, b, u, self.root, tr.next_power_of_2(b), num_warps=4))
        self.compiled = kernels
        return loss, (nodes, cached_x, shifts)

    def loss(self, images):
        return _CircuitLoss.apply(self, images, *self.parameters)

    @torch.no_grad()
    @_device_guard
    def backward_values(self, images, state, grad, *, update, parameters=None, variables=None):
        variables = self.source.variable_ids if variables is None else variables
        parameters = self.parameters if parameters is None else tuple(parameters)
        nodes, cached_x, shifts = state
        u, b = self.u, self.b
        p = max(16, tr.next_power_of_2(u))
        flow = torch.empty_like(nodes)
        model = torch.empty((self.nodes, u), device=images.device)
        if self.cache_inputs:
            kernels = [_seed_cached[(tr.cdiv(b * u, 512),)](grad, flow, model, nodes, shifts, b, u, self.f, self.root, 512, num_warps=4)]
        else:
            kernels = [_seed[(tr.cdiv(b * u, 512),)](grad, flow, model, b, u, self.root, 512, num_warps=4)]
        if update:
            kernels.append(_clock[(1,)](self.step_count, self.bias, self.momentum, num_warps=4))
        result = tuple(torch.empty_like(x) for x in parameters) if not update else parameters
        for group in reversed(self.groups):
            if self.cache_inputs:
                x, q = cached_x, flow
            else:
                x = torch.empty((group.count, b, u), device=images.device)
                q = torch.empty_like(x)
                kernels.append(_prepare_backward[(tr.cdiv(b, 16), group.count)](nodes, flow, group.routes, x, q, b, u, group.outputs, group.arity, 16, p, num_warps=4))
            kernels.append(
                _children[(tr.cdiv(b, self.m), group.count, tr.cdiv(u, self.n))](
                    x,
                    q,
                    parameters[group.owner + 1],
                    group.routes,
                    flow,
                    model,
                    nodes,
                    shifts,
                    self.f,
                    self.cache_inputs,
                    b,
                    u,
                    group.outputs,
                    group.arity,
                    self.m,
                    self.n,
                    32,
                    self.bs,
                    self.precision,
                    update,
                    num_warps=self.warps,
                    num_stages=self.bs,
                )
            )
            kernels.append(
                _weight_backward[(tr.cdiv(group.outputs, self.wm), group.count)](
                    x,
                    q,
                    parameters[group.owner + 1],
                    group.routes,
                    result[group.owner + 1],
                    model,
                    self.running[group.owner + 1],
                    self.bias,
                    self.f,
                    self.cache_inputs,
                    b,
                    u,
                    group.outputs,
                    self.wm,
                    p,
                    32,
                    self.ws,
                    self.precision,
                    update,
                    self.alpha,
                    self.momentum,
                    num_warps=self.ww,
                    num_stages=self.ws,
                )
            )
        kernels.append(
            _categorical_backward[(tr.cdiv(u, self.cn), self.f)](
                images,
                variables,
                flow,
                parameters[0],
                result[0],
                model,
                self.running[0],
                self.bias,
                b,
                images.shape[1],
                self.f,
                self.c,
                u,
                max(16, tr.next_power_of_2(self.c)),
                self.cn,
                32,
                self.cs,
                self.precision,
                update,
                self.alpha,
                self.momentum,
                num_warps=self.cw,
                num_stages=self.cs,
            )
        )
        self.compiled += kernels
        return result

    @torch.no_grad()
    def step(self, images):
        loss, state = self.forward_values(images)
        self.backward_values(images, state, self.one, update=True)
        return loss

    def state_dict(self):
        return dict(steps=self.step_count.detach().clone(), momentum_flows=[x.detach().clone() for x in self.running])

    @torch.no_grad()
    def load_state_dict(self, state):
        if len(state["momentum_flows"]) != len(self.running):
            raise ValueError("Momentum parameter count differs")
        self.step_count.copy_(torch.as_tensor(state["steps"], device=self.source.device, dtype=torch.int32))
        for dst, src in zip(self.running, state["momentum_flows"], strict=True):
            dst.copy_(src)
