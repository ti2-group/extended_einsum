"""Direct, unshared categorical trees for optional custom CUDA training.

Topology is independent of Cirkit. Parameters are either normalized *log*
probabilities (EM/Anemone) or unconstrained logits (softmax Adam).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from extended_einsum.cuda import MODES, Selection


@dataclass(frozen=True)
class Projection:
    """Projection from one leaf/node or a product of two nodes.

    Node IDs: categorical leaves first, then projections in topological order.
    ``offset`` is an element offset in ``weights[owner]`` (output,input order).
    Each parameter element and each node must be used exactly once.
    """

    children: tuple[int, ...]
    owner: int
    offset: int
    outputs: int


class TreeCircuit(torch.nn.Module):
    def __init__(self, categorical, weights, projections, *, parameterization="normalized_log", variable_ids=None):
        super().__init__()
        if parameterization not in ("normalized_log", "logits"):
            raise ValueError("parameterization must be normalized_log or logits")
        if categorical.ndim != 3 or min(categorical.shape) < 1:
            raise ValueError("Expected categorical[leaf,category,unit]")
        self.num_variables, self.categories, self.units = categorical.shape
        self.projections = tuple(projections)
        if not self.projections:
            raise ValueError("At least one projection is required")
        parameters = (categorical, *weights)
        if any(p.dtype != torch.float32 or p.device != categorical.device or not p.is_contiguous() or p.numel() >= 2**31 for p in parameters):
            raise ValueError("Requires contiguous FP32 parameters on one device with 32-bit indexing")
        ranges = sorted((p.data_ptr(), p.data_ptr() + p.numel() * p.element_size()) for p in parameters)
        if any(end > next_start for (_, end), (next_start, _) in zip(ranges, ranges[1:])):
            raise ValueError("Aliased parameter storage requires flow accumulation")
        parents = [0] * (self.num_variables + len(self.projections))
        intervals = [[] for _ in weights]
        widths = [self.units] * self.num_variables
        for node, p in enumerate(self.projections, self.num_variables):
            if len(p.children) not in (1, 2) or len(set(p.children)) != len(p.children) or any(c < 0 or c >= node or widths[c] != self.units for c in p.children):
                raise ValueError("Requires a topologically ordered, unshared binary tree with equal hidden units")
            if p.outputs not in (1, self.units) or (p.outputs != self.units and node != len(parents) - 1):
                raise ValueError("Only the root may have one output")
            if not 0 <= p.owner < len(weights) or p.offset < 0 or p.offset + p.outputs * self.units > weights[p.owner].numel():
                raise ValueError("Projection parameter range is out of bounds")
            intervals[p.owner].append((p.offset, p.offset + p.outputs * self.units))
            for c in p.children:
                parents[c] += 1
            widths.append(p.outputs)
        if widths[-1] != 1 or parents != [1] * (len(parents) - 1) + [0]:
            raise ValueError("Requires one scalar root and exactly one parent per other node")
        for w, ranges in zip(weights, intervals, strict=True):
            end = 0
            for a, b in sorted(ranges):
                if a != end:
                    raise ValueError("Shared or unused parameter ranges are unsupported")
                end = b
            if end != w.numel():
                raise ValueError("All weight elements must be used")
        if any(w.ndim not in (2, 3) or w.shape[-1] != self.units for w in weights):
            raise ValueError("Weights must store [fold, output, input] or [output, input]")
        self.parameterization = parameterization
        self.categorical = torch.nn.Parameter(categorical)
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(w) for w in weights])
        if variable_ids is None:
            variable_ids = torch.arange(self.num_variables, device=categorical.device)
        if variable_ids.shape != (self.num_variables,) or variable_ids.dtype not in (torch.int32, torch.int64) or variable_ids.device != categorical.device:
            raise ValueError("Expected integer variable_ids[leaf] on the parameter device")
        self.register_buffer("variable_ids", variable_ids.contiguous())
        # Python topology is kept on CPU. Build vectorized groups once.
        depths = {i: 0 for i in range(self.num_variables)}
        groups = {}
        for node, p in enumerate(self.projections, self.num_variables):
            depth = max(depths[c] for c in p.children) + 1
            depths[node] = depth
            groups.setdefault((depth, p.owner, p.outputs, len(p.children)), []).append((node, p))
        self.groups = tuple(tuple(rows) for _, rows in sorted(groups.items()))

        references = {i: (0, i) for i in range(self.num_variables)}
        self._routes = []
        for gi, group in enumerate(self.groups):
            terms = []
            for side in range(len(group[0][1].children)):
                sources = tuple(sorted({references[p.children[side]][0] for _, p in group}))
                starts = {}
                total = 0
                for source in sources:
                    starts[source] = total
                    total += self.num_variables if source == 0 else len(self.groups[source - 1])
                indices = [starts[references[p.children[side]][0]] + references[p.children[side]][1] for _, p in group]
                name = f"_route_{gi}_{side}"
                self.register_buffer(name, torch.tensor(indices, dtype=torch.long, device=self.device))
                terms.append((sources, name))
            self._routes.append(tuple(terms))
            for row, (node, _) in enumerate(group):
                references[node] = (gi + 1, row)

    @property
    def device(self):
        return self.categorical.device

    def log_parameters(self):
        if self.parameterization == "normalized_log":
            return (self.categorical,) + tuple(self.weights[i] for i in range(len(self.weights)))
        return (self.categorical.log_softmax(1), *(w.log_softmax(-1) for w in self.weights))

    def check_images(self, images, batch=None):
        if (
            images.ndim != 2
            or images.device != self.device
            or images.dtype not in (torch.uint8, torch.int32, torch.int64)
            or not images.is_contiguous()
            or images.numel() >= 2**31
            or (batch is not None and images.shape[0] != batch)
        ):
            raise ValueError("Expected contiguous integer pixels[batch,variables] on the model device")
        if images.shape[1] < 1 or images.shape[0] < 1:
            raise ValueError("Empty evidence is unsupported")

    def loss(self, images, *, stability="logspace_max"):
        self.check_images(images)
        logs = self.log_parameters()
        pixels = images.index_select(1, self.variable_ids.long()).T.long()
        leaf = logs[0][torch.arange(self.num_variables, device=self.device)[:, None], pixels]
        return -self._evaluate(leaf, logs[1:], stability).mean()

    def _evaluate(self, leaf, logs, stability):
        if stability not in MODES:
            raise ValueError("Unsupported stability mode")
        scaled = stability.startswith("scaled")

        def normalize(x, scale):
            if stability == "scaled_sum":
                norm = x.sum(-1, keepdim=True)
            elif stability == "scaled_min":
                norm = x.amin(-1, keepdim=True)
            else:
                norm = x.amax(-1, keepdim=True)
            safe = torch.where(norm > 0, norm, 1).detach()
            return x / safe, scale + safe.log()

        values = [leaf]
        scales = []
        if scaled:
            shift = leaf.amax(-1, keepdim=True).detach()
            shift = torch.where(torch.isfinite(shift), shift, 0)
            values[0], shift = normalize((leaf - shift).exp(), shift)
            scales.append(shift)

        def route(frontiers, terms):
            sources, name = terms
            value = frontiers[sources[0]] if len(sources) == 1 else torch.cat([frontiers[s] for s in sources], 0)
            return value.index_select(0, getattr(self, name))

        for group, terms in zip(self.groups, self._routes, strict=True):
            z = route(values, terms[0])
            if len(terms) == 2:
                right = route(values, terms[1])
                z = z * right if scaled else z + right
            first = group[0][1]
            weight = logs[first.owner]
            if len(group) * first.outputs * self.units == weight.numel() and all(p.offset == i * first.outputs * self.units for i, (_, p) in enumerate(group)):
                w = weight.reshape(len(group), first.outputs, self.units).exp()
            else:
                w = torch.stack([weight.reshape(-1)[p.offset : p.offset + p.outputs * self.units].reshape(p.outputs, self.units) for _, p in group]).exp()
            if scaled:
                scale = route(scales, terms[0])
                if len(terms) == 2:
                    scale = scale + route(scales, terms[1])
                z, scale = normalize(z, scale)
                y, scale = normalize(torch.bmm(z, w.transpose(1, 2)), scale)
                scales.append(scale)
            else:
                shift = (z.amin(-1, keepdim=True) if stability == "logspace_min" else z.amax(-1, keepdim=True)).detach()
                shift = torch.where(torch.isfinite(shift), shift, 0)
                y = torch.bmm((z - shift).exp(), w.transpose(1, 2)).log() + shift
            values.append(y)
        return values[-1][0].log() + scales[-1][0] if scaled else values[-1][0]

    def partition(self, *, stability="logspace_max"):
        logs = self.log_parameters()
        leaf = logs[0].logsumexp(1).unsqueeze(1)
        return self._evaluate(leaf, logs[1:], stability).sum()


def balanced_tree(*, pixels, units, categories=256, device="cpu", parameterization="normalized_log", seed=42, project_leaves=True):
    """Build a CP/CP-T tree for any pixel count, without padding to a power of two."""
    if pixels < 2 or units < 1 or categories < 1:
        raise ValueError("Requires at least two pixels and positive units/categories")
    gen = torch.Generator(device=device).manual_seed(seed)
    cat = torch.randn(pixels, categories, units, device=device, generator=gen)
    if parameterization == "normalized_log":
        cat = cat.log_softmax(1)
    projections = []
    weights = []
    current = list(range(pixels))
    while len(current) > 1 or (project_leaves and not projections):
        children = [(c,) for c in current] if project_leaves and not projections else [tuple(current[i : i + 2]) for i in range(0, len(current), 2)]
        outputs = 1 if len(children) == 1 else units
        w = torch.randn(len(children), outputs, units, device=device, generator=gen)
        if parameterization == "normalized_log":
            w = w.log_softmax(-1)
        owner = len(weights)
        weights.append(w)
        current = []
        for i, terms in enumerate(children):
            current.append(pixels + len(projections))
            projections.append(Projection(terms, owner, i * outputs * units, outputs))
    return TreeCircuit(cat, weights, projections, parameterization=parameterization)


class TorchAnemone:
    def __init__(self, model, batch_size, *, alpha=0.4, momentum=0.9, stability="logspace_max"):
        self.model, self.b, self.alpha, self.momentum, self.stability = model, batch_size, alpha, momentum, stability
        self.running = tuple(torch.zeros_like(p) for p in model.parameters())
        self.step_count = torch.zeros((), dtype=torch.int32, device=model.device)
        self.selection = Selection("torch", "Torch reference/fallback for this stability, device, or topology", {})

    def loss(self, images):
        self.model.check_images(images, self.b)
        return self.model.loss(images, stability=self.stability)

    def step(self, images):
        params = tuple(self.model.parameters())
        loss = self.loss(images)
        data = torch.autograd.grad(loss, params)
        model = torch.autograd.grad(self.model.partition(stability=self.stability), params)
        self._apply_update(params, data, model)
        return loss.detach()

    def _apply_update(self, params, data, model):
        with torch.no_grad():
            self.step_count.add_(1)
            bias = 1 - self.momentum**self.step_count
            for p, g, m, r in zip(params, data, model, self.running, strict=True):
                r.mul_(self.momentum).add_(g, alpha=-(1 - self.momentum))
                numerator = ((1 - self.alpha) * m + self.alpha * r / bias).clamp_min(torch.finfo(p.dtype).tiny)
                axis = 1 if p is self.model.categorical else -1
                log = numerator.log()
                p.copy_(log - log.logsumexp(axis, keepdim=True))

    def state_dict(self):
        return dict(steps=self.step_count.detach().clone(), momentum_flows=[x.detach().clone() for x in self.running])

    @torch.no_grad()
    def load_state_dict(self, state):
        if len(state["momentum_flows"]) != len(self.running) or any(x.shape != y.shape for x, y in zip(self.running, state["momentum_flows"], strict=True)):
            raise ValueError("Momentum parameter metadata differs")
        self.step_count.copy_(state["steps"])
        for x, y in zip(self.running, state["momentum_flows"], strict=True):
            x.copy_(y)


class AdamTrainer:
    def __init__(self, model, batch_size, *, runner=None, stability="logspace_max", lr=0.01):
        self.model, self.b, self.runner, self.stability = model, batch_size, runner, stability
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=model.device.type == "cuda", capturable=model.device.type == "cuda")
        self.selection = runner.selection if runner else Selection("torch", "Torch softmax Adam fallback", {})

    def loss(self, images):
        self.model.check_images(images, self.b)
        if self.runner is None:
            return self.model.loss(images, stability=self.stability)
        # Normalization and its Jacobian use Torch's optimized log_softmax.
        # Custom circuit backward returns log-probability gradients; autograd
        # applies the exact log_softmax chain rule to the original logits.
        self.runner.parameters = tuple(self.model.log_parameters())
        return self.runner.loss(images)

    def step(self, images):
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss(images)
        loss.backward()
        self.optimizer.step()
        if self.runner is not None:
            self.runner.parameters = tuple(p.detach() for p in self.runner.parameters)
        return loss.detach()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)


def compile_training(model, batch_size, *, trainer="anemone", stability="logspace_max", backend="auto", precision="tf32x3", alpha=0.4, momentum=0.9, lr=0.01):
    """Choose custom circuit forward/backward or a differentiable Torch fallback.

    EM is Anemone with alpha=1 and momentum=0. Normalized log weights must
    already be normalized by the caller; they are never softmaxed per step.
    Scaled training retains scaled arithmetic through the Torch fallback.
    """
    if trainer not in ("adam", "em", "anemone") or backend not in ("auto", "torch", "triton") or stability not in MODES or precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("Unsupported training setup")
    if batch_size < 1 or not 0 <= alpha <= 1 or not 0 <= momentum < 1:
        raise ValueError("Invalid batch size or Anemone coefficients")
    if (trainer == "adam") != (model.parameterization == "logits"):
        raise ValueError("Adam requires logits; EM/Anemone require normalized_log parameters")
    if trainer == "em":
        alpha, momentum = 1.0, 0.0
    custom = model.device.type == "cuda" and stability == "logspace_max"
    if backend == "triton" and not custom:
        raise ValueError("Custom training requires CUDA and logspace_max; scaled training uses Torch")
    selected = backend == "triton"
    if backend == "auto" and custom:
        from importlib.util import find_spec

        selected = bool(find_spec("triton")) and torch.cuda.get_device_capability(model.device) == (8, 6) and model.units in (32, 256) and batch_size in (128, 256) and precision == "tf32"
    runner = None
    if selected:
        from extended_einsum.backends.triton_training import TritonAnemone

        options = dict(block_size=32, num_warps=8, pipeline_stages=2, weight_rows=32, categorical_stages=1, cache_inputs=True)
        options.update(dict(child_units=128, weight_warps=16, categorical_units=64) if model.units >= 128 else dict(child_units=32))
        runner = TritonAnemone(model, batch_size, precision=precision, alpha=alpha, momentum=momentum, **options)
        runner.selection = Selection("triton", "Custom tree forward/backward; fused normalized updates or Torch softmax + fused Adam", options)
    if trainer == "adam":
        return AdamTrainer(model, batch_size, runner=runner, stability=stability, lr=lr)
    return runner if runner else TorchAnemone(model, batch_size, alpha=alpha, momentum=momentum, stability=stability)
