"""Independent Torch update oracle for custom training tests."""

import torch

from extended_einsum.backends.triton import Selection


class TorchAnemone:
    def __init__(self, model, batch_size, *, alpha=0.4, momentum=0.9, stability="logspace_max"):
        self.model, self.b, self.alpha, self.momentum, self.stability = model, batch_size, alpha, momentum, stability
        self.running = tuple(torch.zeros_like(p) for p in model.parameters())
        self.step_count = torch.zeros((), dtype=torch.int32, device=model.device)
        self.selection = Selection("torch", "Independent Torch test reference", {})

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



class TorchAdam:
    def __init__(self, model, batch_size):
        self.model, self.b = model, batch_size
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01, fused=model.device.type == "cuda", capturable=model.device.type == "cuda")

    def loss(self, images):
        return self.model.loss(images)

    def step(self, images):
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss(images)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)


def reference_training(model, batch_size, *, trainer):
    return TorchAdam(model, batch_size) if trainer == "adam" else TorchAnemone(model, batch_size, **(dict(alpha=1.0, momentum=0.0) if trainer == "em" else {}))
