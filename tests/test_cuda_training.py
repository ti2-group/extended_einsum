"""CUDA tree kernels against independent differentiable Torch computations."""

import copy

import pytest
import torch

from extended_einsum.compiled import UnsupportedProgramError
from extended_einsum.tree import Projection, TreeCircuit, balanced_tree, compile_training
from tests.torch_tree_reference import reference_training

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("trainer", ["adam", "em", "anemone"])
@pytest.mark.parametrize("mode", ["logspace_max", "logspace_min", "scaled_max", "scaled_min", "scaled_sum"])
def test_cpu_training_modes(trainer, mode):
    m = balanced_tree(pixels=7, units=5, categories=7, parameterization="logits" if trainer == "adam" else "normalized_log")
    x = torch.randint(7, (17, 7))
    ref = m.loss(x)
    torch.testing.assert_close(m.loss(x, stability=mode), ref)
    with pytest.raises(UnsupportedProgramError):
        compile_training(m, 17, trainer=trainer, stability=mode)


def test_topology_rejects_sharing_and_unused_parameters():
    with pytest.raises(ValueError):
        TreeCircuit(torch.zeros(2, 3, 4), [torch.zeros(1, 1, 4)], [Projection((0, 0), 0, 0, 1)])
    with pytest.raises(ValueError):
        TreeCircuit(torch.zeros(2, 3, 4), [torch.zeros(2, 1, 4)], [Projection((0, 1), 0, 0, 1)])


@CUDA
@pytest.mark.parametrize("trainer", ["adam", "em", "anemone"])
@pytest.mark.parametrize("units,batch", [(5, 17), (32, 32), (65, 33)])
def test_custom_loss_backward_updates_and_routing(trainer, units, batch):
    kwargs = dict(pixels=7, units=units, categories=7, device="cuda", parameterization="logits" if trainer == "adam" else "normalized_log")
    ref = balanced_tree(**kwargs)
    custom = copy.deepcopy(ref)
    # Extra input pixels and non-identity routes must affect both passes.
    ref.variable_ids.copy_(torch.tensor([8, 3, 1, 0, 5, 6, 2], device="cuda"))
    custom.variable_ids.copy_(ref.variable_ids)
    x = torch.randint(7, (batch, 9), device="cuda", generator=torch.Generator(device="cuda").manual_seed(941))
    a = reference_training(ref, batch, trainer=trainer)
    b = compile_training(custom, batch, trainer=trainer, backend="triton", precision="tf32x3")
    la, lb = a.loss(x), b.loss(x)
    torch.testing.assert_close(lb, la, atol=3e-5, rtol=1e-6)
    la.backward()
    lb.backward()
    for p, q in zip(ref.parameters(), custom.parameters(), strict=True):
        torch.testing.assert_close(q.grad, p.grad, atol=4e-6, rtol=5e-4)
    for _ in range(3):
        torch.testing.assert_close(b.step(x), a.step(x), atol=2e-4, rtol=0)
        if trainer == "adam":
            # Adam's epsilon=1e-8 amplifies roundoff in almost-zero gradients.
            # Compare predictions and aggregate drift, alongside the explicit
            # gradient checks above, rather than requiring every logit to match.
            torch.testing.assert_close(custom.loss(x), ref.loss(x), atol=3e-4, rtol=0)
            for p, q in zip(ref.parameters(), custom.parameters(), strict=True):
                assert (q - p).abs().mean() < 1e-3
        else:
            for p, q in zip(ref.parameters(), custom.parameters(), strict=True):
                torch.testing.assert_close(q, p, atol=8e-4, rtol=0)
        x.copy_(x.roll(1, 0))


@CUDA
def test_saved_softmax_parameters_and_arbitrary_upstream_gradient():
    model = balanced_tree(pixels=4, units=5, categories=7, device="cuda", parameterization="logits")
    reference = copy.deepcopy(model)
    custom = compile_training(model, 17, trainer="adam", backend="triton", precision="tf32x3")
    x, y = (torch.randint(7, (17, 4), device="cuda") for _ in range(2))
    (2 * custom.loss(x) - 0.3 * custom.loss(y)).backward()
    (2 * reference.loss(x) - 0.3 * reference.loss(y)).backward()
    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=4e-6, rtol=1e-3)


@CUDA
@pytest.mark.parametrize("trainer", ["adam", "anemone"])
def test_cuda_graph_updates_and_checkpoint(trainer):
    model = balanced_tree(pixels=4, units=32, categories=16, device="cuda", parameterization="logits" if trainer == "adam" else "normalized_log")
    runner = compile_training(model, 32, trainer=trainer, backend="triton", precision="tf32x3")
    x = torch.randint(16, (32, 4), device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner.step(x)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    reference = copy.deepcopy(model)
    ref = reference_training(reference, 32, trainer=trainer)
    ref.load_state_dict(copy.deepcopy(runner.state_dict()))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        loss = runner.step(x)
    for _ in range(3):
        ref.step(x)
        graph.replay()
        torch.cuda.synchronize()
        x.add_(1).remainder_(16)
    assert torch.isfinite(loss)
    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=8e-4, rtol=0)


@CUDA
def test_backward_preserves_forward_pixel_routes():
    model = balanced_tree(pixels=4, units=5, categories=7, device="cuda", parameterization="logits")
    reference = copy.deepcopy(model)
    runner = compile_training(model, 17, trainer="adam", backend="triton", precision="tf32x3")
    x = torch.randint(7, (17, 4), device="cuda")
    loss = runner.loss(x)
    expected = reference.loss(x)
    model.variable_ids = model.variable_ids.flip(0)
    loss.backward()
    expected.backward()
    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=4e-6, rtol=1e-3)
    loss = runner.loss(x)
    model.variable_ids.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@CUDA
@pytest.mark.parametrize("precision", ["tf32", "tf32x3"])
def test_float32_floor_survives_tensor_core_boundaries(precision):
    model = balanced_tree(pixels=2, units=2, categories=2, device="cuda")
    floor = torch.tensor(torch.finfo(torch.float32).tiny, device="cuda").log()
    with torch.no_grad():
        model.categorical.copy_(torch.stack([torch.stack([floor * 0, floor]), torch.stack([floor, floor * 0])]).expand(2, -1, -1))
        model.weights[0].copy_(torch.stack([torch.stack([floor * 0, floor]), torch.stack([floor, floor * 0])]).expand(2, -1, -1))
        model.weights[-1].copy_(torch.stack([floor, floor * 0]).reshape(1, 1, 2))
    reference = copy.deepcopy(model).double()
    x = torch.zeros((17, 2), dtype=torch.uint8, device="cuda")
    runner = compile_training(model, 17, backend="triton", precision=precision)
    loss = runner.loss(x)
    torch.testing.assert_close(loss.double(), reference.loss(x), atol=3e-5, rtol=0)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.isfinite(runner.step(x))
    assert all(torch.isfinite(p).all() for p in model.parameters())
