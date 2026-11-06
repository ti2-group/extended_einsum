"""Cirkit input parity including raw routing, parameters and input gradients."""

import copy

import pytest
import torch

pytest.importorskip("cirkit")
from cirkit.backend.torch.graph.modules import FoldIndexInfo
from cirkit.backend.torch.layers.input import (
    TorchBinomialLayer,
    TorchCategoricalLayer,
    TorchConstantValueLayer,
    TorchEmbeddingLayer,
    TorchEvidenceLayer,
    TorchGaussianLayer,
    TorchPolynomialLayer,
)
from cirkit.backend.torch.parameters.nodes import TorchTensorParameter
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import ComplexLSESumSemiring, LSESumSemiring, SumProductSemiring

from extended_einsum.compiled import UnsupportedProgramError
from extended_einsum.inputs import compile_input


def parameter(value):
    node = TorchTensorParameter(*value.shape[1:], num_folds=value.shape[0])
    info = FoldIndexInfo([node], {0: [[] for _ in range(value.shape[0])]}, [(0, i) for i in range(value.shape[0])])
    p = TorchParameter([node], {}, [node], fold_idx_info=info)
    p.reset_parameters()
    p.to(value.device)
    with torch.no_grad():
        next(p.parameters()).copy_(value)
    return p


def make_layer(kind, device="cpu", units=5, folds=3, log=True, categories=7):
    gen = torch.Generator(device=device).manual_seed(52)

    def rand(*shape):
        return torch.rand(*shape, generator=gen, device=device) + 0.1

    scope = (torch.arange(folds, device=device) * 2 % 5).reshape(-1, 1)
    semiring = LSESumSemiring if log else SumProductSemiring
    args = dict(scope_idx=scope, num_output_units=units, semiring=semiring)
    if kind.startswith("categorical"):
        data = rand(folds, units, categories)
        key = "probs" if kind.endswith("probs") else "logits"
        return TorchCategoricalLayer(**args, num_categories=categories, **{key: parameter(data.softmax(-1) if key == "probs" else data)})
    if kind.startswith("binomial"):
        key = "probs" if kind.endswith("probs") else "logits"
        return TorchBinomialLayer(**args, total_count=categories, **{key: parameter(rand(folds, units).sigmoid() if key == "probs" else rand(folds, units))})
    if kind == "embedding":
        return TorchEmbeddingLayer(**args, num_states=categories, weight=parameter(rand(folds, units, categories)))
    if kind.startswith("gaussian"):
        return TorchGaussianLayer(
            **args, mean=parameter(rand(folds, units)), stddev=parameter(rand(folds, units) + 0.3), log_partition=parameter(rand(folds, units)) if kind.endswith("partition") else None
        )
    if kind == "polynomial":
        return TorchPolynomialLayer(**args, degree=3, coeff=parameter(rand(folds, units, 4)))
    if kind == "constant":
        return TorchConstantValueLayer(units, value=parameter(rand(folds, units)), log_space=not log, semiring=semiring)
    inner = make_layer(kind.removeprefix("evidence_"), device, units, folds, log, categories)
    return TorchEvidenceLayer(inner, observation=parameter(torch.ones(folds, 1, device=device)), semiring=semiring)


KINDS = [
    "categorical_logits",
    "categorical_probs",
    "embedding",
    "binomial_logits",
    "binomial_probs",
    "gaussian",
    "gaussian_partition",
    "polynomial",
    "constant",
    "evidence_gaussian",
    "evidence_categorical_logits",
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("log", [False, True])
def test_cpu_input_is_rejected(kind, log):
    layer = make_layer(kind, log=log)
    data = torch.ones(17, 5)
    program = compile_input(layer)
    with pytest.raises(UnsupportedProgramError):
        program(data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("layout", ["units", "batch"])
def test_custom_forward_and_backward(kind, log, layout):
    layer = make_layer(kind, "cuda", log=log)
    reference = copy.deepcopy(layer)
    discrete = kind.startswith(("categorical", "binomial", "embedding", "evidence"))
    data = (torch.arange(17 * 5, device="cuda").reshape(17, 5) % 7).float() if discrete else torch.rand(17, 5, device="cuda")
    data.requires_grad_(not discrete)
    other = data.detach().clone().requires_grad_(not discrete)
    program = compile_input(layer, backend="triton", output_layout=layout)
    expected = compile_input(reference).reference(other)
    actual = program(data)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=2e-5)
    upstream = torch.linspace(-0.7, 1.3, actual.numel(), device="cuda").reshape(actual.shape)
    actual.backward(upstream)
    expected.backward(upstream)
    for p, q in zip(layer.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=4e-5, rtol=5e-4)
    if data.grad is not None:
        torch.testing.assert_close(data.grad, other.grad, atol=4e-5, rtol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lookup_packed_layout_and_dynamic_parameters():
    layer = make_layer("categorical_logits", "cuda", units=65, folds=4, categories=256)
    x = torch.randint(256, (33, 5), device="cuda")
    runner = compile_input(layer, backend="triton", table_layout="units")
    for _ in range(2):
        torch.testing.assert_close(runner(x), runner.reference(x))
        with torch.no_grad():
            next(layer.parameters()).add_(0.1)
            x.add_(1).remainder_(256)
    runner(x).sum().backward()
    assert torch.isfinite(next(layer.parameters()).grad).all()


def test_complex_input_is_rejected():
    layer = make_layer("embedding", log=False)
    layer.semiring = ComplexLSESumSemiring
    with torch.no_grad():
        next(layer.parameters()).neg_()
    runner = compile_input(layer)
    x = torch.ones(17, 5)
    with pytest.raises(UnsupportedProgramError):
        runner(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("replace", [False, True])
def test_parameter_fold_routing_remains_live(replace):
    layer = make_layer("categorical_logits", "cuda")
    runner = compile_input(layer, backend="triton")
    param = layer.params["logits"]
    target = param.address_book._entry_in_fold_idx_targets[-1][0]
    indices = getattr(param.address_book, target)
    if replace:
        setattr(param.address_book, target, indices.flip(0))
    else:
        indices.copy_(indices.flip(0))
    x = torch.randint(7, (17, 5), device="cuda")
    torch.testing.assert_close(runner(x), runner.reference(x))
    actual = torch.autograd.grad(runner(x).sum(), tuple(layer.parameters()))
    expected = torch.autograd.grad(runner.reference(x).sum(), tuple(layer.parameters()))
    for p, q in zip(actual, expected, strict=True):
        torch.testing.assert_close(p, q)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["categorical_logits", "binomial_probs", "embedding"])
def test_frozen_discrete_input_does_not_create_data_gradients(kind):
    layer = make_layer(kind, "cuda").requires_grad_(False)
    data = torch.ones(17, 5, device="cuda", requires_grad=True)
    runner = compile_input(layer, backend="triton")
    assert not runner(data).requires_grad
    assert not runner.reference(data).requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["gaussian", "binomial_probs", "polynomial", "categorical_logits", "evidence_gaussian", "constant"])
def test_input_graph_replay_uses_changing_data_and_parameters(kind):
    layer = make_layer(kind, "cuda", units=32, categories=255)
    runner = compile_input(layer, backend="triton")
    data = torch.ones(32, 5, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner(data).sum().backward()
        layer.zero_grad(set_to_none=True)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer.zero_grad(set_to_none=True)
        out = runner(data)
        out.sum().backward()
    for _ in range(2):
        graph.replay()
        torch.testing.assert_close(out, runner.reference(data), atol=3e-4, rtol=2e-4)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in layer.parameters())
        with torch.no_grad():
            next(layer.parameters()).add_(0.01)
            data.add_(0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_construct_input_in_inference_mode():
    with torch.inference_mode():
        layer = make_layer("gaussian", "cuda")
        data = torch.rand(17, 5, device="cuda")
        runner = compile_input(layer, backend="triton")
        torch.testing.assert_close(runner(data), runner.reference(data), atol=3e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("count", [0, 255])
@pytest.mark.parametrize("parameterization", ["logits", "probs"])
def test_binomial_trial_and_probability_boundaries(count, parameterization):
    layer = make_layer("binomial_" + parameterization, "cuda", categories=count)
    if parameterization == "probs":
        with torch.no_grad():
            next(layer.parameters()).copy_(torch.tensor([0.0, 1.0, 0.5, 0.0, 1.0], device="cuda").expand(3, -1))
    data = torch.tensor([0, count // 2, count], device="cuda").repeat(29)[:85].reshape(17, 5)
    runner = compile_input(layer, backend="triton")
    result, reference = runner(data), runner.reference(data)
    torch.testing.assert_close(result, reference, atol=3e-4, rtol=3e-5)
    actual_grad = torch.autograd.grad(result.sum(), tuple(layer.parameters()))
    reference_grad = torch.autograd.grad(reference.sum(), tuple(layer.parameters()))
    for p, q in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(p, q, atol=4e-4, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["categorical_logits", "embedding", "binomial_probs", "gaussian", "polynomial"])
def test_input_gradients_through_extended_einsum(kind):
    import extended_einsum as xe

    layer = make_layer(kind, "cuda")
    data = torch.ones(17, 5, device="cuda")
    runner = compile_input(layer, backend="triton")
    scores = runner(data)
    weights = torch.rand(3, 5, 5, device="cuda").softmax(-1)
    expression = xe.log(xe.einsum("fbi,foi->fbo", xe.exp(xe.array(scores)), xe.array(weights)))
    loss = expression.materialize(stability_mode="logspace_max").backend_array.sum()
    expected_scores = runner.reference(data)
    expected = torch.logsumexp(expected_scores[:, :, None, :] + weights[:, None].log(), dim=-1).sum()
    torch.testing.assert_close(loss, expected, atol=1e-4, rtol=1e-5)
    actual_grad = torch.autograd.grad(loss, tuple(layer.parameters()))
    reference_grad = torch.autograd.grad(expected, tuple(layer.parameters()))
    for p, q in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(p, q, atol=5e-5, rtol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["categorical_probs", "embedding"])
def test_tiny_lookup_flows_survive_probability_division(kind):
    layer = make_layer(kind, "cuda")
    tiny = torch.finfo(torch.float32).tiny
    with torch.no_grad():
        values = next(layer.parameters())
        values.fill_(1 / 6)
        values[:, :, 1] = tiny
    data = torch.ones(17, 5, dtype=torch.int64, device="cuda")
    runner = compile_input(layer, backend="triton")
    result, reference = runner(data), runner.reference(data)
    upstream = torch.full_like(result, tiny / 16)
    actual_grad = torch.autograd.grad(result, tuple(layer.parameters()), upstream)
    reference_grad = torch.autograd.grad(reference, tuple(layer.parameters()), upstream)
    for p, q in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(p, q, atol=1e-5, rtol=1e-5)
