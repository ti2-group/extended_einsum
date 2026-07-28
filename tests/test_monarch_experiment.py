import numpy as np
import pytest
import torch
from cirkit.backend.torch.parameters.nodes import TorchTensorParameter
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import LSESumSemiring, SumProductSemiring
from cirkit.pipeline import PipelineContext
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import ConstantTensorInitializer
from cirkit.symbolic.layers import CategoricalLayer
from cirkit.symbolic.parameters import Parameter as CirkitParameter
from cirkit.symbolic.parameters import TensorParameter
from cirkit.utils.scope import Scope

import extended_einsum.interface as xe
import extended_einsum.preprocess as preprocess
from experiments.monarch.benchmark import (
    MEASURED_BATCHES,
    Configuration,
    Scale,
    append_raw_rows,
    append_row,
    base_row,
    completed_keys,
)
from experiments.monarch.cirkit import (
    CompactMixingSumLayer,
    MonarchSumLayer,
    TorchCompactMixingSumLayer,
    TorchMonarchSumLayer,
    make_monarch_parameter_from_logits,
    materialize_monarch_matrix,
    register_monarch_compilation,
)
from experiments.monarch.model import (
    build_symbolic_circuit,
    canonicalize_parameters,
    setup_cirkit,
    setup_xe,
    translate_to_xe,
)
from experiments.monarch.xe import factor_shapes, transform
from extended_einsum.interface.tensor_expression import Parameter
from extended_einsum.preprocess import FoldSameShapedOperations


def _torch_parameter(values: torch.Tensor) -> TorchParameter:
    values = values.detach().clone()

    def initialize(target: torch.Tensor) -> torch.Tensor:
        return target.copy_(values.unsqueeze(0))

    node = TorchTensorParameter(
        *values.shape,
        dtype=values.dtype,
        initializer_=initialize,
    )
    parameter = TorchParameter.from_input(node)
    parameter.reset_parameters()
    return parameter


def _categorical_probs(num_units: int) -> CirkitParameter:
    probabilities = np.full((num_units, 2), 0.5, dtype=np.float32)
    return CirkitParameter.from_input(
        TensorParameter(
            num_units,
            2,
            initializer=ConstantTensorInitializer(probabilities),
        )
    )


def _parallel_monarch_circuit(p: int, q: int) -> Circuit:
    units = p * q
    layers = []
    in_layers = {}
    outputs = []
    for variable in range(2):
        input_layer = CategoricalLayer(
            Scope([variable]),
            units,
            num_categories=2,
            probs=_categorical_probs(units),
        )
        generator = np.random.default_rng(100 + variable)
        monarch_layer = MonarchSumLayer(
            units,
            units,
            p=p,
            q=q,
            factor_a=make_monarch_parameter_from_logits(
                generator.normal(size=(q, p, p)).astype(np.float32)
            ),
            factor_b=make_monarch_parameter_from_logits(
                generator.normal(size=(p, q, q)).astype(np.float32)
            ),
        )
        layers.extend((input_layer, monarch_layer))
        in_layers[monarch_layer] = [input_layer]
        outputs.append(monarch_layer)
    return Circuit(layers, in_layers, outputs)


def _compile(circuit: Circuit, *, fold: bool):
    context = PipelineContext(
        backend="torch",
        semiring="lse-sum",
        fold=fold,
        optimize=False,
    )
    register_monarch_compilation(context)
    return context.compile(circuit)


def test_symbolic_monarch_layer_has_only_independent_factor_storage() -> None:
    p, q = 2, 3
    units = p * q
    layer = MonarchSumLayer(units, units, p=p, q=q)

    assert not hasattr(layer, "weight")
    assert set(layer.params) == {"factor_a", "factor_b"}
    assert layer.factor_a.shape == (q, p, p)
    assert layer.factor_b.shape == (p, q, q)
    assert sum(np.prod(parameter.shape) for parameter in layer.params.values()) == (
        units * (p + q)
    )
    assert all(parameter.shape != (units, units) for parameter in layer.params.values())


@pytest.mark.parametrize(
    ("semiring", "log_space"),
    [(SumProductSemiring, False), (LSESumSemiring, True)],
)
def test_torch_monarch_matches_dense_oracle_outputs_and_gradients(
    semiring,
    log_space: bool,
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(7)
    dtype = torch.float64
    p, q = 2, 3
    units = p * q
    factor_a_values = torch.randn((q, p, p), generator=generator, dtype=dtype)
    factor_b_values = torch.randn((p, q, q), generator=generator, dtype=dtype)
    factor_a = _torch_parameter(factor_a_values)
    factor_b = _torch_parameter(factor_b_values)
    layer = TorchMonarchSumLayer(
        units,
        units,
        p=p,
        q=q,
        factor_a=factor_a,
        factor_b=factor_b,
        semiring=semiring,
    )
    probabilities = torch.rand((1, 5, units), generator=generator, dtype=dtype) + 0.2
    inputs = probabilities.log() if log_space else probabilities
    actual_inputs = inputs.clone().requires_grad_(True)
    actual = layer(actual_inputs.unsqueeze(1))

    oracle_inputs = inputs.clone().requires_grad_(True)
    oracle_a = factor_a_values.unsqueeze(0).clone().requires_grad_(True)
    oracle_b = factor_b_values.unsqueeze(0).clone().requires_grad_(True)
    matrix = materialize_monarch_matrix(oracle_a, oracle_b)
    if log_space:
        expected = torch.logsumexp(
            oracle_inputs.unsqueeze(-2) + matrix.log().unsqueeze(1),
            dim=-1,
        )
    else:
        expected = torch.einsum("fbi,foi->fbo", oracle_inputs, matrix)

    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual_inputs.grad, oracle_inputs.grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        next(factor_a.parameters()).grad,
        oracle_a.grad,
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        next(factor_b.parameters()).grad,
        oracle_b.grad,
        rtol=1e-10,
        atol=1e-10,
    )

    normalized_a, normalized_b = layer.normalized_factors()
    torch.testing.assert_close(
        normalized_a.sum(dim=-1),
        torch.ones_like(normalized_a[..., 0]),
    )
    torch.testing.assert_close(
        normalized_b.sum(dim=-1),
        torch.ones_like(normalized_b[..., 0]),
    )


def test_one_float64_adam_update_matches_the_dense_oracle() -> None:
    generator = torch.Generator(device="cpu").manual_seed(11)
    dtype = torch.float64
    p, q = 2, 3
    units = p * q
    factor_a_values = torch.randn((q, p, p), generator=generator, dtype=dtype)
    factor_b_values = torch.randn((p, q, q), generator=generator, dtype=dtype)
    factor_a = _torch_parameter(factor_a_values)
    factor_b = _torch_parameter(factor_b_values)
    layer = TorchMonarchSumLayer(
        units,
        units,
        p=p,
        q=q,
        factor_a=factor_a,
        factor_b=factor_b,
        semiring=SumProductSemiring,
    )
    layer_a = next(factor_a.parameters())
    layer_b = next(factor_b.parameters())
    oracle_a = torch.nn.Parameter(factor_a_values.unsqueeze(0).clone())
    oracle_b = torch.nn.Parameter(factor_b_values.unsqueeze(0).clone())
    actual_optimizer = torch.optim.Adam((layer_a, layer_b), lr=3e-3)
    oracle_optimizer = torch.optim.Adam((oracle_a, oracle_b), lr=3e-3)
    inputs = torch.rand((1, 4, units), generator=generator, dtype=dtype) + 0.1

    actual_loss = layer(inputs.unsqueeze(1)).square().mean()
    oracle_matrix = materialize_monarch_matrix(oracle_a, oracle_b)
    expected_loss = torch.einsum("fbi,foi->fbo", inputs, oracle_matrix).square().mean()
    actual_loss.backward()
    expected_loss.backward()
    actual_optimizer.step()
    oracle_optimizer.step()

    torch.testing.assert_close(layer_a, oracle_a, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(layer_b, oracle_b, rtol=1e-12, atol=1e-12)


def test_compact_quad_graph_mixing_matches_the_lse_oracle() -> None:
    generator = torch.Generator(device="cpu").manual_seed(19)
    dtype = torch.float64
    units, arity, batch_size = 5, 3, 4
    logits_values = torch.randn((units, arity), generator=generator, dtype=dtype)
    logits = _torch_parameter(logits_values)
    layer = TorchCompactMixingSumLayer(
        units,
        units,
        arity=arity,
        mixing_logits=logits,
        semiring=LSESumSemiring,
    )
    probabilities = (
        torch.rand(
            (1, arity, batch_size, units),
            generator=generator,
            dtype=dtype,
        )
        + 0.2
    )
    inputs = probabilities.log()
    actual_inputs = inputs.clone().requires_grad_(True)
    actual = layer(actual_inputs)
    oracle_inputs = inputs.clone().requires_grad_(True)
    oracle_logits = logits_values.unsqueeze(0).clone().requires_grad_(True)
    weights = torch.softmax(oracle_logits, dim=-1)
    expected = torch.logsumexp(
        oracle_inputs + weights.permute(0, 2, 1).unsqueeze(2).log(),
        dim=1,
    )
    actual.square().sum().backward()
    expected.square().sum().backward()

    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(actual_inputs.grad, oracle_inputs.grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        next(logits.parameters()).grad,
        oracle_logits.grad,
        rtol=1e-10,
        atol=1e-10,
    )
    assert CompactMixingSumLayer(units, units, arity=arity).mixing_logits.shape == (
        units,
        arity,
    )


def test_native_folding_and_torch_compile_preserve_outputs_gradients_and_storage() -> None:
    circuit = _parallel_monarch_circuit(p=2, q=2)
    eager = _compile(circuit, fold=False)
    folded = _compile(circuit, fold=True)
    compiled_source = _compile(circuit, fold=True)
    compiled = torch.compile(compiled_source, backend="eager", fullgraph=True)
    data = torch.tensor([[0, 1], [1, 0], [1, 1]], dtype=torch.int64)

    eager_output = eager(data)
    folded_output = folded(data)
    compiled_output = compiled(data)
    eager_output.square().sum().backward()
    folded_output.square().sum().backward()
    compiled_output.square().sum().backward()

    torch.testing.assert_close(folded_output, eager_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(compiled_output, eager_output, rtol=1e-5, atol=1e-6)
    eager_layers = [
        layer for layer in eager.layers if isinstance(layer, TorchMonarchSumLayer)
    ]
    folded_layer = next(
        layer for layer in folded.layers if isinstance(layer, TorchMonarchSumLayer)
    )
    compiled_layer = next(
        layer
        for layer in compiled_source.layers
        if isinstance(layer, TorchMonarchSumLayer)
    )
    eager_a_grad = torch.cat(
        [next(layer.factor_a.parameters()).grad for layer in eager_layers]
    )
    eager_b_grad = torch.cat(
        [next(layer.factor_b.parameters()).grad for layer in eager_layers]
    )
    folded_a = next(folded_layer.factor_a.parameters())
    folded_b = next(folded_layer.factor_b.parameters())
    torch.testing.assert_close(folded_a.grad, eager_a_grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(folded_b.grad, eager_b_grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        next(compiled_layer.factor_a.parameters()).grad,
        eager_a_grad,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        next(compiled_layer.factor_b.parameters()).grad,
        eager_b_grad,
        rtol=2e-5,
        atol=2e-6,
    )
    assert folded_layer.num_folds == 2
    assert folded_a.untyped_storage().data_ptr() != folded_b.untyped_storage().data_ptr()
    assert 0 not in folded_a.stride()
    assert 0 not in folded_b.stride()
    assert all(
        tuple(parameter.shape[-2:]) != (4, 4)
        for parameter in folded_layer.parameters()
    )


def test_cirkit_and_xe_monarch_match_the_materialized_matrix() -> None:
    generator = torch.Generator(device="cpu").manual_seed(31)
    dtype = torch.float64
    p, q = 2, 3
    units = p * q
    inputs = torch.rand((5, p, q), generator=generator, dtype=dtype) + 0.2
    factor_a = torch.randn((q, p, p), generator=generator, dtype=dtype)
    factor_b = torch.randn((p, q, q), generator=generator, dtype=dtype)

    xe_expression = transform(
        xe.array(inputs),
        Parameter(xe.array(factor_a)),
        Parameter(xe.array(factor_b)),
    )
    xe_output = xe_expression.materialize().backend_array

    cirkit_layer = TorchMonarchSumLayer(
        units,
        units,
        p=p,
        q=q,
        factor_a=_torch_parameter(factor_a),
        factor_b=_torch_parameter(factor_b),
        semiring=SumProductSemiring,
    )
    cirkit_output = cirkit_layer(
        inputs.reshape(1, 1, inputs.shape[0], units)
    ).squeeze(0)

    matrix = materialize_monarch_matrix(factor_a, factor_b)
    expected = torch.einsum(
        "bi,oi->bo",
        inputs.reshape(inputs.shape[0], units),
        matrix,
    )
    torch.testing.assert_close(
        xe_output.reshape_as(expected),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        cirkit_output,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_lse_cirkit_monarch_matches_the_log_dense_oracle() -> None:
    generator = torch.Generator(device="cpu").manual_seed(9)
    dtype = torch.float64
    p = q = 2
    units = p * q
    inputs = torch.rand((3, p, q), generator=generator, dtype=dtype) + 0.2
    factor_a = torch.randn((q, p, p), generator=generator, dtype=dtype)
    factor_b = torch.randn((p, q, q), generator=generator, dtype=dtype)
    layer = TorchMonarchSumLayer(
        units,
        units,
        p=p,
        q=q,
        factor_a=_torch_parameter(factor_a),
        factor_b=_torch_parameter(factor_b),
        semiring=LSESumSemiring,
    )

    actual = layer(
        inputs.log().reshape(1, 1, inputs.shape[0], units)
    ).squeeze(0)
    matrix = materialize_monarch_matrix(factor_a, factor_b)
    expected = torch.logsumexp(
        inputs.log().reshape(inputs.shape[0], 1, units)
        + matrix.log().unsqueeze(0),
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_small_quad_tree_has_only_the_expected_monarch_parameters() -> None:
    circuit = build_symbolic_circuit(
        width=2,
        height=2,
        units=4,
        categories=4,
        region_graph="quad-tree-2",
        parameterization="monarch",
        factors=(2, 2),
    )
    state = canonicalize_parameters(circuit, seed=4)

    # Four categorical leaves, six hidden CP sums, and one dense root.
    assert state.parameters == 4 * 4 * 4 + 6 * 4 * (2 + 2) + 4
    assert factor_shapes(2, 2) == ((2, 2, 2), (2, 2, 2))


@pytest.mark.parametrize("region_graph", ["quad-tree-2", "quad-graph"])
@pytest.mark.parametrize(
    ("parameterization", "factors"),
    [("dense", None), ("monarch", (2, 2))],
)
def test_cached_group_ordering_exactly_matches_the_uncached_algorithm(
    monkeypatch: pytest.MonkeyPatch,
    region_graph: str,
    parameterization: str,
    factors: tuple[int, int] | None,
) -> None:
    symbolic = build_symbolic_circuit(
        width=2,
        height=2,
        units=4,
        categories=4,
        region_graph=region_graph,
        parameterization=parameterization,
        factors=factors,
    )
    canonicalize_parameters(symbolic, seed=29)
    program, _inputs = translate_to_xe(symbolic, batch_size=2)
    cached = FoldSameShapedOperations.apply_with_input_depth_metadata(
        program,
        optimize_group_order=True,
    )

    def uncached_future_order_keys(
        members,
        events,
        consumers,
        event_index_by_op_index,
        ordered_groups,
    ):
        return {
            member.result_id: preprocess._group_member_future_order_key(
                member,
                events,
                consumers,
                event_index_by_op_index,
                ordered_groups,
            )
            for member in members
        }

    def legacy_group_member_order_score(
        group,
        members,
        program,
        events,
        consumers,
        event_index_by_op_index,
        group_position_by_result_id,
        ordered_groups,
        future_order_keys,
        _future_consumer_argument_sequences,
    ):
        return (
            preprocess._future_consumer_segment_count(
                group,
                members,
                events,
                consumers,
                event_index_by_op_index,
                ordered_groups,
            ),
            preprocess._materialization_source_run_count(
                group,
                members,
                program,
                group_position_by_result_id,
            ),
            tuple(
                future_order_keys[member.result_id]
                for member in members
            ),
            tuple(member.op_index for member in members),
        )

    def legacy_event_materialization_instruction_count(
        program,
        events,
        event_index,
        position_by_result_id,
        input_axis0_position_by_index,
    ):
        event = events[event_index]
        if isinstance(event, int):
            return 0
        operand_count = len(
            event.members[0].canonical_argument_order
        )
        return sum(
            preprocess._estimated_ref_materialization_instruction_count(
                program,
                events,
                tuple(
                    preprocess._estimated_materialization_ref(
                        program,
                        member.arguments[
                            member.canonical_argument_order[
                                canonical_position
                            ]
                        ],
                        position_by_result_id,
                        input_axis0_position_by_index,
                    )
                    for member in event.members
                ),
            )
            for canonical_position in range(operand_count)
        )

    def legacy_dependency_only_groups(members, result_dependencies):
        groups = []
        for member in sorted(
            members,
            key=lambda item: item.op_index,
        ):
            for group in groups:
                if all(
                    not preprocess._op_members_depend_on_each_other(
                        member,
                        existing_member,
                        result_dependencies,
                    )
                    for existing_member in group
                ):
                    group.append(member)
                    break
            else:
                groups.append([member])
        return tuple(tuple(group) for group in groups)

    def legacy_beam_optimizer(events, program):
        def state_key(state):
            return tuple(
                (event,)
                if isinstance(event, int)
                else tuple(member.op_index for member in event.members)
                for event in state
            )

        def state_score(state):
            return (
                preprocess._estimated_total_materialization_instruction_count(
                    program,
                    state,
                ),
                state_key(state),
            )

        group_event_indices = tuple(
            event_index
            for event_index, event in enumerate(events)
            if not isinstance(event, int)
        )
        states = (events,)
        for event_index in group_event_indices:
            candidate_states = []
            for state in states:
                event = state[event_index]
                for candidate_members in (
                    preprocess._group_order_candidates_for_state(
                        program,
                        state,
                        event_index,
                    )
                ):
                    state_events = list(state)
                    state_events[event_index] = preprocess.replace(
                        event,
                        members=candidate_members,
                    )
                    candidate_states.append(tuple(state_events))
            unique_states = {
                state_key(state): state for state in candidate_states
            }
            states = tuple(
                sorted(
                    unique_states.values(),
                    key=state_score,
                )[: preprocess._GROUP_ORDER_BEAM_WIDTH]
            )
        return min(states, key=state_score)

    monkeypatch.setattr(
        preprocess,
        "_group_member_future_order_keys",
        uncached_future_order_keys,
    )
    monkeypatch.setattr(
        preprocess,
        "_group_member_order_score",
        legacy_group_member_order_score,
    )
    monkeypatch.setattr(
        preprocess,
        "_estimated_event_materialization_instruction_count",
        legacy_event_materialization_instruction_count,
    )
    monkeypatch.setattr(
        preprocess,
        "_split_dependency_only_op_groups",
        legacy_dependency_only_groups,
    )
    monkeypatch.setattr(
        preprocess,
        "_optimize_group_member_orders_by_materialization_cost",
        legacy_beam_optimizer,
    )
    uncached = FoldSameShapedOperations.apply_with_input_depth_metadata(
        program,
        optimize_group_order=True,
    )

    assert cached == uncached


@pytest.mark.parametrize("region_graph", ["quad-tree-2", "quad-graph"])
@pytest.mark.parametrize(
    ("parameterization", "factors"),
    [("dense", None), ("monarch", (2, 2))],
)
def test_small_end_to_end_cirkit_and_input_depth_xe_match(
    region_graph: str,
    parameterization: str,
    factors: tuple[int, int] | None,
) -> None:
    common = {
        "width": 2,
        "height": 2,
        "units": 4,
        "categories": 4,
        "batch_size": 2,
        "region_graph": region_graph,
        "parameterization": parameterization,
        "factors": factors,
        "seed": 13,
        "device": torch.device("cpu"),
    }
    xe_setup = setup_xe(**common)
    cirkit_setup = setup_cirkit(**common)
    batch = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

    assert xe_setup.initialization_hash == cirkit_setup.initialization_hash
    assert xe_setup.parameters == cirkit_setup.parameters
    assert xe_setup.monarch_layers == cirkit_setup.monarch_layers
    assert (xe_setup.monarch_layers > 0) == (parameterization == "monarch")
    torch.testing.assert_close(
        xe_setup.step(batch),
        cirkit_setup.step(batch),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("backend", ["cirkit", "xe"])
def test_benchmark_labels_only_native_cirkit_or_input_depth_xe(backend: str) -> None:
    row = base_row(
        Configuration(
            backend,
            0,
            Scale("quad-tree-2", "monarch", 256, 512, 16, 32),
        ),
        device=torch.device("cpu"),
    )
    assert row["fold_strategy"] == (
        "cirkit-native" if backend == "cirkit" else "xe-input-depth"
    )


def test_resume_requires_the_summary_and_all_raw_batch_rows(tmp_path) -> None:
    configuration = Configuration(
        "xe",
        0,
        Scale("quad-tree-2", "monarch", 256, 512, 16, 32),
    )
    summary = tmp_path / "summary.csv"
    raw = tmp_path / "raw.csv"
    row = base_row(configuration, device=torch.device("cpu"))
    row["status"] = "ok"
    append_row(summary, row)

    assert completed_keys(summary, raw) == set()
    append_raw_rows(
        raw,
        [
            {
                "backend": "xe",
                "seed": 0,
                "region_graph": "quad-tree-2",
                "parameterization": "monarch",
                "units": 512,
                "batch_size": 256,
                "measured_batch": batch,
            }
            for batch in range(MEASURED_BATCHES)
        ],
    )
    assert completed_keys(summary, raw) == {configuration.key}


def test_resume_does_not_count_duplicate_raw_batches_as_complete(tmp_path) -> None:
    configuration = Configuration(
        "xe",
        0,
        Scale("quad-tree-2", "monarch", 256, 512, 16, 32),
    )
    summary = tmp_path / "summary.csv"
    raw = tmp_path / "raw.csv"
    row = base_row(configuration, device=torch.device("cpu"))
    row["status"] = "ok"
    append_row(summary, row)
    append_raw_rows(
        raw,
        [
            {
                "backend": "xe",
                "seed": 0,
                "region_graph": "quad-tree-2",
                "parameterization": "monarch",
                "units": 512,
                "batch_size": 256,
                "measured_batch": batch % (MEASURED_BATCHES // 2),
            }
            for batch in range(MEASURED_BATCHES)
        ],
    )

    assert completed_keys(summary, raw) == set()
