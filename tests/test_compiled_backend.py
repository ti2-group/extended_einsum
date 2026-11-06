"""Compiled-only registration bypasses eager primitive lowering entirely."""

import subprocess
import sys

import numpy as np
import pytest

import extended_einsum as xe
from extended_einsum.backends.numpy import NumpyBackendFunctions
from extended_einsum.backends.registry import get_backend_compiler, get_backend_functions, get_backend_of_array, get_compiled_backend


class ExpCompiler:
    def compile(self, program, inputs, *, marker=None):
        assert marker == "whole-program"
        assert program.instructions[0].operator.name == "exp"
        assert program.stability_mode == "logspace_max"
        shape = inputs[0].shape

        def execute(values):
            if len(values) != 1 or values[0].shape != shape:
                raise xe.UnsupportedProgramError("input specialization changed")
            return np.exp(values[0])

        return execute


def test_compiled_backend_registration_and_expression_materialization():
    xe.register_compiled_backend("compiled_test", ExpCompiler())
    source = np.array([0.0, 1.0])
    # Registration does not change automatic detection for existing backends.
    assert get_backend_of_array(source) == "numpy"
    wrapped = xe.array(source, backend="compiled_test")
    expression = xe.exp(wrapped)
    np.testing.assert_allclose(expression.materialize("logspace_max", marker="whole-program").backend_array, np.exp(source))
    program, _ = xe.extract_program(expression, stability_mode="logspace_max")
    executable = xe.compile_program(program, [source], backend="compiled_test", marker="whole-program")
    np.testing.assert_allclose(executable([source + 2]), np.exp(source + 2))
    with pytest.raises(xe.UnsupportedProgramError, match="specialization"):
        executable([np.ones(3)])
    with pytest.raises(TypeError, match="compiled-only"):
        get_backend_functions("compiled_test")
    with pytest.raises(TypeError, match="RichProgram"):
        get_backend_compiler("compiled_test")


def test_compiled_and_eager_registration_replace_each_other():
    xe.register_backend("replacement_test", NumpyBackendFunctions(), is_array=lambda _: False)
    xe.register_compiled_backend("replacement_test", ExpCompiler())
    assert get_compiled_backend("replacement_test") is not None
    with pytest.raises(TypeError, match="compiled-only"):
        get_backend_functions("replacement_test")
    xe.register_backend("replacement_test", NumpyBackendFunctions())
    assert get_compiled_backend("replacement_test") is None
    result = xe.exp(xe.array(np.ones(2), backend="replacement_test")).materialize()
    np.testing.assert_allclose(result.backend_array, np.exp(np.ones(2)))


def test_compiled_backend_registration_validation():
    with pytest.raises(ValueError, match="non-empty"):
        xe.register_compiled_backend("", ExpCompiler())
    with pytest.raises(TypeError, match="compile method"):
        xe.register_compiled_backend("invalid_compiled", object())


def test_triton_registration_does_not_import_optional_runtime():
    code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "triton" or name.startswith("triton."):
        raise ImportError("blocked optional Triton runtime")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import extended_einsum as xe
from extended_einsum.backends.registry import get_compiled_backend
assert get_compiled_backend("triton") is not None
import numpy as np
np.testing.assert_allclose(xe.exp(xe.array(np.zeros(2))).materialize().backend_array, np.ones(2))
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
