"""Whole-program compilation without an eager array-operations interface."""

from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeVar

from extended_einsum.language.rich_program import RichProgram

T = TypeVar("T")


class UnsupportedProgramError(NotImplementedError):
    """A compiled backend cannot execute this program or input specialization."""


class CompiledBackend(Protocol):
    """Compile a RichProgram directly, before primitive backend lowering.

    Inputs are examples for shape/dtype/device specialization. Values remain
    dynamic unless explicitly declared constant in backend-specific options.
    The returned executable accepts the same input sequence. Implementations
    must reject unsupported programs, never silently change execution backend.
    """

    def compile(self, program: RichProgram, inputs: Sequence[Any], **options: Any) -> Callable: ...


def compile_program(program: RichProgram, inputs: Sequence[T], *, backend: str, **options: Any) -> Callable[[Sequence[T]], T]:
    """Compile extracted or preprocessed IR using an explicitly chosen backend.

    Compiled-only backends consume RichProgram directly. Eager backends retain
    their existing stability translation and BackendCompiler interface.
    """
    from extended_einsum.backend_translation import translate_to_backend_program
    from extended_einsum.backends.registry import get_backend_compiler, get_backend_functions, get_compiled_backend

    compiled = get_compiled_backend(backend)
    if compiled is not None:
        return compiled.compile(program, inputs, **options)
    if options:
        raise TypeError(f"Compiler options {sorted(options)} are unsupported for eager backend {backend!r}")
    lowered = translate_to_backend_program(program, get_backend_functions(backend))
    return get_backend_compiler(backend).compile(lowered, inputs)
