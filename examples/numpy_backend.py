"""Use the NumPy backend included in the base installation."""

import numpy as np

import extended_einsum as xe


def main() -> None:
    source_numpy = np.arange(6.0).reshape(2, 3)
    source = xe.array(source_numpy)
    result = xe.softmax(source, axis=1).materialize()

    shifted = source_numpy - source_numpy.max(axis=1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(result.backend_array, expected)
    print(result.backend_array)


if __name__ == "__main__":
    main()
