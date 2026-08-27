"""Use the NumPy backend included in the base installation."""

import numpy as np

import extended_einsum as xe


def main() -> None:
    source = np.arange(6.0).reshape(2, 3)
    result = xe.softmax(source, axis=1).materialize()

    shifted = source - source.max(axis=1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(result, expected)
    print(result)


if __name__ == "__main__":
    main()
