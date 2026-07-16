"""Build and materialize a small extended einsum expression with PyTorch."""

import torch

import extended_einsum as xe


def main() -> None:
    left_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right_torch = torch.tensor([[2.0, 0.0], [1.0, 2.0]])

    left = xe.array(left_torch)
    right = xe.array(right_torch)
    expression = xe.einsum("ik,kj->ij", xe.exp(left), right)
    result = expression.materialize()

    expected = torch.einsum("ik,kj->ij", torch.exp(left_torch), right_torch)
    torch.testing.assert_close(result.backend_array, expected)
    print(result.backend_array)


if __name__ == "__main__":
    main()
