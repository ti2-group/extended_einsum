"""Build and materialize a small extended einsum expression with PyTorch."""

import torch

import extended_einsum as xe


def main() -> None:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = torch.tensor([[2.0, 0.0], [1.0, 2.0]])

    expression = xe.einsum("ik,kj->ij", xe.exp(left), right)
    result = expression.materialize()

    expected = torch.einsum("ik,kj->ij", torch.exp(left), right)
    torch.testing.assert_close(result, expected)
    print(result)


if __name__ == "__main__":
    main()
