"""Compare direct, scaled, and log-space evaluation on positive tensors."""

import torch

import extended_einsum as xe


def main() -> None:
    left = torch.tensor([[0.2, 0.3], [0.4, 0.5]])
    right = torch.tensor([[0.6, 0.7], [0.8, 0.9]])
    expression = xe.einsum("ik,kj->ij", left, right)

    direct = expression.materialize("unstable")
    scaled = expression.materialize("scaled_sum")
    logspace = expression.materialize("logspace_max")

    torch.testing.assert_close(scaled, direct)
    torch.testing.assert_close(logspace, direct)
    print(direct)


if __name__ == "__main__":
    main()
