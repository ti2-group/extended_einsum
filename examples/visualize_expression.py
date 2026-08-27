"""Render an expression DAG (`pip install extended-einsum[visualization]`)."""

import sys
from pathlib import Path

import torch

import extended_einsum as xe
from extended_einsum.visualization import plot_expression_dag


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("expression.png")

    left = torch.ones((2, 3))
    right = torch.ones((3, 4))
    expression = xe.softmax(xe.einsum("ik,kj->ij", left, right), axis=1)
    program, _ = xe.extract_program(expression, stability_mode="unstable")

    plot_expression_dag(program, save_path=output, input_labels=["left", "right"])
    print(output.resolve())


if __name__ == "__main__":
    main()
