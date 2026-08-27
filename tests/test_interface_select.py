import unittest

import torch

import extended_einsum.interface as xe


class SelectInterfaceTests(unittest.TestCase):
    def test_select_uses_index_on_default_axis(self) -> None:
        source = torch.arange(24).reshape(2, 3, 4)

        selected = xe.select(source, 1)

        self.assertEqual(selected.shape, (3, 4))
        torch.testing.assert_close(
            selected.materialize(stability_mode="unstable"),
            source[1],
        )

    def test_select_uses_explicit_axis(self) -> None:
        source = torch.arange(24).reshape(2, 3, 4)

        selected = xe.select(source, 2, axis=1)

        self.assertEqual(selected.shape, (2, 4))
        torch.testing.assert_close(
            selected.materialize(stability_mode="unstable"),
            source[:, 2],
        )


if __name__ == "__main__":
    unittest.main()
