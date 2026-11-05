"""Categorical evidence shared by Triton inference and its Torch reference."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch


@dataclass(frozen=True)
class CategoricalInput:
    """A virtual ``(fold, batch, units)`` log-valued input.

    ``pixels`` contains integer category IDs in ``(batch, variables)`` order.
    ``tables`` contains FP32 log probabilities in ``(banks, categories, units)``
    order, or unnormalized logits when ``logits=True``. Each folded row selects
    a pixel through ``variable_ids`` and a parameter bank through ``table_ids``
    (defaults to ``variable_ids``). Repeated and reordered IDs are supported.

    Triton reads evidence inside its consuming subtree. Custom circuits require
    normalized log-valued tables. ``materialize`` also supports logits and
    probability-valued output through Torch, including autograd.

    Inputs must be contiguous, on one device, and retain their shapes/dtypes.
    Category, variable and bank IDs must be in range. Compiled execution masks
    invalid addresses and returns NaNs for invalid evidence. Routing IDs and
    pixels may change between calls.
    """

    pixels: torch.Tensor
    tables: torch.Tensor
    variable_ids: torch.Tensor
    table_ids: torch.Tensor | None = None
    logits: bool = False
    log_values: bool = True

    def __post_init__(self):
        if self.table_ids is None:
            object.__setattr__(self, "table_ids", self.variable_ids)
        if self.pixels.ndim != 2 or self.tables.ndim != 3 or self.variable_ids.ndim != 1 or self.table_ids.shape != self.variable_ids.shape:
            raise ValueError("Categorical inputs require pixels[B,V], tables[P,C,U], and matching one-dimensional variable/table IDs.")
        if (
            self.tables.dtype != torch.float32
            or self.pixels.dtype not in (torch.uint8, torch.int32, torch.int64)
            or any(a.dtype not in (torch.int32, torch.int64) for a in (self.variable_ids, self.table_ids))
        ):
            raise ValueError("Categorical tables must be float32, pixels uint8/int32/int64, and routing IDs int32/int64.")
        if any(a.device != self.pixels.device or a.layout != torch.strided or not a.is_contiguous() for a in self.tensors):
            raise ValueError("Categorical tensors must be contiguous and on one device.")
        if any(a.numel() == 0 or a.numel() >= 2**31 for a in self.tensors):
            raise ValueError("Categorical tensors must be nonempty with fewer than 2**31 elements each.")

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return self.pixels, self.tables, self.variable_ids, self.table_ids

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.variable_ids.numel(), self.pixels.shape[0], self.tables.shape[2]

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self):
        return self.tables.dtype

    @property
    def device(self):
        return self.pixels.device

    @property
    def layout(self):
        return torch.strided

    @property
    def requires_grad(self):
        return self.tables.requires_grad

    def numel(self):
        return prod(self.shape)

    def is_contiguous(self):
        return all(a.is_contiguous() for a in self.tensors)

    def materialize(self, method: str = "advanced", *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Torch reference, including normalization when tables are logits."""
        table = self.tables.to(dtype=dtype) if dtype is not None else self.tables
        table = table.log_softmax(dim=1) if self.logits else table
        categories = self.pixels.index_select(1, self.variable_ids.long()).T.long()
        banks = self.table_ids.long()
        if method == "advanced":
            result = table[banks[:, None], categories].contiguous()
        elif method == "gather":
            selected = table.index_select(0, banks)
            result = torch.gather(selected, 1, categories[:, :, None].expand(-1, -1, table.shape[2]))
        elif method == "flattened":
            rows = (banks[:, None] * table.shape[1] + categories).reshape(-1)
            result = table.reshape(-1, table.shape[2]).index_select(0, rows).reshape(self.shape)
        else:
            raise ValueError("Categorical lookup must be advanced, gather, or flattened.")
        return result if self.log_values else result.exp()
