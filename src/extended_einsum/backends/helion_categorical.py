"""Categorical evidence bindings for the inference-only Helion compiler."""

from __future__ import annotations

from dataclasses import dataclass, replace
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

    The compiler reads evidence inside its consuming subtree, avoiding the
    full leaf tensor. With logits, a separate kernel normalizes the tables on
    every call. Inputs must be contiguous, on one device, and retain their
    shapes/dtypes. Category, variable, and bank IDs must be in range; compiled
    execution masks invalid addresses and returns NaNs for invalid evidence.
    Routing IDs and pixels may change between calls. ``log_values=False``
    supplies probabilities to a circuit that expects linear-valued evidence.
    Gradients are unsupported.
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


@dataclass
class _Preparation:
    source: int
    kernel: object
    code: str


def _normalizer(source: int, binding: CategoricalInput) -> _Preparation:
    from extended_einsum.backends.helion import _make_kernel

    banks, categories, units = binding.tables.shape
    code = (
        "def categorical_normalize(table: torch.Tensor):\n"
        "    out = torch.empty_like(table)\n"
        f"    flat = table.view([-1, {units}])\n"
        f"    result = out.view([-1, {units}])\n"
        f"    for bank, chunk in hl.grid([{banks}, {(units + 31) // 32}]):\n"
        f"        categories = hl.arange({categories})\n"
        "        units = chunk * 32 + hl.arange(32)\n"
        f"        valid_categories = categories < {categories}\n"
        f"        valid = valid_categories[:, None] & (units < {units})[None, :]\n"
        f"        values = hl.load(flat, [bank * {categories} + categories, units], extra_mask=valid)\n"
        "        masked = torch.where(valid_categories[:, None], values, float('-inf'))\n"
        "        shift = torch.amax(masked, dim=0, keepdim=True)\n"
        "        exponentials = torch.where(valid_categories[:, None], torch.exp(values - shift), 0.0)\n"
        "        total = torch.sum(exponentials, dim=0, keepdim=True)\n"
        "        normalized = values - shift - torch.log(total)\n"
        f"        hl.store(result, [bank * {categories} + categories, units], normalized, extra_mask=valid)\n"
        "    return out\n"
    )
    return _Preparation(source, _make_kernel(code, "categorical_normalize"), code)


def _prepared(binding: CategoricalInput, table: torch.Tensor) -> CategoricalInput:
    return replace(binding, tables=table, logits=False)


def _lookup_lines(i: int, binding: CategoricalInput, row_code: str, unit_code: str | None = None):
    """Shared address/mask generation for fused and isolated lookups."""
    banks, categories, units = binding.tables.shape
    variables = binding.pixels.shape[1]
    views = [f"    pixels{i} = arg{i}_pixels.view([-1])\n", f"    tables{i} = arg{i}_tables.view([-1, {units}])\n"]
    column_index = "slice(None)" if unit_code is None else unit_code
    mask = f"valid_evidence{i}[:, None]" + (" & valid_units[None, :]" if unit_code is not None else "")
    # Torch compares a uint8 tensor to 256 after narrowing the scalar to zero.
    # Widen byte pixels before bounds checks so all 256 categories remain valid.
    category_cast = ".to(torch.int32)" if binding.pixels.dtype == torch.uint8 else ""
    loads = [
        f"pixel_ids{i} = hl.load(arg{i}_variables, [{row_code} + samples * 0], extra_mask=valid)",
        f"bank_ids{i} = hl.load(arg{i}_banks, [{row_code} + samples * 0], extra_mask=valid)",
        f"valid_pixels{i} = valid & (pixel_ids{i} >= 0) & (pixel_ids{i} < {variables})",
        f"categories{i} = hl.load(pixels{i}, [samples * {variables} + pixel_ids{i}.to(torch.int32)], extra_mask=valid_pixels{i}){category_cast}",
        f"valid_evidence{i} = valid_pixels{i} & (bank_ids{i} >= 0) & (bank_ids{i} < {banks}) & (categories{i} >= 0) & (categories{i} < {categories})",
        f"lookup{i} = hl.load(tables{i}, [bank_ids{i}.to(torch.int32) * {categories} + categories{i}.to(torch.int32), {column_index}], extra_mask={mask})",
        f"categorical_values{i} = torch.where(valid_evidence{i}[:, None], lookup{i}, float('nan'))",
        f"x{i} = " + (f"categorical_values{i}" if binding.log_values else f"torch.exp(categorical_values{i})"),
    ]
    return views, loads


def _lookup_kernel(binding: CategoricalInput, *, sample_block: int, unit_block: int, num_warps: int = 4, grid_order: str = "samples"):
    """Materializing control for indexing benchmarks, without circuit work."""
    from extended_einsum.backends.helion import _make_kernel

    if binding.logits:
        raise ValueError("The isolated lookup benchmark requires normalized log-probability tables.")
    if grid_order not in ("samples", "roots") or any(n < 1 or n & (n - 1) for n in (sample_block, unit_block)):
        raise ValueError("Lookup tiles must be powers of two and order samples/roots.")
    folds, batch, units = binding.shape
    chunks, unit_chunks = (batch + sample_block - 1) // sample_block, (units + unit_block - 1) // unit_block
    grid = f"chunk, unit_chunk, root in hl.grid([{chunks}, {unit_chunks}, {folds}])" if grid_order == "samples" else f"root, chunk, unit_chunk in hl.grid([{folds}, {chunks}, {unit_chunks}])"
    views, loads = _lookup_lines(0, binding, "root", "unit_indices")
    source = (
        "def categorical_lookup(arg0_pixels: torch.Tensor, arg0_tables: torch.Tensor, arg0_variables: torch.Tensor, arg0_banks: torch.Tensor):\n"
        f"    out = torch.empty([{folds}, {batch}, {units}], dtype=torch.float32, device=arg0_pixels.device)\n" + "".join(views) + f"    for {grid}:\n"
        f"        samples = chunk * {sample_block} + hl.arange({sample_block})\n"
        f"        valid = samples < {batch}\n"
        f"        unit_indices = unit_chunk * {unit_block} + hl.arange({unit_block})\n"
        f"        valid_units = unit_indices < {units}\n"
        + "".join(f"        {line}\n" for line in loads)
        + "        hl.store(out, [root, samples, unit_indices], x0, extra_mask=valid[:, None] & valid_units[None, :])\n"
        "    return out\n"
    )
    return _make_kernel(source, "categorical_lookup", num_warps=num_warps), source
