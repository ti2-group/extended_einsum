"""ImageNet64 loading and preprocessing for the Monarch image experiments.

The downsampled ImageNet archives used by the Monarch Circuits repository store
RGB images in ``.npz`` shards under the keys ``data`` and ``labels``.  The
canonical array is flattened NCHW (channel-major), although this loader also
accepts already-shaped NCHW and NHWC arrays so that converted archives do not
silently scramble channels.

The paper models aligned 8 x 8 patches.  Its prose describes using every patch,
whereas the released implementation randomly selects one aligned patch whenever
an image is read.  ``patch_mode`` exposes both behaviours.  The random mode here
is deliberately stateless and reproducible from ``sample_seed``, ``epoch``, and
the source-image index.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_SIZE = 64
NUM_CHANNELS = 3
NUM_RGB_CATEGORIES = 256
NUM_YCOCG_R_CATEGORIES = 512

Split = Literal["train", "validation", "test"]
ColorTransform = Literal["rgb", "grayscale", "ycocg-r", "ycocg"]
PatchMode = Literal["random", "all"]
OutputLayout = Literal["flat", "chw", "hwc"]

_MASK_64 = (1 << 64) - 1
_NATURAL_PART = re.compile(r"(\d+)")
_TRAIN_SHARD = re.compile(r"^train_data_batch_(\d+)\.npz$", re.IGNORECASE)
_VALIDATION_SHARD = re.compile(r"^(?:val|validation)_data(?:_batch_(\d+))?\.npz$", re.IGNORECASE)


class ImageNet64FormatError(ValueError):
    """Raised when a shard is not a supported downsampled-ImageNet NPZ file."""


def num_categories_for_transform(color_transform: ColorTransform) -> int:
    """Return the homogeneous categorical cardinality used by the PC leaves."""

    _validate_color_transform(color_transform)
    if color_transform == "ycocg-r":
        # The official experiment shifts both signed chroma channels by 256 and
        # configures one homogeneous 512-way categorical distribution.
        return NUM_YCOCG_R_CATEGORIES
    return NUM_RGB_CATEGORIES


def num_channels_for_transform(color_transform: ColorTransform) -> int:
    """Return the number of categorical channels produced by a transform."""

    _validate_color_transform(color_transform)
    return 1 if color_transform == "grayscale" else NUM_CHANNELS


def rgb_to_grayscale(rgb: torch.Tensor) -> torch.Tensor:
    """Convert 8-bit RGB to one deterministic 8-bit luminance channel.

    Integer coefficients approximate BT.601 luma and sum to 256. Adding 128
    implements round-to-nearest before the exact division by 256.
    """

    rgb = _validate_rgb_chw(rgb).to(torch.long)
    red, green, blue = rgb.unbind(dim=0)
    luminance = (77 * red + 150 * green + 29 * blue + 128) >> 8
    return luminance.unsqueeze(0)


def rgb_to_ycocg_r(rgb: torch.Tensor) -> torch.Tensor:
    """Apply the paper's reversible integer YCoCg-R transform to a CHW tensor.

    The result is categorical: Y remains in [0, 255], while Co and Cg are
    shifted by +256 and fit in [0, 511].  This matches the released code's
    encoding and homogeneous 512-category input distribution.  We operate
    directly on integers, as specified in the paper, rather than reproducing the
    released loader's avoidable float32 normalization round trip.
    """

    rgb = _validate_rgb_chw(rgb).to(torch.long)
    red, green, blue = rgb.unbind(dim=0)
    co = red - blue
    tmp = blue + torch.div(co, 2, rounding_mode="floor")
    cg = green - tmp
    y = tmp + torch.div(cg, 2, rounding_mode="floor")
    return torch.stack((y, co + 256, cg + 256), dim=0)


def ycocg_r_to_rgb(ycocg_r: torch.Tensor) -> torch.Tensor:
    """Invert :func:`rgb_to_ycocg_r` exactly, returning a long CHW tensor."""

    if ycocg_r.ndim != 3 or ycocg_r.shape[0] != NUM_CHANNELS:
        raise ValueError(f"Expected a CHW tensor with three channels, got shape {tuple(ycocg_r.shape)}")
    values = ycocg_r.to(torch.long)
    y, co, cg = values.unbind(dim=0)
    co = co - 256
    cg = cg - 256
    tmp = y - torch.div(cg, 2, rounding_mode="floor")
    green = cg + tmp
    blue = tmp - torch.div(co, 2, rounding_mode="floor")
    red = blue + co
    rgb = torch.stack((red, green, blue), dim=0)
    if torch.any((rgb < 0) | (rgb >= NUM_RGB_CATEGORIES)):
        raise ValueError("YCoCg-R values do not decode to 8-bit RGB")
    return rgb


def rgb_to_quantized_ycocg(rgb: torch.Tensor) -> torch.Tensor:
    """Apply the released lossy YCoCg transform and quantize to 256 bins."""

    rgb = _validate_rgb_chw(rgb).to(torch.float64) / 255.0
    red, green, blue = rgb.unbind(dim=0)
    co = red - blue
    tmp = blue + co / 2.0
    cg = green - tmp
    y = (tmp + cg / 2.0) * 2.0 - 1.0
    transformed = torch.stack((y, co, cg), dim=0)
    return torch.floor((transformed + 1.0) * 0.5 * NUM_RGB_CATEGORIES).to(torch.long).clamp_(0, NUM_RGB_CATEGORIES - 1)


class ImageNet64Dataset(Dataset[torch.Tensor]):
    """Lazy, deterministic view of official ImageNet64 NPZ shards.

    ``sample_limit`` counts returned PC examples.  Thus, in ``patch_mode="all"``
    it counts patches rather than source images.  A value of zero or ``None``
    selects the entire split.  A bounded selection is deterministic and depends
    only on ``sample_seed``.

    ``split="test"`` intentionally aliases the validation split.  The Monarch
    repository defines train and validation loaders only, but calls held-out
    ImageNet likelihood a test metric in the paper.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: Split = "train",
        color_transform: ColorTransform = "ycocg-r",
        patch_size: int | None = 8,
        patch_mode: PatchMode = "random",
        sample_limit: int | None = 0,
        sample_seed: int = 0,
        epoch: int = 0,
        output_layout: OutputLayout = "flat",
    ) -> None:
        super().__init__()
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unsupported ImageNet64 split: {split!r}")
        _validate_color_transform(color_transform)
        if patch_mode not in ("random", "all"):
            raise ValueError(f"Unsupported patch mode: {patch_mode!r}")
        if output_layout not in ("flat", "chw", "hwc"):
            raise ValueError(f"Unsupported output layout: {output_layout!r}")
        if patch_size is not None and (patch_size <= 0 or IMAGE_SIZE % patch_size != 0):
            raise ValueError(f"patch_size must be a positive divisor of {IMAGE_SIZE}, got {patch_size}")
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        if sample_limit is not None and sample_limit < 0:
            raise ValueError(f"sample_limit must be non-negative, got {sample_limit}")

        self.root = Path(root)
        self.split: Split = split
        self.resolved_split: Literal["train", "validation"] = "train" if split == "train" else "validation"
        self.color_transform = color_transform
        self.patch_size = patch_size
        self.patch_mode = patch_mode
        self.sample_seed = sample_seed
        self.epoch = epoch
        self.output_layout = output_layout
        self.num_categories = num_categories_for_transform(color_transform)

        self.shard_paths = tuple(_discover_shards(self.root, self.resolved_split))
        shard_lengths: list[int] = []
        shard_labels: list[np.ndarray | None] = []
        for path in self.shard_paths:
            length, labels = _read_shard_metadata(path)
            shard_lengths.append(length)
            shard_labels.append(labels)
        self._shard_lengths = tuple(shard_lengths)
        self._shard_labels = tuple(shard_labels)
        self._cumulative_lengths = tuple(np.cumsum(self._shard_lengths, dtype=np.int64).tolist())
        self.num_source_images = self._cumulative_lengths[-1]

        self.patches_per_image = 1
        if patch_size is not None and patch_mode == "all":
            self.patches_per_image = (IMAGE_SIZE // patch_size) ** 2
        total_examples = self.num_source_images * self.patches_per_image
        self._selected_indices = _deterministic_subset(total_examples, sample_limit, sample_seed)
        self._length = total_examples if self._selected_indices is None else len(self._selected_indices)

        self._cached_shard_index: int | None = None
        self._cached_data: np.ndarray | None = None

    def __len__(self) -> int:
        return self._length

    @property
    def selected_indices(self) -> Sequence[int]:
        """Logical split indices selected before patch preprocessing."""

        if self._selected_indices is None:
            return range(self._length)
        return self._selected_indices

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by deterministic random-aligned patch selection."""

        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self.epoch = epoch

    def epoch_indices(self, *, seed: int, epoch: int) -> torch.Tensor:
        """Return a deterministic, shard-local shuffle of dataset indices.

        Compressed NPZ arrays must be decompressed a shard at a time.  A global
        permutation would repeatedly evict and reload those large arrays.  This
        order shuffles the shards, then independently shuffles examples within
        each shard, so every source shard remains one contiguous run.  The
        returned int64 tensor is always on CPU and addresses this dataset after
        ``sample_limit`` has been applied.
        """

        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        shard_ranges = self._dataset_ranges_by_shard()
        if not shard_ranges:
            return torch.empty(0, dtype=torch.long)

        epoch_key = (seed & _MASK_64) ^ (((epoch + 1) * 0xD2B74407B1CE6E93) & _MASK_64)
        shard_generator = torch.Generator(device="cpu")
        shard_generator.manual_seed(_splitmix64(epoch_key) & ((1 << 63) - 1))
        shard_order = torch.randperm(len(shard_ranges), generator=shard_generator).tolist()

        shuffled_ranges: list[torch.Tensor] = []
        for range_index in shard_order:
            shard_index, dataset_range = shard_ranges[range_index]
            count = len(dataset_range)
            within_generator = torch.Generator(device="cpu")
            within_key = epoch_key ^ (((shard_index + 1) * 0xCA5A826395121157) & _MASK_64)
            within_generator.manual_seed(_splitmix64(within_key) & ((1 << 63) - 1))
            shuffled_ranges.append(torch.randperm(count, generator=within_generator) + dataset_range.start)
        return torch.cat(shuffled_ranges)

    def label_at(self, index: int) -> int | None:
        """Return the source classification label, if present in the archive."""

        source_index, _ = self._map_index(index)
        shard_index, local_index = self._locate_source_image(source_index)
        labels = self._shard_labels[shard_index]
        if labels is None:
            return None
        return int(labels[local_index])

    def __getitem__(self, index: int) -> torch.Tensor:
        source_index, fixed_patch_index = self._map_index(index)
        shard_index, local_index = self._locate_source_image(source_index)
        shard = self._load_shard(shard_index)
        rgb_view = _image_to_chw_view(shard[local_index])

        if self.patch_size is not None:
            patch_index = fixed_patch_index
            if patch_index is None:
                patch_index = _random_patch_index(
                    source_index=source_index,
                    epoch=self.epoch,
                    seed=self.sample_seed,
                    patches_per_side=IMAGE_SIZE // self.patch_size,
                )
            row_slice, column_slice = _aligned_patch_slices(self.patch_size, patch_index)
            rgb_view = rgb_view[:, row_slice, column_slice]

        # Copy and cast only the selected patch.  In exhaustive evaluation this
        # avoids materializing a full 64 x 64 tensor for each of the 64 patches
        # returned from the same source image.
        rgb = _copy_chw_to_long(rgb_view)
        categorical = _apply_color_transform(rgb, self.color_transform)
        if self.output_layout == "flat":
            return categorical.reshape(-1).contiguous()
        if self.output_layout == "hwc":
            return categorical.permute(1, 2, 0).contiguous()
        return categorical.contiguous()

    def _map_index(self, index: int) -> tuple[int, int | None]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        logical_index = index if self._selected_indices is None else self._selected_indices[index]
        if self.patch_size is not None and self.patch_mode == "all":
            return divmod(logical_index, self.patches_per_image)
        return logical_index, None

    def _locate_source_image(self, source_index: int) -> tuple[int, int]:
        shard_index = bisect_right(self._cumulative_lengths, source_index)
        previous_total = 0 if shard_index == 0 else self._cumulative_lengths[shard_index - 1]
        return shard_index, source_index - previous_total

    def _dataset_ranges_by_shard(self) -> list[tuple[int, range]]:
        logical_boundaries = [length * self.patches_per_image for length in self._cumulative_lengths]
        if self._selected_indices is None:
            dataset_boundaries = logical_boundaries
        else:
            dataset_boundaries = [bisect_left(self._selected_indices, boundary) for boundary in logical_boundaries]

        ranges: list[tuple[int, range]] = []
        start = 0
        for shard_index, end in enumerate(dataset_boundaries):
            if end > start:
                ranges.append((shard_index, range(start, end)))
            start = end
        return ranges

    def _load_shard(self, shard_index: int) -> np.ndarray:
        if self._cached_shard_index == shard_index and self._cached_data is not None:
            return self._cached_data
        path = self.shard_paths[shard_index]
        with np.load(path, allow_pickle=False) as archive:
            if "data" not in archive.files:
                raise ImageNet64FormatError(f"{path} does not contain a 'data' array")
            data = np.asarray(archive["data"])
        _validate_shard_data(data, path)
        expected_length = self._shard_lengths[shard_index]
        if data.shape[0] != expected_length:
            raise ImageNet64FormatError(f"{path} contains {data.shape[0]} images but {expected_length} labels")
        self._cached_shard_index = shard_index
        self._cached_data = data
        return data


def load_imagenet64_images(
    root: str | Path,
    *,
    split: Split = "train",
    color_transform: ColorTransform = "ycocg-r",
    patch_size: int | None = 8,
    patch_mode: PatchMode = "random",
    sample_limit: int | None = 0,
    sample_seed: int = 0,
    epoch: int = 0,
    output_layout: OutputLayout = "flat",
) -> torch.Tensor:
    """Materialize an ImageNet64 split (or bounded subset) as categorical data.

    This convenience function matches the in-memory interface used by the demo
    trainer.  Full ImageNet64 is large, so callers should normally provide a
    ``sample_limit`` or use :class:`ImageNet64Dataset` directly.
    """

    dataset = ImageNet64Dataset(
        root,
        split=split,
        color_transform=color_transform,
        patch_size=patch_size,
        patch_mode=patch_mode,
        sample_limit=sample_limit,
        sample_seed=sample_seed,
        epoch=epoch,
        output_layout=output_layout,
    )
    return torch.stack([dataset[index] for index in range(len(dataset))], dim=0)


def _validate_color_transform(color_transform: str) -> None:
    if color_transform not in ("rgb", "grayscale", "ycocg-r", "ycocg"):
        raise ValueError(f"Unsupported color transform: {color_transform!r}")


def _validate_rgb_chw(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 3 or rgb.shape[0] != NUM_CHANNELS:
        raise ValueError(f"Expected a CHW tensor with three channels, got shape {tuple(rgb.shape)}")
    if rgb.is_floating_point() or rgb.is_complex():
        raise ValueError(f"Expected integer RGB values, got dtype {rgb.dtype}")
    values = rgb.to(torch.long)
    if torch.any((values < 0) | (values >= NUM_RGB_CATEGORIES)):
        raise ValueError("RGB values must be integers in [0, 255]")
    return rgb


def _apply_color_transform(rgb: torch.Tensor, color_transform: ColorTransform) -> torch.Tensor:
    if color_transform == "grayscale":
        return rgb_to_grayscale(rgb)
    if color_transform == "ycocg-r":
        return rgb_to_ycocg_r(rgb)
    if color_transform == "ycocg":
        return rgb_to_quantized_ycocg(rgb)
    return _validate_rgb_chw(rgb).to(torch.long)


def _candidate_directories(root: Path, split: Literal["train", "validation"]) -> list[Path]:
    split_dir = "train" if split == "train" else "val"
    archive_dir = "Imagenet64_train_npz" if split == "train" else "Imagenet64_val_npz"
    bases = (root, root / "imagenet64", root / "ImageNet64", root / "ImageNet" / "imagenet64")
    candidates = [root / archive_dir, root / archive_dir.replace("Imagenet", "ImageNet")]
    candidates.extend(base / split_dir for base in bases)
    if split == "validation":
        candidates.extend(base / "validation" for base in bases)
    candidates.append(root)
    deduplicated: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized not in seen:
            seen.add(normalized)
            deduplicated.append(candidate)
    return deduplicated


def _discover_shards(root: Path, split: Literal["train", "validation"]) -> list[Path]:
    matcher = _TRAIN_SHARD if split == "train" else _VALIDATION_SHARD
    searched: list[Path] = []
    for directory in _candidate_directories(root, split):
        searched.append(directory)
        if not directory.is_dir():
            continue
        canonical = [path for path in directory.iterdir() if path.is_file() and matcher.match(path.name)]
        if canonical:
            return sorted(canonical, key=_natural_sort_key)
        if directory.name.lower() in ({"train"} if split == "train" else {"val", "validation"}):
            fallback = sorted(directory.glob("*.npz"), key=_natural_sort_key)
            if fallback:
                return fallback
    locations = ", ".join(str(path) for path in searched)
    raise FileNotFoundError(f"Could not find ImageNet64 {split} NPZ shards under {root}. Searched: {locations}")


def _natural_sort_key(path: Path) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in _NATURAL_PART.split(path.name))


def _read_shard_metadata(path: Path) -> tuple[int, np.ndarray | None]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "data" not in archive.files:
                raise ImageNet64FormatError(f"{path} does not contain a 'data' array")
            if "labels" in archive.files:
                labels = np.asarray(archive["labels"])
                if labels.ndim != 1:
                    labels = labels.reshape(-1)
                return len(labels), labels.copy()
            data = np.asarray(archive["data"])
            _validate_shard_data(data, path)
            return data.shape[0], None
    except (OSError, ValueError) as error:
        if isinstance(error, ImageNet64FormatError):
            raise
        raise ImageNet64FormatError(f"Could not read ImageNet64 shard {path}: {error}") from error


def _validate_shard_data(data: np.ndarray, path: Path) -> None:
    flat_width = NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE
    valid_shape = (data.ndim == 2 and data.shape[1] == flat_width) or (
        data.ndim == 4 and (data.shape[1:] == (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE) or data.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    )
    if not valid_shape:
        raise ImageNet64FormatError(f"Unsupported data shape in {path}: {data.shape}; expected (N, {flat_width}), (N, 3, 64, 64), or (N, 64, 64, 3)")
    if not np.issubdtype(data.dtype, np.integer):
        raise ImageNet64FormatError(f"Expected integer pixels in {path}, got dtype {data.dtype}")
    if data.size and (int(data.min()) < 0 or int(data.max()) >= NUM_RGB_CATEGORIES):
        raise ImageNet64FormatError(f"Expected 8-bit RGB values in {path}")


def _image_to_chw_view(image: np.ndarray) -> np.ndarray:
    """Return a zero-copy CHW view for any supported on-disk layout."""

    if image.ndim == 1:
        image = image.reshape(NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    elif image.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS):
        image = image.transpose(2, 0, 1)
    elif image.shape != (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
        raise ImageNet64FormatError(f"Unsupported ImageNet64 image shape: {image.shape}")
    return image


def _copy_chw_to_long(image: np.ndarray) -> torch.Tensor:
    """Copy a possibly strided CHW view and convert it to categorical longs."""

    return torch.from_numpy(np.array(image, copy=True)).to(torch.long)


def _image_to_chw(image: np.ndarray) -> torch.Tensor:
    """Materialize one full image as CHW categorical longs."""

    return _copy_chw_to_long(_image_to_chw_view(image))


def _aligned_patch_slices(patch_size: int, patch_index: int) -> tuple[slice, slice]:
    """Return the row and column slices for a row-major aligned patch."""

    patches_per_side = IMAGE_SIZE // patch_size
    if patch_index < 0 or patch_index >= patches_per_side**2:
        raise IndexError(f"Patch index {patch_index} is outside [0, {patches_per_side**2})")
    patch_row, patch_column = divmod(patch_index, patches_per_side)
    row_start = patch_row * patch_size
    column_start = patch_column * patch_size
    return slice(row_start, row_start + patch_size), slice(column_start, column_start + patch_size)


def _extract_aligned_patch(rgb: torch.Tensor, patch_size: int, patch_index: int) -> torch.Tensor:
    row_slice, column_slice = _aligned_patch_slices(patch_size, patch_index)
    return rgb[:, row_slice, column_slice]


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
    return (value ^ (value >> 31)) & _MASK_64


def _random_patch_index(*, source_index: int, epoch: int, seed: int, patches_per_side: int) -> int:
    key = seed & _MASK_64
    key ^= ((epoch + 1) * 0xD2B74407B1CE6E93) & _MASK_64
    key ^= ((source_index + 1) * 0xCA5A826395121157) & _MASK_64
    return _splitmix64(key) % (patches_per_side**2)


def _deterministic_subset(total: int, limit: int | None, seed: int) -> tuple[int, ...] | None:
    if limit in (None, 0) or limit == total:
        return None
    if limit is None:
        return None
    if limit > total:
        raise ValueError(f"sample_limit ({limit}) exceeds the split size ({total})")

    offset = _splitmix64(seed & _MASK_64) % total
    step = _splitmix64((seed ^ 0xA0761D6478BD642F) & _MASK_64) % total
    if step == 0:
        step = 1
    while math.gcd(step, total) != 1:
        step = (step + 1) % total
        if step == 0:
            step = 1
    # Sorting keeps shard access sequential while retaining a seed-dependent,
    # without-replacement subset.  Batch shuffling belongs to the trainer.
    return tuple(sorted((offset + step * index) % total for index in range(limit)))
