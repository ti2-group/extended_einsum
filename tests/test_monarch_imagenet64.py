from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import experiments.monarch.imagenet64 as imagenet64
from experiments.monarch.imagenet64 import (
    ImageNet64Dataset,
    ImageNet64FormatError,
    load_imagenet64_images,
    num_categories_for_transform,
    num_channels_for_transform,
    rgb_to_grayscale,
    rgb_to_quantized_ycocg,
    rgb_to_ycocg_r,
    ycocg_r_to_rgb,
)


def test_rgb_to_grayscale_uses_deterministic_integer_luminance() -> None:
    rgb = torch.tensor(
        [
            [[0, 255, 0, 0, 255]],
            [[0, 0, 255, 0, 255]],
            [[0, 0, 0, 255, 255]],
        ],
        dtype=torch.uint8,
    )

    grayscale = rgb_to_grayscale(rgb)

    assert grayscale.shape == (1, 1, 5)
    assert grayscale.tolist() == [[[0, 77, 149, 29, 255]]]


def make_images(values: list[int]) -> np.ndarray:
    images = np.empty((len(values), 3, 64, 64), dtype=np.uint8)
    for index, value in enumerate(values):
        images[index].fill(value)
    return images


def write_shard(path: Path, images: np.ndarray, *, labels: np.ndarray | None = None, layout: str = "flat") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if layout == "flat":
        data = images.reshape(len(images), -1)
    elif layout == "chw":
        data = images
    elif layout == "hwc":
        data = images.transpose(0, 2, 3, 1)
    else:
        raise AssertionError(layout)
    if labels is None:
        labels = np.arange(1, len(images) + 1, dtype=np.int64)
    np.savez(path, data=data, labels=labels)


def make_official_tree(root: Path, train_values: list[int] | None = None, val_values: list[int] | None = None) -> None:
    if train_values is None:
        train_values = [3, 5, 7]
    if val_values is None:
        val_values = [11, 13]
    write_shard(root / "imagenet64" / "train" / "train_data_batch_1.npz", make_images(train_values))
    write_shard(root / "imagenet64" / "val" / "val_data.npz", make_images(val_values))


def test_loads_official_repo_layout_and_flattens_chw(tmp_path: Path) -> None:
    make_official_tree(tmp_path)

    images = load_imagenet64_images(tmp_path, color_transform="rgb", patch_size=None)

    assert images.shape == (3, 3 * 64 * 64)
    assert images.dtype == torch.long
    assert torch.equal(images[:, 0], torch.tensor([3, 5, 7]))
    assert torch.equal(images[:, 64 * 64], torch.tensor([3, 5, 7]))


def test_discovers_download_archive_names_and_naturally_sorts_batches(tmp_path: Path) -> None:
    train_dir = tmp_path / "Imagenet64_train_npz"
    write_shard(train_dir / "train_data_batch_10.npz", make_images([10]))
    write_shard(train_dir / "train_data_batch_2.npz", make_images([2]))

    dataset = ImageNet64Dataset(tmp_path, color_transform="rgb", patch_size=None, output_layout="chw")

    assert [path.name for path in dataset.shard_paths] == ["train_data_batch_2.npz", "train_data_batch_10.npz"]
    assert [int(dataset[index][0, 0, 0]) for index in range(len(dataset))] == [2, 10]


@pytest.mark.parametrize("stored_layout", ["flat", "chw", "hwc"])
def test_accepts_supported_storage_layouts(tmp_path: Path, stored_layout: str) -> None:
    image = np.arange(3 * 64 * 64, dtype=np.uint16).reshape(1, 3, 64, 64) % 256
    image = image.astype(np.uint8)
    write_shard(tmp_path / "train" / "part.npz", image, layout=stored_layout)

    chw = ImageNet64Dataset(tmp_path, color_transform="rgb", patch_size=None, output_layout="chw")[0]
    hwc = ImageNet64Dataset(tmp_path, color_transform="rgb", patch_size=None, output_layout="hwc")[0]

    assert torch.equal(chw, torch.from_numpy(image[0]).long())
    assert torch.equal(hwc, torch.from_numpy(image[0].transpose(1, 2, 0)).long())


@pytest.mark.parametrize("stored_layout", ["flat", "chw", "hwc"])
@pytest.mark.parametrize("color_transform", ["rgb", "grayscale", "ycocg-r", "ycocg"])
@pytest.mark.parametrize("patch_mode", ["random", "all"])
@pytest.mark.parametrize("output_layout", ["flat", "chw", "hwc"])
def test_numpy_patch_fast_path_matches_full_image_reference(
    tmp_path: Path,
    stored_layout: str,
    color_transform: str,
    patch_mode: str,
    output_layout: str,
) -> None:
    image = (np.arange(3 * 64 * 64, dtype=np.uint16).reshape(1, 3, 64, 64) * 17 % 256).astype(np.uint8)
    write_shard(tmp_path / "train" / "images.npz", image, layout=stored_layout)
    sample_seed = 29
    epoch = 4
    dataset = ImageNet64Dataset(
        tmp_path,
        color_transform=color_transform,
        patch_size=8,
        patch_mode=patch_mode,
        sample_seed=sample_seed,
        epoch=epoch,
        output_layout=output_layout,
    )
    dataset_index = 37 if patch_mode == "all" else 0

    # This intentionally reproduces the former implementation: copy and cast
    # the complete image first, then select a patch in Torch.
    reference = torch.from_numpy(np.array(image[0], copy=True)).to(torch.long)
    patch_index = dataset_index
    if patch_mode == "random":
        patch_index = imagenet64._random_patch_index(
            source_index=0,
            epoch=epoch,
            seed=sample_seed,
            patches_per_side=8,
        )
    patch_row, patch_column = divmod(patch_index, 8)
    reference = reference[
        :,
        patch_row * 8 : (patch_row + 1) * 8,
        patch_column * 8 : (patch_column + 1) * 8,
    ]
    if color_transform == "grayscale":
        reference = rgb_to_grayscale(reference)
    elif color_transform == "ycocg-r":
        reference = rgb_to_ycocg_r(reference)
    elif color_transform == "ycocg":
        reference = rgb_to_quantized_ycocg(reference)
    if output_layout == "flat":
        reference = reference.reshape(-1).contiguous()
    elif output_layout == "hwc":
        reference = reference.permute(1, 2, 0).contiguous()
    else:
        reference = reference.contiguous()

    actual = dataset[dataset_index]

    assert actual.dtype == torch.long
    assert actual.is_contiguous()
    assert torch.equal(actual, reference)


@pytest.mark.parametrize("stored_layout", ["flat", "chw", "hwc"])
def test_all_patch_mode_copies_only_the_numpy_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stored_layout: str,
) -> None:
    image = np.arange(3 * 64 * 64, dtype=np.uint16).reshape(1, 3, 64, 64).astype(np.uint8)
    write_shard(tmp_path / "train" / "images.npz", image, layout=stored_layout)
    dataset = ImageNet64Dataset(tmp_path, color_transform="rgb", patch_size=8, patch_mode="all")
    copied_shapes: list[tuple[int, ...]] = []
    original_copy = imagenet64._copy_chw_to_long

    def tracked_copy(chw_view: np.ndarray) -> torch.Tensor:
        copied_shapes.append(chw_view.shape)
        return original_copy(chw_view)

    monkeypatch.setattr(imagenet64, "_copy_chw_to_long", tracked_copy)

    _ = dataset[0]
    _ = dataset[63]

    assert copied_shapes == [(3, 8, 8), (3, 8, 8)]


def test_validation_and_test_both_use_the_official_held_out_split(tmp_path: Path) -> None:
    make_official_tree(tmp_path)

    validation = ImageNet64Dataset(tmp_path, split="validation", color_transform="rgb", patch_size=None)
    test = ImageNet64Dataset(tmp_path, split="test", color_transform="rgb", patch_size=None)

    assert validation.resolved_split == "validation"
    assert test.resolved_split == "validation"
    assert validation.shard_paths == test.shard_paths
    assert torch.equal(validation[0], test[0])


def test_lossless_transform_is_exactly_invertible_and_uses_512_categories() -> None:
    generator = torch.Generator().manual_seed(123)
    rgb = torch.randint(0, 256, (3, 8, 8), generator=generator, dtype=torch.long)

    transformed = rgb_to_ycocg_r(rgb)

    assert transformed.dtype == torch.long
    assert int(transformed.min()) >= 0
    assert int(transformed.max()) < 512
    assert torch.equal(ycocg_r_to_rgb(transformed), rgb)
    assert num_categories_for_transform("ycocg-r") == 512
    assert num_categories_for_transform("rgb") == 256
    assert num_categories_for_transform("grayscale") == 256
    assert num_categories_for_transform("ycocg") == 256
    assert num_channels_for_transform("grayscale") == 1
    assert num_channels_for_transform("rgb") == 3


def test_lossy_transform_matches_released_formula_and_is_categorical() -> None:
    rgb = torch.tensor(
        [
            [[0, 255], [255, 0]],
            [[0, 255], [0, 255]],
            [[0, 255], [255, 0]],
        ],
        dtype=torch.long,
    )

    transformed = rgb_to_quantized_ycocg(rgb)

    assert transformed.dtype == torch.long
    assert int(transformed.min()) >= 0
    assert int(transformed.max()) <= 255
    assert transformed.shape == rgb.shape
    assert torch.equal(
        transformed,
        torch.tensor(
            [
                [[0, 255], [128, 128]],
                [[128, 128], [128, 128]],
                [[128, 128], [0, 255]],
            ]
        ),
    )


def test_all_patches_are_aligned_and_row_major(tmp_path: Path) -> None:
    image = np.zeros((1, 3, 64, 64), dtype=np.uint8)
    rows = np.arange(64, dtype=np.uint8)[:, None]
    columns = np.arange(64, dtype=np.uint8)[None, :]
    image[0, 0] = rows
    image[0, 1] = columns
    image[0, 2] = rows + columns
    write_shard(tmp_path / "train" / "images.npz", image)

    patches = load_imagenet64_images(
        tmp_path,
        color_transform="rgb",
        patch_size=8,
        patch_mode="all",
        output_layout="chw",
    )

    assert patches.shape == (64, 3, 8, 8)
    assert torch.equal(patches[0], torch.from_numpy(image[0, :, :8, :8]).long())
    assert torch.equal(patches[1], torch.from_numpy(image[0, :, :8, 8:16]).long())
    assert torch.equal(patches[8], torch.from_numpy(image[0, :, 8:16, :8]).long())


def test_random_patch_is_reproducible_and_epoch_dependent(tmp_path: Path) -> None:
    image = np.zeros((1, 3, 64, 64), dtype=np.uint8)
    for patch_row in range(8):
        for patch_column in range(8):
            image[:, :, patch_row * 8 : (patch_row + 1) * 8, patch_column * 8 : (patch_column + 1) * 8] = patch_row * 8 + patch_column
    write_shard(tmp_path / "train" / "images.npz", image)
    dataset = ImageNet64Dataset(tmp_path, color_transform="rgb", patch_size=8, patch_mode="random", sample_seed=17, output_layout="chw")

    first = dataset[0].clone()
    assert torch.equal(first, dataset[0])
    dataset.set_epoch(1)

    assert not torch.equal(first, dataset[0])


def test_sample_limit_is_deterministic_and_counts_returned_patches(tmp_path: Path) -> None:
    make_official_tree(tmp_path, train_values=list(range(20)))
    kwargs = dict(color_transform="rgb", patch_size=None, sample_limit=5)

    first = load_imagenet64_images(tmp_path, sample_seed=4, **kwargs)
    repeated = load_imagenet64_images(tmp_path, sample_seed=4, **kwargs)
    different = load_imagenet64_images(tmp_path, sample_seed=5, **kwargs)
    patches = ImageNet64Dataset(tmp_path, color_transform="rgb", patch_size=8, patch_mode="all", sample_limit=7, sample_seed=4)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)
    assert len(patches) == 7


def test_epoch_indices_shuffle_deterministically_without_interleaving_shards(tmp_path: Path) -> None:
    train_dir = tmp_path / "imagenet64" / "train"
    write_shard(train_dir / "train_data_batch_1.npz", make_images([1, 2, 3]))
    write_shard(train_dir / "train_data_batch_2.npz", make_images([4, 5, 6, 7]))
    dataset = ImageNet64Dataset(tmp_path, color_transform="rgb", patch_size=None)

    first = dataset.epoch_indices(seed=19, epoch=0)
    repeated = dataset.epoch_indices(seed=19, epoch=0)
    next_epoch = dataset.epoch_indices(seed=19, epoch=1)

    assert first.device.type == "cpu"
    assert first.dtype == torch.long
    assert torch.equal(first, repeated)
    assert not torch.equal(first, next_epoch)
    assert torch.equal(first.sort().values, torch.arange(len(dataset)))
    shard_ids = [0 if index < 3 else 1 for index in first.tolist()]
    assert sum(left != right for left, right in zip(shard_ids, shard_ids[1:], strict=False)) == 1


@pytest.mark.parametrize(("patch_mode", "sample_limit"), [("random", 5), ("all", 100)])
def test_epoch_indices_are_shard_local_for_selected_patch_subsets(tmp_path: Path, patch_mode: str, sample_limit: int) -> None:
    train_dir = tmp_path / "imagenet64" / "train"
    write_shard(train_dir / "train_data_batch_1.npz", make_images([1, 2, 3]))
    write_shard(train_dir / "train_data_batch_2.npz", make_images([4, 5, 6, 7]))
    dataset = ImageNet64Dataset(
        tmp_path,
        color_transform="rgb",
        patch_size=8,
        patch_mode=patch_mode,
        sample_limit=sample_limit,
        sample_seed=23,
    )

    order = dataset.epoch_indices(seed=31, epoch=2)

    assert torch.equal(order.sort().values, torch.arange(len(dataset)))
    logical_indices = [dataset.selected_indices[index] for index in order.tolist()]
    source_indices = [logical // dataset.patches_per_image for logical in logical_indices]
    shard_ids = [0 if source < 3 else 1 for source in source_indices]
    assert set(shard_ids) == {0, 1}
    assert sum(left != right for left, right in zip(shard_ids, shard_ids[1:], strict=False)) == 1


def test_labels_remain_available_as_metadata(tmp_path: Path) -> None:
    labels = np.array([101, 202], dtype=np.int64)
    write_shard(tmp_path / "train" / "images.npz", make_images([1, 2]), labels=labels)
    dataset = ImageNet64Dataset(tmp_path, color_transform="rgb", patch_size=8, patch_mode="all")

    assert dataset.label_at(0) == 101
    assert dataset.label_at(63) == 101
    assert dataset.label_at(64) == 202


def test_rejects_unsupported_shard_shape(tmp_path: Path) -> None:
    path = tmp_path / "train" / "bad.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, data=np.zeros((2, 12), dtype=np.uint8), labels=np.array([1, 2]))
    dataset = ImageNet64Dataset(tmp_path, color_transform="rgb", patch_size=None)

    with pytest.raises(ImageNet64FormatError, match="Unsupported data shape"):
        _ = dataset[0]


def test_missing_split_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ImageNet64 train NPZ shards"):
        ImageNet64Dataset(tmp_path)
