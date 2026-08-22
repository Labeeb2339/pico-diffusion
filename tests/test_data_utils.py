"""Tests for fresh-clone and local-mirror dataset routing."""

import numpy as np
import torch
import torchvision
from PIL import Image

import data_utils
from data_utils import canonical_cifar10_evaluation_subset, cifar10_dataset


def test_cifar_loader_prefers_existing_imagefolder(tmp_path, monkeypatch) -> None:
    mirror = tmp_path / "cifar10" / "train"
    mirror.mkdir(parents=True)
    sentinel = object()
    called = {}

    def fake_imagefolder(*, root, transform):
        called.update(root=root, transform=transform)
        return sentinel

    monkeypatch.setattr(torchvision.datasets, "ImageFolder", fake_imagefolder)
    result = cifar10_dataset(train=True, transform="tf", root=str(tmp_path))

    assert result is sentinel
    assert called == {"root": str(mirror), "transform": "tf"}


def test_cifar_loader_downloads_standard_archive_when_mirror_missing(
    tmp_path, monkeypatch
) -> None:
    sentinel = object()
    called = {}

    def fake_cifar10(*, root, train, download, transform):
        called.update(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
        return sentinel

    monkeypatch.setattr(torchvision.datasets, "CIFAR10", fake_cifar10)
    result = cifar10_dataset(train=False, transform="tf", root=str(tmp_path))

    assert result is sentinel
    assert called == {
        "root": str(tmp_path),
        "train": False,
        "download": True,
        "transform": "tf",
    }


class _TinyDataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, label = self.samples[index]
        return image.copy(), label


def _sample(value: int, label: int):
    pixels = np.full((32, 32, 3), value, dtype=np.uint8)
    pixels[0, 0] = np.array([value, (value + 1) % 256, (value + 2) % 256])
    return Image.fromarray(pixels, mode="RGB"), label


def test_canonical_subset_is_content_stable_across_dataset_backends(
    tmp_path, monkeypatch
) -> None:
    samples = [_sample(11, 0), _sample(37, 1), _sample(83, 2), _sample(149, 3)]
    imagefolder_order = [samples[index] for index in (2, 0, 3, 1)]
    archive_order = [samples[index] for index in (1, 3, 0, 2)]

    local_root = tmp_path / "local"
    (local_root / "cifar10" / "test").mkdir(parents=True)
    fresh_root = tmp_path / "fresh"
    fresh_root.mkdir()

    monkeypatch.setattr(data_utils, "_CIFAR10_TEST_SIZE", len(samples))
    monkeypatch.setattr(
        torchvision.datasets,
        "ImageFolder",
        lambda **_kwargs: _TinyDataset(imagefolder_order),
    )
    monkeypatch.setattr(
        torchvision.datasets,
        "CIFAR10",
        lambda **_kwargs: _TinyDataset(archive_order),
    )

    local_images, local_identity = canonical_cifar10_evaluation_subset(
        n=3, seed=19, root=str(local_root)
    )
    fresh_images, fresh_identity = canonical_cifar10_evaluation_subset(
        n=3, seed=19, root=str(fresh_root)
    )

    assert torch.equal(local_images, fresh_images)
    assert local_identity["dataset_sha256"] == fresh_identity["dataset_sha256"]
    assert local_identity["subset_sha256"] == fresh_identity["subset_sha256"]
    assert (
        local_identity["selected_sample_sha256"]
        == fresh_identity["selected_sample_sha256"]
    )
    assert local_identity["source_backend"] != fresh_identity["source_backend"]
    assert local_identity["canonical_backend"] == fresh_identity["canonical_backend"]


def test_canonical_subset_rejects_incomplete_test_split(tmp_path, monkeypatch) -> None:
    mirror = tmp_path / "cifar10" / "test"
    mirror.mkdir(parents=True)
    monkeypatch.setattr(
        torchvision.datasets,
        "ImageFolder",
        lambda **_kwargs: _TinyDataset([_sample(1, 0)]),
    )

    try:
        canonical_cifar10_evaluation_subset(n=1, root=str(tmp_path))
    except ValueError as error:
        assert "complete test split" in str(error)
    else:
        raise AssertionError("incomplete CIFAR-10 split was accepted")
