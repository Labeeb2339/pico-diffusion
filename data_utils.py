"""Dataset loading shared by training and evaluation scripts."""

import hashlib
import struct
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as TF

_CIFAR10_TEST_SIZE = 10_000
_CIFAR10_EVAL_BACKEND = "pico-cifar10-rgb-content-v1"


def cifar10_dataset(*, train: bool, transform, root: str = "./data"):
    """Load a local ImageFolder mirror, or download torchvision CIFAR-10.

    The original experiments used ``data/cifar10/{train,test}/<class>/*.png``.
    A fresh clone instead falls back to torchvision's standard archive so the
    documented quickstart does not require an undocumented preparation step.
    """
    split = "train" if train else "test"
    mirror_root = Path(root) / "cifar10" / split
    if mirror_root.is_dir():
        return torchvision.datasets.ImageFolder(
            root=str(mirror_root),
            transform=transform,
        )
    return torchvision.datasets.CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=transform,
    )


def _rgb_uint8(image) -> np.ndarray:
    """Convert a dataset image to canonical contiguous RGB uint8 pixels."""
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(
                f"expected a 3D image tensor, got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        if tensor.is_floating_point():
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("image tensor contains non-finite values")
            tensor = (tensor.clamp(0, 1) * 255).round().to(torch.uint8)
        else:
            tensor = tensor.to(torch.uint8)
        array = tensor.numpy()
    elif hasattr(image, "convert"):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(image, dtype=np.uint8)

    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError(f"expected an RGB-compatible image, got shape {array.shape}")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 4:
        array = array[:, :, :3]
    return np.array(array, dtype=np.uint8, order="C", copy=True)


def canonical_sample_sha256(image, label: int) -> str:
    """Hash label, dimensions, and decoded RGB pixels independent of file format."""
    array = _rgb_uint8(image)
    height, width, channels = array.shape
    digest = hashlib.sha256()
    digest.update(b"pico-cifar10-sample-v1\0")
    digest.update(struct.pack("<IIIi", height, width, channels, int(label)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _hash_sequence(values: list[str], *, domain: bytes) -> str:
    digest = hashlib.sha256(domain + b"\0")
    for value in values:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _raw_cifar10_test_dataset(root: str):
    mirror_root = Path(root) / "cifar10" / "test"
    if mirror_root.is_dir():
        return (
            torchvision.datasets.ImageFolder(root=str(mirror_root), transform=None),
            "torchvision.datasets.ImageFolder",
        )
    return (
        torchvision.datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=None,
        ),
        "torchvision.datasets.CIFAR10",
    )


def canonical_cifar10_evaluation_subset(
    *, n: int, seed: int = 0, root: str = "./data"
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return a backend-independent CIFAR-10 test subset and its identity.

    ImageFolder mirrors group samples by class while torchvision's archive keeps
    the original test-batch order.  We therefore order the complete test corpus
    by a hash of decoded RGB pixels plus label before applying the seeded subset
    selection.  Equivalent mirrors and archives select the same image content.
    """
    if n < 1 or n > _CIFAR10_TEST_SIZE:
        raise ValueError(f"n must be between 1 and {_CIFAR10_TEST_SIZE}")

    dataset, source_backend = _raw_cifar10_test_dataset(root)
    if len(dataset) != _CIFAR10_TEST_SIZE:
        raise ValueError(
            "canonical CIFAR-10 evaluation requires the complete test split: "
            f"expected {_CIFAR10_TEST_SIZE} images, found {len(dataset)}"
        )

    records: list[tuple[str, int]] = []
    for source_index in range(len(dataset)):
        image, label = dataset[source_index]
        records.append((canonical_sample_sha256(image, int(label)), source_index))
    records.sort(key=lambda item: item[0])

    ordered_hashes = [sample_hash for sample_hash, _ in records]
    selected_positions = np.random.default_rng(seed).choice(
        len(records), n, replace=False
    )
    selected_records = [records[int(position)] for position in selected_positions]
    selected_hashes = [sample_hash for sample_hash, _ in selected_records]

    images = []
    for expected_hash, source_index in selected_records:
        image, label = dataset[source_index]
        if canonical_sample_sha256(image, int(label)) != expected_hash:
            raise RuntimeError("CIFAR-10 source changed while selecting the subset")
        images.append(TF.to_tensor(_rgb_uint8(image)))

    identity: dict[str, object] = {
        "canonical_backend": _CIFAR10_EVAL_BACKEND,
        "source_backend": source_backend,
        "split": "test",
        "dataset_samples": len(records),
        "dataset_sha256": _hash_sequence(
            ordered_hashes, domain=b"pico-cifar10-dataset-v1"
        ),
        "subset_seed": seed,
        "subset_samples": n,
        "subset_sha256": _hash_sequence(
            selected_hashes, domain=b"pico-cifar10-subset-v1"
        ),
        "selected_sample_sha256": selected_hashes,
        "selection": "numpy-pcg64-choice-over-content-hash-order-v1",
    }
    return torch.stack(images), identity
