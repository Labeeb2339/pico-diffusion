"""Unit tests for FID math and receipt helpers (no model download needed)."""

import hashlib
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

import fid
from fid import (
    _sanitize_argv,
    _sqrtm,
    _write_receipt,
    checkpoint_architecture,
    evaluation_cache_key,
    file_sha256,
    frechet_distance,
    generate_samples,
    validate_sampling_options,
)


def test_sqrtm_is_psd_sqrt():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 8))
    S = A @ A.T
    S = (S + S.T) / 2  # symmetric PSD
    root = _sqrtm(S)
    assert np.allclose(root @ root, S, atol=1e-6)


def test_fid_identical_is_zero():
    rng = np.random.default_rng(1)
    act = rng.standard_normal((100, 16))
    mu = act.mean(axis=0)
    sig = np.cov(act, rowvar=False)
    assert frechet_distance(mu, sig, mu, sig) < 1e-9


def test_fid_increases_with_distance():
    mu = np.zeros(8)
    sig = np.eye(8)
    d_same = frechet_distance(mu, sig, mu, sig)
    d_far = frechet_distance(mu, sig, np.full(8, 5.0), sig)
    assert d_same < 1e-9
    assert d_far > 1.0  # a 5.0 mean shift is a large FID


def test_fid_different_covariances_nonnegative():
    # Regression: S1 @ S2 is NOT symmetric in general. The old code ran eigh on
    # it directly and produced a NEGATIVE "FID" (-401.51 on the real CIFAR run).
    # The fixed trace-identity path must give a non-negative, finite, symmetric
    # distance even for two unrelated covariance matrices.
    rng = np.random.default_rng(7)
    A = rng.standard_normal((20, 20))
    B = rng.standard_normal((20, 20))
    S1 = A @ A.T + 0.1 * np.eye(20)
    S2 = B @ B.T + 0.1 * np.eye(20)
    mu1 = rng.standard_normal(20)
    mu2 = rng.standard_normal(20)

    d = frechet_distance(mu1, S1, mu2, S2)
    assert np.isfinite(d)
    assert d >= 0.0, f"FID must be non-negative, got {d}"

    # Frechet distance is symmetric in its arguments.
    d2 = frechet_distance(mu2, S2, mu1, S1)
    assert abs(d - d2) < 1e-6


def test_file_sha256_matches_reference(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"pico-diffusion receipt\n")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert file_sha256(artifact) == expected


def test_cache_key_binds_batch_sizes():
    common = {
        "n": 2048,
        "seed": 0,
        "checkpoint_hash": "a" * 64,
        "channels": 3,
        "image_size": 32,
        "num_classes": None,
        "cfg_scale": 0.0,
        "sampler": "ddim",
        "steps": 50,
        "order": 2,
        "sample_batch_size": 64,
        "feature_batch_size": 64,
        "dataset_identity": {
            "canonical_backend": "content-v1",
            "source_backend": "imagefolder",
            "dataset_sha256": "b" * 64,
            "subset_sha256": "c" * 64,
        },
        "source_identity": {"fid.py": {"sha256": "d" * 64}},
        "environment_identity": {"torch": "2.11", "device": "cpu"},
    }
    reference = evaluation_cache_key(**common)
    assert evaluation_cache_key(**{**common, "sample_batch_size": 32}) != reference
    assert evaluation_cache_key(**{**common, "feature_batch_size": 32}) != reference


def test_cache_key_has_exact_cfg_dataset_source_and_environment_identity():
    common = {
        "n": 2048,
        "seed": 0,
        "checkpoint_hash": "a" * 64,
        "channels": 3,
        "image_size": 32,
        "num_classes": 10,
        "cfg_scale": 1.2345671,
        "sampler": "dpm",
        "steps": 20,
        "order": 2,
        "sample_batch_size": 64,
        "feature_batch_size": 64,
        "dataset_identity": {
            "canonical_backend": "content-v1",
            "source_backend": "imagefolder",
            "dataset_sha256": "b" * 64,
            "subset_sha256": "c" * 64,
        },
        "source_identity": {"fid.py": {"sha256": "d" * 64}},
        "environment_identity": {"torch": "2.11", "device": "cuda"},
    }
    reference = evaluation_cache_key(**common)

    # These values collide under the previous six-significant-digit formatting.
    assert evaluation_cache_key(**{**common, "cfg_scale": 1.2345672}) != reference
    assert (
        evaluation_cache_key(**{**common, "checkpoint_hash": ("a" * 12) + ("f" * 52)})
        != reference
    )
    changed_dataset = {
        **common["dataset_identity"],
        "source_backend": "torchvision-cifar10",
    }
    assert (
        evaluation_cache_key(**{**common, "dataset_identity": changed_dataset})
        != reference
    )
    assert (
        evaluation_cache_key(
            **{**common, "source_identity": {"fid.py": {"sha256": "e" * 64}}}
        )
        != reference
    )
    assert (
        evaluation_cache_key(
            **{
                **common,
                "environment_identity": {"torch": "2.12", "device": "cuda"},
            }
        )
        != reference
    )


@pytest.mark.parametrize(
    "override, match",
    [
        ({"n": 10_001}, "between 1 and 10000"),
        ({"steps": 1}, "steps must be between"),
        ({"steps": 1_001}, "steps must be between"),
        ({"cfg_scale": -0.1}, "finite and non-negative"),
        ({"cfg_scale": float("nan")}, "finite and non-negative"),
        ({"cfg_scale": float("inf")}, "finite and non-negative"),
        ({"sampler": "unknown"}, "sampler"),
        ({"order": 3}, "order must be 1 or 2"),
    ],
)
def test_sampling_option_validation_rejects_unsafe_values(override, match):
    options = {
        "n": 2,
        "steps": 50,
        "cfg_scale": 0.0,
        "sampler": "dpm",
        "order": 2,
        "timesteps": 1_000,
    }
    with pytest.raises(ValueError, match=match):
        validate_sampling_options(**{**options, **override})


def test_sampling_option_validation_accepts_supported_boundaries():
    validate_sampling_options(
        n=10_000,
        steps=1_000,
        cfg_scale=0.0,
        sampler="dpm",
        order=1,
        timesteps=1_000,
    )


def test_generate_rejects_cfg_for_unconditional_checkpoint_before_model_load(
    tmp_path,
):
    checkpoint = tmp_path / "unconditional.pt"
    torch.save(
        {"model": {"in_conv.weight": torch.zeros(8, 3, 3, 3)}},
        checkpoint,
    )

    with pytest.raises(ValueError, match="unconditional checkpoint"):
        generate_samples(
            checkpoint,
            n=1,
            channels=3,
            image_size=32,
            device=torch.device("cpu"),
            steps=2,
            cfg_scale=1.0,
        )


def test_checkpoint_architecture_detects_conditioning(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "ema": {
                "in_conv.weight": torch.zeros(64, 3, 3, 3),
                "class_emb.weight": torch.zeros(11, 256),
            }
        },
        checkpoint,
    )
    assert checkpoint_architecture(checkpoint) == {
        "channels": 3,
        "base_ch": 64,
        "num_classes": 10,
    }


def test_receipt_binds_identity_without_leaking_paths(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    array = tmp_path / "generated.npy"
    array.write_bytes(b"array")
    receipt = tmp_path / "receipt.json"
    args = Namespace(
        ckpt=str(checkpoint),
        n=8,
        seed=0,
        channels=3,
        image_size=32,
        steps=2,
        sampler="ddim",
        order=2,
        sample_batch_size=4,
        batch_size=4,
        num_classes=None,
        cfg_scale=0.0,
        no_cache=True,
    )
    artifacts = {
        "generated_images": {
            "path": str(array),
            "bytes": array.stat().st_size,
            "sha256": file_sha256(array),
            "cache_hit": False,
        }
    }
    dataset_identity = {
        "canonical_backend": "content-v1",
        "source_backend": "torchvision.datasets.CIFAR10",
        "dataset_sha256": "d" * 64,
        "subset_sha256": "e" * 64,
        "selected_sample_sha256": ["f" * 64],
    }
    monkeypatch.setattr(fid, "_inception_weights_snapshot", lambda: {"enum": "test"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(tmp_path / "fid.py"),
            "--ckpt",
            str(checkpoint),
            f"--receipt={receipt}",
        ],
    )
    _write_receipt(
        receipt,
        args=args,
        device=torch.device("cpu"),
        checkpoint_hash=file_sha256(checkpoint),
        artifacts=artifacts,
        real_dataset_identity=dataset_identity,
        score=12.5,
        self_score=0.0,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["checkpoint"]["sha256"] == file_sha256(checkpoint)
    assert payload["evaluation"]["no_cache_requested"] is True
    assert payload["evaluation"]["real_dataset"] == dataset_identity
    assert payload["artifacts"]["generated_images"]["cache_hit"] is False
    assert payload["results"]["internal_fid_style_score"] == 12.5
    assert not Path(payload["checkpoint"]["path"]).is_absolute()
    assert not Path(payload["artifacts"]["generated_images"]["path"]).is_absolute()
    serialized = json.dumps(payload)
    assert str(tmp_path) not in serialized
    assert all(str(tmp_path) not in argument for argument in payload["argv"])


def test_argv_sanitizer_redacts_external_absolute_paths():
    external = Path(tempfile.gettempdir()).resolve() / "pico-private"
    sanitized = _sanitize_argv(
        [
            str(external / "fid.py"),
            "--ckpt",
            str(external / "checkpoint.pt"),
            f"--receipt={external / 'receipt.json'}",
        ]
    )
    assert sanitized == [
        "<external-path>",
        "--ckpt",
        "<external-path>",
        "--receipt=<external-path>",
    ]
