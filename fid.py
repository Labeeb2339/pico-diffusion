"""A reproducible FID-style evaluation for the diffusion model.

This harness measures the Fréchet distance between torchvision InceptionV3
feature distributions for real and generated images. Lower is better and a
perfect generator scores ~0. The default 2,048-sample result is an internal
comparison metric, not canonical 50,000-sample CIFAR-10 FID.

Uses torchvision's pretrained InceptionV3 (features from the final pooling
layer) and a dependency-free Frechet distance (via eigendecomposition, no scipy).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3

from data_utils import canonical_cifar10_evaluation_subset
from diffusion import GaussianDiffusion
from model import UNet

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
_CACHE_VERSION = "v3"
_METRIC_ID = "pico-diffusion-fid-style-v1"
_REPO_ROOT = Path(__file__).resolve().parent


def get_inception(device) -> nn.Module:
    """InceptionV3 returning 2048-d features (the avgpool output)."""
    model = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False,
    )
    model.fc = nn.Identity()
    return model.to(device).eval()


@torch.no_grad()
def compute_activations(
    model, images: torch.Tensor, device, batch_size: int = 64
) -> np.ndarray:
    """Extract 2048-d Inception features for a batch of images in [0, 1]."""
    mean = _IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
    std = _IMAGENET_STD.to(device).view(1, 3, 1, 1)
    feats = []
    for i in range(0, images.shape[0], batch_size):
        x = images[i : i + batch_size].to(device)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        feats.append(model(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


def _sqrtm(mat: np.ndarray) -> np.ndarray:
    """Matrix square root of a SYMMETRIC PSD matrix (no scipy dependency).

    Only valid when ``mat`` is symmetric: ``eigh`` reads just the lower
    triangle, so passing a non-symmetric matrix silently returns wrong values.
    """
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def _regularize(sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Keep covariance conditioned when sample and feature counts are similar."""
    return sigma + eps * np.eye(sigma.shape[0])


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    """Frechet distance between two Gaussians, dependency-free.

    FID = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 sqrt(S1 S2)).

    ``S1 @ S2`` is NOT symmetric in general, so we cannot ``eigh`` it directly.
    We instead use the trace identity

        Tr(sqrt(S1 S2)) = Tr(sqrt(S1^(1/2) S2 S1^(1/2)))

    where ``S1 S2`` is *similar* to the symmetric PSD matrix
    ``S1^(1/2) S2 S1^(1/2)``, so ``eigh`` is valid there. This avoids both
    scipy and a matrix inverse (we only ever need the trace).
    """
    diff = mu1 - mu2
    sigma1 = _regularize(sigma1)
    sigma2 = _regularize(sigma2)
    s1_sqrt = _sqrtm(sigma1)  # symmetric PSD -> eigh valid
    middle = s1_sqrt @ sigma2 @ s1_sqrt  # symmetric PSD -> eigh valid
    covmean_trace = float(np.trace(_sqrtm(middle)))
    score = float(
        diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * covmean_trace
    )
    return max(0.0, score)  # suppress tiny negative round-off for identical sets


def fid_from_activations(act1: np.ndarray, act2: np.ndarray) -> float:
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


def load_real_cifar(
    n: int,
    *,
    seed: int = 0,
    root: str = "./data",
    return_identity: bool = False,
):
    """Load a content-canonical CIFAR-10 test subset in ``[0, 1]``."""
    images, identity = canonical_cifar10_evaluation_subset(
        n=n,
        seed=seed,
        root=root,
    )
    return (images, identity) if return_identity else images


def validate_sampling_options(
    *,
    n: int,
    steps: int,
    cfg_scale: float,
    sampler: str,
    order: int,
    timesteps: int,
) -> None:
    """Reject ambiguous or numerically unsafe evaluation settings."""
    if n < 1 or n > 10_000:
        raise ValueError("n must be between 1 and 10000")
    if steps < 2 or steps > timesteps:
        raise ValueError(f"steps must be between 2 and {timesteps}")
    if not math.isfinite(cfg_scale) or cfg_scale < 0:
        raise ValueError("cfg_scale must be finite and non-negative")
    if sampler not in {"ddim", "dpm"}:
        raise ValueError("sampler must be 'ddim' or 'dpm'")
    if order not in {1, 2}:
        raise ValueError("DPM-Solver order must be 1 or 2")


@torch.no_grad()
def generate_samples(
    ckpt_path,
    n,
    channels,
    image_size,
    device,
    steps=50,
    num_classes=None,
    cfg_scale=0.0,
    sampler="ddim",
    order=2,
    sample_batch_size=64,
    seed=0,
):
    if sample_batch_size < 1:
        raise ValueError("sample_batch_size must be at least 1")
    diffusion = GaussianDiffusion()
    validate_sampling_options(
        n=n,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler=sampler,
        order=order,
        timesteps=diffusion.timesteps,
    )
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ck["ema"] if "ema" in ck else ck["model"]
    checkpoint_num_classes = (
        int(state["class_emb.weight"].shape[0]) - 1
        if "class_emb.weight" in state
        else None
    )
    if num_classes is not None and num_classes != checkpoint_num_classes:
        raise ValueError(
            f"num_classes={num_classes} does not match checkpoint "
            f"({checkpoint_num_classes})"
        )
    num_classes = checkpoint_num_classes
    if num_classes is None and cfg_scale != 0.0:
        raise ValueError("cfg_scale must be 0 for an unconditional checkpoint")
    base_ch = int(state["in_conv.weight"].shape[0])
    model = UNet(
        in_channels=channels,
        base_ch=base_ch,
        image_size=image_size,
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    diffusion.to(device)
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)

    batches = []
    for start in range(0, n, sample_batch_size):
        batch_n = min(sample_batch_size, n - start)
        y = None
        if num_classes is not None:
            # Uniform random labels match the balanced real CIFAR-10 distribution.
            y = torch.randint(0, num_classes, (batch_n,), device=device)
        shape = (batch_n, channels, image_size, image_size)
        if sampler == "dpm":
            x = diffusion.dpm_solver_sample(
                model,
                shape,
                device,
                sampling_steps=steps,
                order=order,
                y=y,
                w=cfg_scale,
            )
        else:
            x = diffusion.ddim_sample(
                model,
                shape,
                device,
                sampling_steps=steps,
                y=y,
                w=cfg_scale,
            )
        batches.append(torch.clamp((x + 1) / 2, 0, 1).cpu())
    return torch.cat(batches, dim=0)


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Return the full SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(path: str | os.PathLike[str]) -> str:
    """Full content hash used to prevent stale cross-checkpoint cache hits."""
    return file_sha256(path)


def checkpoint_architecture(path: str | os.PathLike[str]) -> dict[str, int | None]:
    """Read the model dimensions needed to validate evaluation arguments."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"]
    in_conv = state["in_conv.weight"]
    return {
        "channels": int(in_conv.shape[1]),
        "base_ch": int(in_conv.shape[0]),
        "num_classes": (
            int(state["class_emb.weight"].shape[0]) - 1
            if "class_emb.weight" in state
            else None
        ),
    }


def evaluation_cache_key(
    *,
    n: int,
    seed: int,
    checkpoint_hash: str,
    channels: int,
    image_size: int,
    num_classes: int | None,
    cfg_scale: float,
    sampler: str,
    steps: int,
    order: int,
    sample_batch_size: int,
    feature_batch_size: int,
    dataset_identity: dict[str, object],
    source_identity: dict[str, object],
    environment_identity: dict[str, object],
) -> str:
    """Hash every generator, dataset, source, and environment cache input."""
    if not math.isfinite(cfg_scale):
        raise ValueError("cfg_scale must be finite")
    identity = {
        "cache_version": _CACHE_VERSION,
        "metric_id": _METRIC_ID,
        "n": n,
        "seed": seed,
        "checkpoint_sha256": checkpoint_hash,
        "channels": channels,
        "image_size": image_size,
        "num_classes": num_classes,
        "cfg_scale_hex": float(cfg_scale).hex(),
        "sampler": sampler,
        "steps": steps,
        "order": order,
        "sample_batch_size": sample_batch_size,
        "feature_batch_size": feature_batch_size,
        "real_dataset": dataset_identity,
        "source": source_identity,
        "environment": environment_identity,
    }
    return f"{_CACHE_VERSION}_{_identity_sha256(identity)}"


def _identity_sha256(identity: object) -> str:
    payload = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _derived_cache_key(kind: str, identity: object) -> str:
    return f"{_CACHE_VERSION}_{kind}_{_identity_sha256(identity)}"


def _relative_path(path: str | os.PathLike[str]) -> str:
    """Return a repository-relative label without exposing external paths."""
    if str(path).startswith("<"):
        return str(path)
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return "<external-path>"


def _sanitize_argv(argv: list[str]) -> list[str]:
    """Redact absolute and external checkpoint/receipt paths from an argv copy."""
    path_options = {"--ckpt", "--receipt"}
    sanitized: list[str] = []
    expect_path = False
    for index, argument in enumerate(argv):
        if expect_path:
            sanitized.append(_relative_path(argument))
            expect_path = False
            continue
        if argument in path_options:
            sanitized.append(argument)
            expect_path = True
            continue
        matched = False
        for option in path_options:
            prefix = option + "="
            if argument.startswith(prefix):
                sanitized.append(prefix + _relative_path(argument[len(prefix) :]))
                matched = True
                break
        if matched:
            continue
        if Path(argument).is_absolute() or (
            index == 0 and ("/" in argument or "\\" in argument)
        ):
            sanitized.append(_relative_path(argument))
        else:
            sanitized.append(argument)
    return sanitized


def _git_snapshot() -> dict[str, object]:
    """Return the current Git identity without making Git a runtime requirement."""
    repo_dir = Path(__file__).resolve().parent

    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def _artifact_record(
    path: str | os.PathLike[str], *, cache_hit: bool
) -> dict[str, object]:
    artifact = Path(path)
    return {
        "path": _relative_path(artifact),
        "bytes": artifact.stat().st_size,
        "sha256": file_sha256(artifact),
        "cache_hit": cache_hit,
    }


def _environment_snapshot(device: torch.device) -> dict[str, object]:
    gpu_name = None
    gpu_capability = None
    gpu_memory_bytes = None
    nvidia_driver = None
    cudnn_version = None
    if device.type == "cuda":
        device_index = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        properties = torch.cuda.get_device_properties(device_index)
        gpu_name = properties.name
        gpu_memory_bytes = properties.total_memory
        gpu_capability = list(torch.cuda.get_device_capability(device_index))
        cudnn_version = torch.backends.cudnn.version()
        try:
            driver_result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={device_index}",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            nvidia_driver = driver_result.stdout.strip().splitlines()[0]
        except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
            nvidia_driver = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": cudnn_version,
        "device": str(device),
        "gpu": gpu_name,
        "gpu_memory_bytes": gpu_memory_bytes,
        "gpu_compute_capability": gpu_capability,
        "nvidia_driver": nvidia_driver,
    }


def _source_snapshot() -> dict[str, dict[str, object]]:
    repo_dir = Path(__file__).resolve().parent
    source_files = ("fid.py", "diffusion.py", "model.py", "data_utils.py")
    return {
        name: {
            "bytes": (repo_dir / name).stat().st_size,
            "sha256": file_sha256(repo_dir / name),
        }
        for name in source_files
        if (repo_dir / name).is_file()
    }


def _inception_weights_snapshot() -> dict[str, object]:
    weights = Inception_V3_Weights.IMAGENET1K_V1
    filename = Path(urlparse(weights.url).path).name
    cached_file = Path(torch.hub.get_dir()) / "checkpoints" / filename
    return {
        "enum": "Inception_V3_Weights.IMAGENET1K_V1",
        "url": weights.url,
        "filename": filename,
        "bytes": cached_file.stat().st_size if cached_file.is_file() else None,
        "sha256": file_sha256(cached_file) if cached_file.is_file() else None,
    }


def _write_receipt(
    path: str | os.PathLike[str],
    *,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_hash: str,
    artifacts: dict[str, dict[str, object]],
    real_dataset_identity: dict[str, object],
    score: float,
    self_score: float,
) -> None:
    receipt_path = Path(path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    harness_path = Path(__file__).resolve()
    payload = {
        "schema_version": 2,
        "metric_id": _METRIC_ID,
        "status": "current-harness-run",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": _git_snapshot(),
        "harness": {
            "path": _relative_path(harness_path),
            "sha256": file_sha256(harness_path),
            "cache_version": _CACHE_VERSION,
            "source_files": _source_snapshot(),
            "inception_weights": _inception_weights_snapshot(),
            "preprocess": "bilinear resize 299x299; ImageNet mean/std",
        },
        "environment": _environment_snapshot(device),
        "checkpoint": {
            "path": _relative_path(args.ckpt),
            "bytes": Path(args.ckpt).stat().st_size,
            "sha256": checkpoint_hash,
        },
        "evaluation": {
            "n": args.n,
            "seed": args.seed,
            "channels": args.channels,
            "image_size": args.image_size,
            "steps": args.steps,
            "sampler": args.sampler,
            "order": args.order,
            "sample_batch_size": args.sample_batch_size,
            "feature_batch_size": args.batch_size,
            "num_classes": args.num_classes,
            "cfg_scale": args.cfg_scale,
            "real_subset_seed": 0,
            "real_dataset": real_dataset_identity,
            "no_cache_requested": args.no_cache,
        },
        "results": {
            "internal_fid_style_score": score,
            "real_vs_real_sanity_score": self_score,
        },
        "artifacts": {
            name: {
                **record,
                "path": _relative_path(str(record.get("path", "<unknown>"))),
            }
            for name, record in artifacts.items()
        },
        "argv": _sanitize_argv(list(sys.argv)),
        "reproducibility_note": (
            "The fixed seed and recorded environment support reruns, but CUDA "
            "results are not claimed bitwise-identical across hardware or "
            "library versions."
        ),
    }
    receipt_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"receipt: {_relative_path(receipt_path)}")
    print(f"receipt sha256: {file_sha256(receipt_path)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="FID of a trained diffusion model vs CIFAR-10"
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--sample-batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="conditional model: number of classes",
    )
    ap.add_argument(
        "--cfg-scale", type=float, default=0.0, help="classifier-free guidance scale"
    )
    ap.add_argument("--sampler", choices=["ddim", "dpm"], default="ddim")
    ap.add_argument(
        "--order",
        type=int,
        choices=(1, 2),
        default=2,
        help="DPM-Solver order (1 or 2)",
    )
    ap.add_argument(
        "--no-cache", action="store_true", help="recompute images+features from scratch"
    )
    ap.add_argument(
        "--receipt",
        help="write a machine-readable JSON receipt after a successful evaluation",
    )
    args = ap.parse_args()

    if args.batch_size < 1 or args.sample_batch_size < 1:
        ap.error("batch sizes must be at least 1")
    try:
        validate_sampling_options(
            n=args.n,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            sampler=args.sampler,
            order=args.order,
            timesteps=GaussianDiffusion().timesteps,
        )
    except ValueError as error:
        ap.error(str(error))
    if args.n < 2:
        ap.error("--n must be at least 2 so sample covariance is defined")

    architecture = checkpoint_architecture(args.ckpt)
    if args.channels != architecture["channels"]:
        ap.error(
            f"--channels={args.channels} does not match checkpoint "
            f"({architecture['channels']})"
        )
    checkpoint_num_classes = architecture["num_classes"]
    if args.num_classes is not None and args.num_classes != checkpoint_num_classes:
        ap.error(
            f"--num-classes={args.num_classes} does not match checkpoint "
            f"({checkpoint_num_classes})"
        )
    args.num_classes = checkpoint_num_classes
    if args.num_classes is None and args.cfg_scale != 0.0:
        ap.error("--cfg-scale must be 0 for an unconditional checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    cache_dir = os.path.join(ckpt_dir, "_fid_cache")
    os.makedirs(cache_dir, exist_ok=True)

    artifacts: dict[str, dict[str, object]] = {}

    def _load_or(label, name, cache_identity, fn):
        p = os.path.join(cache_dir, name)
        cache_hit = os.path.exists(p) and not args.no_cache
        if cache_hit:
            print(f"  cache hit: {name}")
            array = np.load(p, allow_pickle=False)
        else:
            array = fn()
            np.save(p, array)
        artifacts[label] = {
            **_artifact_record(p, cache_hit=cache_hit),
            "cache_identity_sha256": _identity_sha256(cache_identity),
        }
        return array

    print("loading canonical real CIFAR-10 test subset ...")
    real_images, real_dataset_identity = load_real_cifar(
        args.n,
        seed=0,
        return_identity=True,
    )

    print("loading inception (first run downloads weights) ...")
    inception = get_inception(device)

    source_identity = _source_snapshot()
    environment_identity = _environment_snapshot(device)
    inception_identity = _inception_weights_snapshot()

    print(f"generating {args.n} samples ({args.sampler}, {args.steps} steps) ...")
    checkpoint_hash = file_sha256(args.ckpt)
    key = evaluation_cache_key(
        n=args.n,
        seed=args.seed,
        checkpoint_hash=checkpoint_hash,
        channels=args.channels,
        image_size=args.image_size,
        num_classes=args.num_classes,
        cfg_scale=args.cfg_scale,
        sampler=args.sampler,
        steps=args.steps,
        order=args.order,
        sample_batch_size=args.sample_batch_size,
        feature_batch_size=args.batch_size,
        dataset_identity=real_dataset_identity,
        source_identity=source_identity,
        environment_identity=environment_identity,
    )
    generated_identity = {
        "kind": "generated-images",
        "evaluation_cache_key": key,
    }
    fake_np = _load_or(
        "generated_images",
        f"fake_imgs_{key}.npy",
        generated_identity,
        lambda: (
            generate_samples(
                args.ckpt,
                args.n,
                args.channels,
                args.image_size,
                device,
                args.steps,
                num_classes=args.num_classes,
                cfg_scale=args.cfg_scale,
                sampler=args.sampler,
                order=args.order,
                sample_batch_size=args.sample_batch_size,
                seed=args.seed,
            )
            .cpu()
            .numpy()
        ),
    )

    real_identity = {
        "kind": "real-images",
        "dataset": real_dataset_identity,
        "source": source_identity.get("data_utils.py"),
        "environment": {
            "numpy": environment_identity["numpy"],
            "torch": environment_identity["torch"],
            "torchvision": environment_identity["torchvision"],
        },
    }
    real_key = _derived_cache_key("real", real_identity)
    real_np = _load_or(
        "real_images",
        f"real_imgs_{real_key}.npy",
        real_identity,
        lambda: real_images.numpy(),
    )

    print("extracting features ...")
    feature_identity = {
        "metric_id": _METRIC_ID,
        "preprocess": "bilinear resize 299x299; ImageNet mean/std",
        "inception_weights": inception_identity,
        "source": source_identity.get("fid.py"),
        "environment": environment_identity,
        "feature_batch_size": args.batch_size,
    }
    fake_feature_identity = {
        **feature_identity,
        "image_artifact_sha256": artifacts["generated_images"]["sha256"],
    }
    fake_feature_key = _derived_cache_key("features", fake_feature_identity)
    act_fake = _load_or(
        "generated_activations",
        f"act_fake_{fake_feature_key}.npy",
        fake_feature_identity,
        lambda: compute_activations(
            inception, torch.from_numpy(fake_np), device, args.batch_size
        ),
    )
    real_feature_identity = {
        **feature_identity,
        "image_artifact_sha256": artifacts["real_images"]["sha256"],
        "real_dataset": real_dataset_identity,
    }
    real_feature_key = _derived_cache_key("features", real_feature_identity)
    act_real = _load_or(
        "real_activations",
        f"act_real_{real_feature_key}.npy",
        real_feature_identity,
        lambda: compute_activations(
            inception, torch.from_numpy(real_np), device, args.batch_size
        ),
    )

    fid = fid_from_activations(act_real, act_fake)

    # Sanity: FID of the real set against itself must be ~0, else the
    # Frechet implementation is wrong (this catches the non-symmetric sqrtm bug).
    mu_r, sig_r = act_real.mean(axis=0), np.cov(act_real, rowvar=False)
    fid_self = frechet_distance(mu_r, sig_r, mu_r, sig_r)

    print(f"\nInternal FID-style score (n={args.n}): {fid:.2f}")
    print(f"sanity FID(real, real) = {fid_self:.5f}  (must be ~0)")
    if args.receipt:
        _write_receipt(
            args.receipt,
            args=args,
            device=device,
            checkpoint_hash=checkpoint_hash,
            artifacts=artifacts,
            real_dataset_identity=real_dataset_identity,
            score=fid,
            self_score=fid_self,
        )


if __name__ == "__main__":
    main()
