"""Unit tests for the FID math (no GPU or Inception download needed)."""

import numpy as np

from fid import _sqrtm, frechet_distance


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
