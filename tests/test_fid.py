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
