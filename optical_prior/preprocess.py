"""Preprocessing and fixed-basis utilities for structured optical prior experiments."""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, fftshift


Array = np.ndarray


def hadamard(n: int) -> Array:
    """Normalized real Hadamard matrix. Requires n to be a power of two."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("Hadamard basis requires N to be a power of two.")
    H = np.array([[1.0]], dtype=np.complex128)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def identity_basis(n: int) -> Array:
    return np.eye(n, dtype=np.complex128)


def random_unitary(n: int, seed: int = 0) -> Array:
    """Haar-like random unitary from QR factorization."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    ph = d / np.maximum(np.abs(d), 1e-12)
    return Q @ np.diag(np.conj(ph))


def normalize_power(x: Array, P0: float = 10.0, eps: float = 1e-9) -> Array:
    """Normalize each sample to total optical power P0."""
    x = np.asarray(x)
    flat = x.reshape(x.shape[0], -1)
    norm = np.sqrt(np.sum(np.abs(flat) ** 2, axis=1, keepdims=True))
    flat = np.sqrt(P0) * flat / (norm + eps)
    return flat.astype(np.complex128)


def _fft_center_patches(images: Array, N: int) -> Array:
    """Return the central sqrt(N) x sqrt(N) block of the centered 2D FFT."""
    side = int(np.sqrt(N))
    if side * side != N:
        raise ValueError("N must be a perfect square for FFT-center preprocessing.")
    if side > images.shape[-1]:
        raise ValueError("sqrt(N) cannot exceed the image side length.")
    start = (images.shape[-1] - side) // 2
    patches = []
    for img in images:
        f = fftshift(fft2(img))
        c = f[start:start + side, start:start + side]
        patches.append(c.flatten())
    return np.asarray(patches, dtype=np.complex128)


def preprocess_images_fft_center(images: Array, N: int, P0: float = 10.0, eps: float = 1e-9) -> Array:
    """Legacy FFT-center preprocessing used for the reported ~0.88 anchor results.

    Each sample is normalized by the maximum magnitude in the selected FFT block,
    then scaled by sqrt(P0). This matches the original implementation in
    `Ug block 权值共享.py` and the earlier global-runner experiments.
    """
    patches = _fft_center_patches(images, N)
    denom = np.max(np.abs(patches), axis=1, keepdims=True) + eps
    return (np.sqrt(P0) * patches / denom).astype(np.complex128)


def preprocess_images_fft_center_power(images: Array, N: int, P0: float = 10.0) -> Array:
    """Power-normalized FFT-center preprocessing.

    Kept as an optional control. This was used in one intermediate final package
    and gives lower absolute accuracy because the input distribution differs from
    the original reported setup.
    """
    return normalize_power(_fft_center_patches(images, N), P0=P0)


def preprocess_images_raw_first16(images: Array, N: int, P0: float = 10.0) -> Array:
    """Legacy raw baseline: first N flattened pixels. Kept only for reproducing old runs."""
    x = images.reshape(len(images), -1).astype(np.float64)[:, :N]
    return normalize_power(x, P0=P0)


def preprocess_images_raw_center(images: Array, N: int, P0: float = 10.0) -> Array:
    """Use a centered sqrt(N) x sqrt(N) crop as a stable raw-coordinate baseline."""
    side = int(np.sqrt(N))
    if side * side != N:
        raise ValueError("N must be a perfect square for raw_center preprocessing.")
    h, w = images.shape[-2], images.shape[-1]
    top = (h - side) // 2
    left = (w - side) // 2
    x = images[:, top:top + side, left:left + side].reshape(len(images), -1).astype(np.float64)
    return normalize_power(x, P0=P0)


def preprocess_images_raw_pool(images: Array, N: int, P0: float = 10.0) -> Array:
    """Average-pool each image to sqrt(N) x sqrt(N) as a stable raw baseline."""
    side = int(np.sqrt(N))
    if side * side != N:
        raise ValueError("N must be a perfect square for raw_pool preprocessing.")
    h, w = images.shape[-2], images.shape[-1]
    if h % side != 0 or w % side != 0:
        raise ValueError("Image size must be divisible by sqrt(N) for raw_pool.")
    bh, bw = h // side, w // side
    x = images.reshape(len(images), side, bh, side, bw).mean(axis=(2, 4))
    x = x.reshape(len(images), -1).astype(np.float64)
    return normalize_power(x, P0=P0)


def preprocess_images_pca(train_images: Array, test_images: Array, N: int, P0: float = 10.0, eps: float = 1e-9):
    """PCA projection learned on the training set, then power normalized."""
    xtr = train_images.reshape(len(train_images), -1).astype(np.float64)
    xte = test_images.reshape(len(test_images), -1).astype(np.float64)
    mu = xtr.mean(axis=0, keepdims=True)
    xtr0 = xtr - mu
    xte0 = xte - mu
    _, _, Vt = np.linalg.svd(xtr0, full_matrices=False)
    Vn = Vt[:N].T
    ztr = xtr0 @ Vn
    zte = xte0 @ Vn
    return normalize_power(ztr, P0=P0), normalize_power(zte, P0=P0)
