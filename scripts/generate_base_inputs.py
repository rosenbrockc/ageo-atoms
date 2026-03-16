#!/usr/bin/env python3
"""Generate representative base inputs for uncertainty measurement.

Provides a registry of atom_name -> Callable[[], np.ndarray] for atoms
that need hand-crafted inputs.  Falls back to standard_normal(256) for
unknown atoms.
"""
from __future__ import annotations

import numpy as np


def _time_signal(n: int = 256, fs: float = 1000.0) -> np.ndarray:
    """Synthetic time-domain signal: sum of sinusoids + noise."""
    t = np.arange(n) / fs
    return (
        np.sin(2 * np.pi * 50 * t)
        + 0.5 * np.sin(2 * np.pi * 120 * t)
        + 0.1 * np.random.default_rng(42).standard_normal(n)
    )


def _freq_signal(n: int = 256) -> np.ndarray:
    """Frequency-domain signal (complex)."""
    return np.fft.fft(_time_signal(n))


def _positive_signal(n: int = 256) -> np.ndarray:
    """Positive-valued signal (e.g. for log-domain ops)."""
    return np.abs(_time_signal(n)) + 0.01


def _unit_interval(n: int = 256) -> np.ndarray:
    """Values in [0, 1]."""
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1, size=n)


def _matrix_2d(n: int = 32) -> np.ndarray:
    """Square matrix."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, n))


def _symmetric_matrix(n: int = 32) -> np.ndarray:
    """Symmetric positive-definite matrix."""
    A = _matrix_2d(n)
    return A @ A.T + np.eye(n)


# Registry: atom_name -> callable returning a representative input
BASE_INPUT_REGISTRY: dict[str, object] = {
    # FFT family
    "fft": _time_signal,
    "ifft": _freq_signal,
    "rfft": _time_signal,
    "irfft": lambda: np.fft.rfft(_time_signal()),
    "stft": lambda: _time_signal(1024),
    "istft": lambda: _freq_signal(1024),
    # Filter application
    "lfilter": _time_signal,
    "sosfilt": _time_signal,
    "convolve": _time_signal,
    "correlate": _time_signal,
    "resample": _time_signal,
    "hilbert": _time_signal,
    "welch": lambda: _time_signal(1024),
    # Linear algebra
    "svd": _matrix_2d,
    "eig": _matrix_2d,
    "eigh": _symmetric_matrix,
    "cholesky": _symmetric_matrix,
    "inv": _symmetric_matrix,
    "det": _matrix_2d,
    "norm": _time_signal,
    "lstsq": _matrix_2d,
    # Element-wise
    "log": _positive_signal,
    "exp": lambda: _unit_interval(256) * 5,  # avoid overflow
    "sqrt": _positive_signal,
    "abs": _time_signal,
}


def get_base_input(atom_name: str, size: int = 256) -> np.ndarray:
    """Return a representative input for the given atom.

    Uses the registry if available, otherwise falls back to standard_normal.
    """
    generator = BASE_INPUT_REGISTRY.get(atom_name)
    if generator is not None:
        return np.asarray(generator())
    # Fallback
    rng = np.random.default_rng(hash(atom_name) % (2**31))
    return rng.standard_normal(size)
