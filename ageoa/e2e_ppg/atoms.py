"""Deterministic atom wrappers for Photoplethysmography (PPG) processing."""

from __future__ import annotations

import icontract
import numpy as np

from ageoa.ghost.registry import register_atom

from .witnesses import witness_kazemi_peak_detection, witness_ppg_reconstruction, witness_ppg_sqa


def _flatten_index_groups(groups: list[list[int]]) -> np.ndarray:
    if not groups:
        return np.array([], dtype=np.intp)
    flattened = [index for group in groups for index in group]
    return np.asarray(flattened, dtype=np.intp)


def _group_sorted_indices(indices: np.ndarray) -> list[list[int]]:
    if indices.size == 0:
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    for index in indices[1:]:
        if int(index) == groups[-1][-1] + 1:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    return groups


def _is_finite_1d_array(value: object) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 1 and value.shape[0] > 0 and np.isfinite(value).all()


@register_atom(witness_kazemi_peak_detection)
@icontract.require(lambda signal: signal is not None, "signal must not be None")
@icontract.require(lambda signal: _is_finite_1d_array(signal), "signal must be a non-empty finite 1D numpy array")
@icontract.require(lambda sampling_freq: sampling_freq > 0, "sampling_freq must be positive")
@icontract.require(lambda seconds: seconds > 0, "seconds must be positive")
@icontract.require(lambda overlap: overlap >= 0, "overlap must be non-negative")
@icontract.require(lambda seconds, overlap: overlap < seconds, "overlap must be smaller than seconds")
@icontract.require(lambda minlen: minlen > 0, "minlen must be positive")
@icontract.ensure(lambda result: result is not None, "result must not be None")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be a numpy array")
@icontract.ensure(lambda result: result.ndim == 1, "result must be a 1D numpy array")
def kazemi_peak_detection(
    signal: np.ndarray,
    sampling_freq: int,
    seconds: int,
    overlap: int,
    minlen: int,
) -> np.ndarray:
    """Detect photoplethysmography peaks with a deterministic local-maxima proxy.

    Args:
        signal: Finite one-dimensional PPG trace.
        sampling_freq: Sampling frequency in hertz.
        seconds: Window length parameter kept for API compatibility.
        overlap: Window overlap parameter kept for API compatibility.
        minlen: Minimum signal duration, in seconds, required before detection.

    Returns:
        Monotonic integer peak indices into ``signal``.
    """
    del seconds, overlap

    from scipy.signal import find_peaks

    if signal.shape[0] < int(minlen * sampling_freq):
        return np.array([], dtype=np.intp)

    min_distance = max(1, int(round(sampling_freq * 0.35)))
    prominence = max(1e-6, float(np.std(signal)) * 0.1)
    peaks, _ = find_peaks(signal, distance=min_distance, prominence=prominence)
    return peaks.astype(np.intp)


@register_atom(witness_ppg_reconstruction)
@icontract.require(lambda sig: sig is not None, "sig must not be None")
@icontract.require(lambda sig: _is_finite_1d_array(sig), "sig must be a non-empty finite 1D numpy array")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "sampling_rate must be positive")
@icontract.ensure(lambda result: result is not None, "result must not be None")
@icontract.ensure(lambda result: isinstance(result, tuple), "result must be a tuple")
def ppg_reconstruction(
    sig: np.ndarray,
    clean_indices: list[list[int]],
    noisy_indices: list[list[int]],
    sampling_rate: int,
    filter_signal: bool = True,
) -> tuple[np.ndarray, list[list[int]], list[list[int]]]:
    """Reconstruct noisy PPG spans by interpolating over marked noisy regions.

    Args:
        sig: Finite one-dimensional input signal.
        clean_indices: Groups of clean sample indices already accepted as reliable.
        noisy_indices: Groups of noisy sample indices to be reconstructed.
        sampling_rate: Sampling frequency in hertz.
        filter_signal: Compatibility flag for the original pipeline shape.

    Returns:
        Tuple of reconstructed signal, updated clean index groups, and remaining noisy groups.
    """
    del sampling_rate, filter_signal

    reconstructed = sig.astype(np.float64, copy=True)
    clean_flat = _flatten_index_groups(clean_indices)
    noisy_flat = _flatten_index_groups(noisy_indices)
    if noisy_flat.size == 0:
        return reconstructed, [list(group) for group in clean_indices], []

    all_indices = np.arange(reconstructed.shape[0], dtype=np.intp)
    valid_mask = np.ones(reconstructed.shape[0], dtype=bool)
    valid_mask[noisy_flat] = False

    if valid_mask.sum() >= 2:
        reconstructed[noisy_flat] = np.interp(
            noisy_flat.astype(np.float64),
            all_indices[valid_mask].astype(np.float64),
            reconstructed[valid_mask],
        )

    clean_union = np.union1d(clean_flat, noisy_flat)
    return reconstructed, _group_sorted_indices(clean_union.astype(np.intp)), []


@register_atom(witness_ppg_sqa)
@icontract.require(lambda sig: sig is not None, "sig must not be None")
@icontract.require(lambda sig: _is_finite_1d_array(sig), "sig must be a non-empty finite 1D numpy array")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "sampling_rate must be positive")
@icontract.ensure(lambda result: result is not None, "result must not be None")
@icontract.ensure(lambda result: isinstance(result, tuple), "result must be a tuple")
def ppg_sqa(
    sig: np.ndarray,
    sampling_rate: int,
    filter_signal: bool = True,
) -> tuple[list[list[int]], list[list[int]]]:
    """Classify a PPG trace into clean and noisy sample groups.

    Args:
        sig: Finite one-dimensional input signal.
        sampling_rate: Sampling frequency in hertz.
        filter_signal: Compatibility flag for the original pipeline shape.

    Returns:
        Tuple of clean index groups and noisy index groups.
    """
    del filter_signal

    window = max(int(30 * sampling_rate), 1)
    shift = max(int(2 * sampling_rate), 1)
    if sig.shape[0] < window:
        return [], [list(range(sig.shape[0]))]

    clean_indices: list[int] = []
    for start in range(0, sig.shape[0] - window + 1, shift):
        segment = sig[start : start + window]
        centered = segment - np.mean(segment)
        signal_energy = float(np.mean(centered ** 2))
        noise_proxy = float(np.mean(np.diff(segment) ** 2)) if segment.shape[0] > 1 else 0.0
        quality = signal_energy / (noise_proxy + 1e-12)
        if np.isfinite(quality) and quality >= 1.0:
            clean_indices.extend(range(start, start + window))

    clean_unique = np.asarray(sorted(set(clean_indices)), dtype=np.intp)
    if clean_unique.size == 0:
        return [], [list(range(sig.shape[0]))]

    noisy_unique = np.asarray(
        [index for index in range(sig.shape[0]) if index not in set(clean_unique.tolist())],
        dtype=np.intp,
    )
    return _group_sorted_indices(clean_unique), _group_sorted_indices(noisy_unique)
