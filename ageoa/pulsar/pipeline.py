"""Pulsar Folding atoms implementing DM-trial dedispersion and signal folding."""

from __future__ import annotations

import math
from typing import Any

import icontract
import numpy as np

from ageoa.ghost.registry import register_atom
from ageoa.pulsar.witnesses import (
    witness_de_disperse,
    witness_delay_from_dm,
    witness_fold_signal,
    witness_snr,
)


@register_atom(witness_delay_from_dm)
@icontract.require(lambda DM: DM >= 0, "DM must be non-negative")
@icontract.require(lambda freq_emitted: freq_emitted >= 0, "Frequency must be non-negative")
@icontract.ensure(lambda result: result >= 0, "Delay must be non-negative")
def delay_from_DM(DM: float, freq_emitted: float) -> float:
    """Calculate time delay for a given DM and emission frequency."""
    if freq_emitted > 0.0:
        return DM / (0.000241 * freq_emitted * freq_emitted)
    return 0.0


@register_atom(witness_de_disperse)
@icontract.require(lambda data: data.ndim == 2, "Input data must be 2D (Time, Frequency)")
@icontract.require(lambda tsamp: tsamp > 0, "tsamp must be positive")
@icontract.require(lambda width: width > 0, "Channel width must be positive")
@icontract.ensure(lambda result, data: result.shape == data.shape, "Output must preserve input shape")
def de_disperse(
    data: np.ndarray[Any, Any],
    DM: float,
    fchan: float,
    width: float,
    tsamp: float,
) -> np.ndarray[Any, Any]:
    """Apply dedispersion to 2D spectrogram data."""
    clean = np.array(data, copy=True)
    n_time, n_chans = clean.shape

    for chan in range(n_chans):
        freq_emitted = chan * width + fchan
        time_delay = int(delay_from_DM(DM, freq_emitted) / tsamp)

        if 0 < time_delay < n_time:
            shifted = clean[: n_time - time_delay, chan]
            clean[time_delay:n_time, chan] = shifted
            clean[:time_delay, chan] = 0.0
        elif time_delay >= n_time:
            clean[:, chan] = 0.0

    return clean


@register_atom(witness_fold_signal)
@icontract.require(lambda data: data.ndim == 2, "Input data must be 2D")
@icontract.require(lambda period: period > 0, "Folding period must be positive")
@icontract.ensure(lambda result, period: result.shape == (period,), "Profile length must match period")
def fold_signal(data: np.ndarray[Any, Any], period: int) -> np.ndarray[Any, Any]:
    """Fold the 2D data into a 1D pulse profile given a period."""
    n_time = data.shape[0]
    n_chans = data.shape[1]
    multiples = n_time // period

    if multiples < 1:
        return np.zeros(period, dtype=np.float64)

    folded = np.zeros((period, n_chans), dtype=np.float64)
    for i in range(multiples):
        folded += data[i * period : (i + 1) * period, :]

    folded /= float(multiples)
    return folded.mean(axis=1)


@register_atom(witness_snr)
@icontract.require(lambda arr: arr.ndim == 1, "Input must be 1D")
@icontract.require(lambda arr: len(arr) > 0, "Input array must not be empty")
@icontract.ensure(lambda result: result >= 0, "SNR must be non-negative")
def SNR(arr: np.ndarray[Any, Any]) -> float:
    """Calculate log SNR of a 1D pulse profile."""
    if np.all(arr == 0):
        return 0.0

    peak = float(arr[int(np.argmax(arr))])
    avg_noise = float(abs(np.mean(arr)))
    if avg_noise <= 0:
        return 0.0

    ratio = peak / avg_noise
    if ratio <= 0:
        return 0.0

    return float(math.log(ratio))
