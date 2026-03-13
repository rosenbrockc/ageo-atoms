"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations

import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore

try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
except ImportError:
    pass

def witness_zhao2018hrvanalysis(ecg_cleaned: AbstractSignal, rpeaks: AbstractSignal, sampling_rate: AbstractSignal, window: AbstractSignal, mode: AbstractSignal) -> AbstractSignal:
    """Ghost witness for Zhao2018HRVAnalysis."""
    result = AbstractSignal(
        shape=ecg_cleaned.shape,
        dtype="float64",
        sampling_rate=getattr(ecg_cleaned, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result

def witness_averageqrstemplate(ecg_cleaned: AbstractSignal, rpeaks: AbstractSignal, sampling_rate: AbstractSignal) -> AbstractSignal:
    """Ghost witness for AverageQRSTemplate."""
    result = AbstractSignal(
        shape=ecg_cleaned.shape,
        dtype="float64",
        sampling_rate=getattr(ecg_cleaned, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result
