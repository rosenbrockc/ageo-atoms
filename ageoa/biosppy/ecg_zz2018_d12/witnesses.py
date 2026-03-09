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

def witness_computebeatagreementsqi(detector_1: AbstractArray, detector_2: AbstractArray, fs: AbstractArray, mode: AbstractArray, search_window: AbstractArray) -> AbstractArray:
    """Ghost witness for ComputeBeatAgreementSQI."""
    result = AbstractArray(
        shape=detector_1.shape,
        dtype="float64",
    )
    return result

def witness_computefrequencysqi(ecg_signal: AbstractSignal, fs: AbstractSignal, nseg: AbstractSignal, num_spectrum: AbstractSignal, dem_spectrum: AbstractSignal, mode: AbstractSignal) -> AbstractSignal:
    """Ghost witness for ComputeFrequencySQI."""
    result = AbstractSignal(
        shape=ecg_signal.shape,
        dtype="float64",
        sampling_rate=getattr(ecg_signal, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result

def witness_computekurtosissqi(signal: AbstractArray, fisher: AbstractArray) -> AbstractArray:
    """Ghost witness for ComputeKurtosisSQI."""
    result = AbstractArray(
        shape=signal.shape,
        dtype="float64",
    )
    return result

def witness_assemblezz2018sqi(signal: AbstractArray, detector_1: AbstractArray, detector_2: AbstractArray, fs: AbstractArray, search_window: AbstractArray, nseg: AbstractArray, mode: AbstractArray, b_sqi: AbstractArray, f_sqi: AbstractArray, k_sqi: AbstractArray) -> AbstractArray:
    """Ghost witness for AssembleZZ2018SQI."""
    result = AbstractArray(
        shape=signal.shape,
        dtype="float64",
    )
    return result
