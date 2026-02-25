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

def witness_detect_signal_onsets_elgendi2013(signal: AbstractSignal, sampling_rate: AbstractSignal, peakwindow: AbstractSignal, beatwindow: AbstractSignal, beatoffset: AbstractSignal, mindelay: AbstractSignal) -> AbstractSignal:
    """Ghost witness for detect_signal_onsets_elgendi2013."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result
