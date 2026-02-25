"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

from typing import Any, Callable

# Witness functions should be imported from the generated witnesses module
witness_gan_patch_reconstruction: Callable[..., bool] = lambda *args, **kwargs: True
witness_windowed_signal_reconstruction: Callable[..., bool] = lambda *args, **kwargs: True
@register_atom(witness_gan_patch_reconstruction)
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "GAN Patch Reconstruction output must not be None")
def gan_patch_reconstruction(ppg_clean, noise, sampling_rate, generator, device):
    """Generates a reconstructed signal patch from clean PPG context and injected noise using a provided generator on a specified device.

    Args:
        ppg_clean: Shape compatible with generator input
        noise: Shape compatible with generator input
        sampling_rate: Positive
        generator: Stateless from this graph perspective
        device: Valid runtime device

    Returns:
        Aligned to target patch length
    """
    raise NotImplementedError("Wire to original implementation")
def windowed_signal_reconstruction(sig: Any, clean_indices: Any, noisy_indices: Any, sampling_rate: Any, filter_signal: Any) -> Any:
@register_atom(witness_windowed_signal_reconstruction)
@icontract.ensure(lambda result, **kwargs: result is not None, "Windowed Signal Reconstruction output must not be None")
def windowed_signal_reconstruction(sig, clean_indices, noisy_indices, sampling_rate, filter_signal):
    """Constructs a full reconstructed signal from clean/noisy index partitions, with optional filtering controlled by input flag.

    Args:
        sig: 1D or channel-first signal supported by implementation
        clean_indices: Valid indices into sig
        noisy_indices: Valid indices into sig; may be disjoint from clean_indices
        sampling_rate: Positive
        filter_signal: If truthy/callable, applies filtering path

    Returns:
        Same length/index domain as input sig
    """
    raise NotImplementedError("Wire to original implementation")