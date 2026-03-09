"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations
from typing import Any

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

# Witness functions should be imported from the generated witnesses module
witness_gan_reconstruction: Any = None
@register_atom(witness_gan_reconstruction)  # type: ignore[untyped-decorator]
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
def gan_reconstruction(ppg_clean: Any, noise: Any, sampling_rate: int | float, generator: Any, device: str | torch.device) -> Any:
    """Reconstructs a clean ECG-like signal from a noisy PPG input using a pre-trained GAN generator. Accepts a clean PPG signal and additive noise, feeds them through the generator network on the specified device, and returns the reconstructed output signal.

    Args:
        ppg_clean: shape compatible with generator input; values typically normalized
        noise: same shape as ppg_clean; treated as frozen/static stochastic input
        sampling_rate: positive; determines temporal resolution of the waveform
        generator: stateless inference only; weights frozen during call
        device: generator and tensors must reside on same device

    Returns:
        same temporal length as input; values in generator output range
    """
    raise NotImplementedError("Wire to original implementation")