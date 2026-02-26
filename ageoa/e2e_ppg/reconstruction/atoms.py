"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom

from .witnesses import witness_gan_patch_reconstruction, witness_windowed_signal_reconstruction


@register_atom(witness_gan_patch_reconstruction)
@icontract.require(lambda ppg_clean: isinstance(ppg_clean, np.ndarray), "ppg_clean must be np.ndarray")
@icontract.require(lambda noise: isinstance(noise, np.ndarray), "noise must be np.ndarray")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
def gan_patch_reconstruction(ppg_clean: np.ndarray, noise: np.ndarray, sampling_rate: float, generator: torch.nn.Module, device: str) -> np.ndarray:
    """Generate a reconstructed signal patch from clean PPG context and injected noise using a provided generator on a specified device.

    Args:
        ppg_clean: Clean PPG signal; shape compatible with generator input.
        noise: Latent noise vector; shape compatible with generator input.
        sampling_rate: Sampling frequency in Hz; must be > 0.
        generator: Stateless generator model from this graph perspective.
        device: Valid runtime device string.

    Returns:
        Reconstructed signal patch aligned to target patch length.
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_windowed_signal_reconstruction)
@icontract.require(lambda sig: isinstance(sig, np.ndarray), "sig must be np.ndarray")
@icontract.require(lambda clean_indices: isinstance(clean_indices, np.ndarray), "clean_indices must be np.ndarray")
@icontract.require(lambda noisy_indices: isinstance(noisy_indices, np.ndarray), "noisy_indices must be np.ndarray")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
def windowed_signal_reconstruction(sig: np.ndarray, clean_indices: np.ndarray, noisy_indices: np.ndarray, sampling_rate: float, filter_signal: bool) -> np.ndarray:
    """Construct a full reconstructed signal from clean/noisy index partitions, with optional filtering controlled by input flag.

    Args:
        sig: Input signal; 1-D or channel-first supported by implementation.
        clean_indices: Valid indices into sig marking clean segments.
        noisy_indices: Valid indices into sig marking noisy segments; may be disjoint from clean_indices.
        sampling_rate: Sampling frequency in Hz; must be > 0.
        filter_signal: If True, applies filtering path to output.

    Returns:
        Reconstructed signal with same length/index domain as input sig.
    """
    raise NotImplementedError("Wire to original implementation")
