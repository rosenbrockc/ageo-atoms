from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

from .gan_reconstruction_witnesses import witness_generatereconstructedppg
from ppg_reconstruction import gan_rec

@register_atom(witness_generatereconstructedppg)
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.ensure(lambda result: result is not None, "GenerateReconstructedPPG output must not be None")
def generatereconstructedppg(ppg_clean: np.ndarray | torch.Tensor, noise: np.ndarray | torch.Tensor, sampling_rate: int | float, generator: torch.nn.Module, device: str | torch.device) -> np.ndarray | torch.Tensor:  # type: ignore[type-arg]
    """Applies a Generative Adversarial Network (GAN) generator to a clean photoplethysmogram (PPG) pulse-wave signal and latent noise to produce a reconstructed PPG output.

Args:
    ppg_clean: Clean PPG signal input; shape must be compatible with generator input.
    noise: Latent noise vector/tensor aligned with generator latent dimensions.
    sampling_rate: Sampling frequency in Hz; must be > 0.
    generator: Inference-ready generator that maps inputs to reconstructed signal.
    device: Execution device compatible with model and tensors.

Returns:
    Generated/reconstructed PPG signal; shape determined by generator architecture."""
    return gan_rec(ppg_clean=ppg_clean, noise=noise, sampling_rate=sampling_rate, generator=generator, device=device)

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .gan_reconstruction_witnesses import witness_gan_reconstruction
from ppg_reconstruction import gan_rec


@register_atom(witness_gan_reconstruction)
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "sampling_rate must be positive")
@icontract.ensure(lambda result: isinstance(result, list), "gan_reconstruction must return a list")
def gan_reconstruction(ppg_clean: np.ndarray, noise: list[int], sampling_rate: int, generator: torch.nn.Module, device: str | torch.device) -> list[float]:
    """Reconstructs a clean electrocardiogram (ECG)-like signal from a noisy photoplethysmography (PPG) input using a pre-trained GAN generator. Accepts a clean PPG signal and additive noise, feeds them through the generator network on the specified device, and returns the reconstructed output signal.

    Args:
        ppg_clean: shape compatible with generator input; values typically normalized
        noise: list of noise indices to reconstruct
        sampling_rate: positive; determines temporal resolution of the waveform
        generator: stateless inference only; weights frozen during call
        device: generator and tensors must reside on same device

    Returns:
        Reconstructed noise signal as a list of floats; same temporal length as noise input."""
    return gan_rec(ppg_clean=ppg_clean, noise=noise, sampling_rate=sampling_rate, generator=generator, device=device)
