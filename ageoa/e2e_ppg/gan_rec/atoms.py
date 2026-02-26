"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

from .witnesses import witness_generatereconstructedppg

@register_atom(witness_generatereconstructedppg)
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.ensure(lambda result: result is not None, "GenerateReconstructedPPG output must not be None")
def generatereconstructedppg(ppg_clean: np.ndarray | torch.Tensor, noise: np.ndarray | torch.Tensor, sampling_rate: int | float, generator: torch.nn.Module, device: str | torch.device) -> np.ndarray | torch.Tensor:  # type: ignore[type-arg]
    """Applies the GAN generator to clean PPG context and latent noise to produce reconstructed or synthetic PPG output.

    Args:
        ppg_clean: Clean PPG signal input; shape must be compatible with generator input.
        noise: Latent noise vector/tensor aligned with generator latent dimensions.
        sampling_rate: Sampling frequency in Hz; must be > 0.
        generator: Inference-ready generator that maps inputs to reconstructed signal.
        device: Execution device compatible with model and tensors.

    Returns:
        Generated/reconstructed PPG signal; shape determined by generator architecture.
    """
    raise NotImplementedError("Wire to original implementation")