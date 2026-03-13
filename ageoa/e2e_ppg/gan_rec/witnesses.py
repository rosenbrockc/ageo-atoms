from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal, ANYTHING

def witness_generatereconstructedppg(
    ppg_clean: AbstractSignal,
    noise: AbstractSignal,
    sampling_rate: AbstractScalar,
    generator: AbstractSignal,
    device: AbstractScalar,
) -> AbstractSignal:
    """Ghost witness for GenerateReconstructedPPG.

    Args:
        ppg_clean: Clean PPG signal metadata.
        noise: Latent noise metadata.
        sampling_rate: Sampling frequency metadata.
        generator: Generator model metadata.
        device: Execution device metadata.

    Returns:
        Reconstructed PPG signal metadata with shape inherited from input.
    """
    return AbstractSignal(
        shape=ppg_clean.shape,
        dtype=ppg_clean.dtype,
        sampling_rate=ppg_clean.sampling_rate,
        domain=ppg_clean.domain,
        units=ppg_clean.units,
    )
