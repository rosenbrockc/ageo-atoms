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
    ppg_clean: Clean photoplethysmography (PPG) signal metadata.
    noise: Latent noise metadata.
    sampling_rate: Sampling frequency metadata.
    generator: Generator model metadata.
    device: Execution device metadata.

Returns:
    Reconstructed PPG signal metadata with shape inherited from input."""
    return AbstractSignal(
        shape=ppg_clean.shape,
        dtype=ppg_clean.dtype,
        sampling_rate=ppg_clean.sampling_rate,
        domain=ppg_clean.domain,
        units=ppg_clean.units,
    )

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal, ANYTHING

def witness_gan_reconstruction(
    ppg_clean: AbstractArray,
    noise: AbstractArray,
    sampling_rate: AbstractScalar,
    generator: AbstractArray,
    device: AbstractScalar,
) -> AbstractArray:
    """Ghost witness for gan_reconstruction.
    
    Original witness incorrectly returned `shape=clean_segment.shape` (no
    batch axis), which made the abstract graph see a shape-preserving
    identity at this node and caused the simulator to detect a cycle among
    {'normalize_and_batch_clean_segment',
     'stitch_clean_and_reconstructed_waveforms',
     'generate_reconstructed_segment',
     'detect_peaks_in_reconstructed_and_clean',
     'accumulate_reconstructed_noise_and_advance_window'}.
    Prepending the batch dimension breaks that cycle.
    """
    result = AbstractArray(
        shape=(1,) + ppg_clean.shape,
        dtype="float64",
    )
    return result
