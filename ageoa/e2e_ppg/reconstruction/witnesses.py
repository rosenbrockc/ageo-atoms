"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations



try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
except ImportError:
    pass

def witness_gan_patch_reconstruction(ppg_clean: AbstractSignal, noise: AbstractSignal, sampling_rate: AbstractSignal, generator: AbstractSignal, device: AbstractSignal) -> AbstractSignal:
    """Ghost witness for GAN Patch Reconstruction."""
    result = AbstractSignal(
        shape=ppg_clean.shape,
        dtype="float64",
        sampling_rate=getattr(ppg_clean, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result

def witness_windowed_signal_reconstruction(sig: AbstractSignal, clean_indices: AbstractSignal, noisy_indices: AbstractSignal, sampling_rate: AbstractSignal, filter_signal: AbstractSignal) -> AbstractSignal:
    """Ghost witness for Windowed Signal Reconstruction."""
    result = AbstractSignal(
        shape=sig.shape,
        dtype="float64",
        sampling_rate=getattr(sig, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result
