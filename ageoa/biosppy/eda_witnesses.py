"""Auto-generated ghost witness functions for EDA simulation."""

from __future__ import annotations

try:
    from ageoa.ghost.abstract import AbstractSignal
except ImportError:
    pass

def witness_gamboa_segmenter(signal: AbstractSignal) -> AbstractSignal:
    """Ghost witness for Gamboa Phasic Response Detection.
    
    Postconditions:
        - Output is a 1D array of onset indices.
        - Domain is 'index'.
    """
    signal.assert_domain("time")
    return AbstractSignal(
        shape=(0,),
        dtype="int64",
        sampling_rate=signal.sampling_rate,
        domain="index",
        units="samples",
    )

def witness_eda_feature_extraction(signal: AbstractSignal, onsets: AbstractSignal) -> tuple[AbstractSignal, AbstractSignal, AbstractSignal]:
    """Ghost witness for EDA feature extraction.
    
    Postconditions:
        - Returns (amplitudes, rise_times, decay_times).
    """
    signal.assert_domain("time")
    onsets.assert_domain("index")
    return (
        AbstractSignal(shape=onsets.shape, dtype="float64", sampling_rate=signal.sampling_rate, domain="time", units="uS"),
        AbstractSignal(shape=onsets.shape, dtype="float64", sampling_rate=signal.sampling_rate, domain="time", units="seconds"),
        AbstractSignal(shape=onsets.shape, dtype="float64", sampling_rate=signal.sampling_rate, domain="time", units="seconds"),
    )
