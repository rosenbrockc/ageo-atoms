"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations

try:
    from ageoa.ghost.abstract import AbstractSignal
except ImportError:
    pass

def witness_bandpass_filter(signal: AbstractSignal) -> AbstractSignal:
    """Ghost witness for Bandpass Filter.

    Postconditions:
        - Output shape matches input.
        - Domain is 'time'.
    """
    signal.assert_domain("time")
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=signal.sampling_rate,
        domain="time",
    )

def witness_r_peak_detection(filtered: AbstractSignal) -> AbstractSignal:
    """Ghost witness for R-Peak Detection.

    Postconditions:
        - Output is a 1D array of indices.
        - Domain is 'index'.
    """
    filtered.assert_domain("time")
    return AbstractSignal(
        shape=(0,),  # Dynamic length
        dtype="int64",
        sampling_rate=filtered.sampling_rate,
        domain="index",
        units="samples",
    )

def witness_peak_correction(filtered: AbstractSignal, rpeaks: AbstractSignal) -> AbstractSignal:
    """Ghost witness for Peak Correction.

    Postconditions:
        - Output shape matches rpeaks input (same number of peaks).
        - Domain is 'index'.
    """
    filtered.assert_domain("time")
    rpeaks.assert_domain("index")
    return AbstractSignal(
        shape=rpeaks.shape,
        dtype="int64",
        sampling_rate=filtered.sampling_rate,
        domain="index",
        units="samples",
    )

def witness_template_extraction(filtered: AbstractSignal, rpeaks: AbstractSignal) -> tuple[AbstractSignal, AbstractSignal]:
    """Ghost witness for Template Extraction.

    Postconditions:
        - Returns (templates, corrected_rpeaks).
        - templates shape is (N_peaks, Window_size).
    """
    filtered.assert_domain("time")
    rpeaks.assert_domain("index")
    # Assume 0.6s window by default (0.2 + 0.4)
    window_size = int(0.6 * filtered.sampling_rate)
    return (
        AbstractSignal(
            shape=(0, window_size), # Dynamic N_peaks
            dtype="float64",
            sampling_rate=filtered.sampling_rate,
            domain="time",
            units="volts",
        ),
        AbstractSignal(
            shape=rpeaks.shape,
            dtype="int64",
            sampling_rate=filtered.sampling_rate,
            domain="index",
            units="samples",
        )
    )

def witness_heart_rate_computation(rpeaks: AbstractSignal) -> tuple[AbstractSignal, AbstractSignal]:
    """Ghost witness for Heart Rate Computation.

    Postconditions:
        - Returns (index, heart_rate).
        - Heart rate is one element shorter than rpeaks.
    """
    rpeaks.assert_domain("index")
    return (
        AbstractSignal(
            shape=(0,),
            dtype="int64",
            sampling_rate=rpeaks.sampling_rate,
            domain="index",
            units="samples",
        ),
        AbstractSignal(
            shape=(0,),
            dtype="float64",
            sampling_rate=rpeaks.sampling_rate,
            domain="time",
            units="bpm",
        )
    )
