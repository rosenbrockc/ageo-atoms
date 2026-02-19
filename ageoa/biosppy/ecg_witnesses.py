"""Ghost witness functions for ECG atoms."""

from __future__ import annotations

try:
    from ageoa.ghost.abstract import AbstractSignal
except ImportError:
    pass


def _template_width(signal: AbstractSignal, before_sec: float = 0.2, after_sec: float = 0.4) -> int:
    return max(1, int(round((before_sec + after_sec) * signal.sampling_rate)))


def witness_bandpass_filter(signal: AbstractSignal) -> AbstractSignal:
    """Bandpass filter preserves time-domain shape and sampling rate."""
    signal.assert_domain("time")
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=signal.sampling_rate,
        domain="time",
        units=signal.units,
    )


def witness_r_peak_detection(filtered: AbstractSignal) -> AbstractSignal:
    """R-peak detection returns index-domain sample positions."""
    filtered.assert_domain("time")
    return AbstractSignal(
        shape=(0,),
        dtype="int64",
        sampling_rate=filtered.sampling_rate,
        domain="index",
        units="samples",
    )


def witness_peak_correction(filtered: AbstractSignal, rpeaks: AbstractSignal) -> AbstractSignal:
    """Peak correction preserves index-domain cardinality."""
    filtered.assert_domain("time")
    rpeaks.assert_domain("index")
    return AbstractSignal(
        shape=rpeaks.shape,
        dtype="int64",
        sampling_rate=filtered.sampling_rate,
        domain="index",
        units="samples",
    )


def witness_template_extraction(
    filtered: AbstractSignal,
    rpeaks: AbstractSignal,
) -> tuple[AbstractSignal, AbstractSignal]:
    """Template extraction emits (templates, kept_rpeaks)."""
    filtered.assert_domain("time")
    rpeaks.assert_domain("index")

    num_templates = rpeaks.shape[0] if len(rpeaks.shape) > 0 else 0
    width = _template_width(filtered)

    return (
        AbstractSignal(
            shape=(num_templates, width),
            dtype="float64",
            sampling_rate=filtered.sampling_rate,
            domain="time",
            units=filtered.units,
        ),
        AbstractSignal(
            shape=(num_templates,),
            dtype="int64",
            sampling_rate=filtered.sampling_rate,
            domain="index",
            units="samples",
        ),
    )


def witness_heart_rate_computation(rpeaks: AbstractSignal) -> tuple[AbstractSignal, AbstractSignal]:
    """Heart-rate computation returns (indices, bpm_values)."""
    rpeaks.assert_domain("index")

    n = max(0, (rpeaks.shape[0] - 1) if len(rpeaks.shape) > 0 else 0)
    return (
        AbstractSignal(
            shape=(n,),
            dtype="int64",
            sampling_rate=rpeaks.sampling_rate,
            domain="index",
            units="samples",
        ),
        AbstractSignal(
            shape=(n,),
            dtype="float64",
            sampling_rate=rpeaks.sampling_rate,
            domain="time",
            units="bpm",
        ),
    )


def witness_ssf_segmenter(signal: AbstractSignal) -> AbstractSignal:
    """SSF detector maps time-domain signal to index-domain peaks."""
    signal.assert_domain("time")
    return AbstractSignal(
        shape=(0,),
        dtype="int64",
        sampling_rate=signal.sampling_rate,
        domain="index",
        units="samples",
    )


def witness_christov_segmenter(signal: AbstractSignal) -> AbstractSignal:
    """Christov detector maps time-domain signal to index-domain peaks."""
    signal.assert_domain("time")
    return AbstractSignal(
        shape=(0,),
        dtype="int64",
        sampling_rate=signal.sampling_rate,
        domain="index",
        units="samples",
    )
