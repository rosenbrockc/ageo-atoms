"""Abstract value types for Ghost Witness metadata propagation.

These models carry *only* metadata about signals and state — shape, dtype,
sampling rate, domain — never the actual sample data.  Witnesses consume
and produce these types to simulate a computation graph at near-zero cost.
"""

from __future__ import annotations

from typing import Tuple

from pydantic import BaseModel, Field


class AbstractSignal(BaseModel):
    """The 'Ghost' representation of a signal array.

    Carries the structural envelope of an ndarray without any sample data.
    Witnesses read and transform these to verify that a graph is wired
    correctly before any heavy computation runs.
    """

    shape: Tuple[int, ...] = Field(..., description="Array shape, e.g. (1024,) or (128, 2)")
    dtype: str = Field(..., description="NumPy dtype string, e.g. 'float64', 'complex128'")
    sampling_rate: float = Field(..., gt=0, description="Sampling frequency in Hz")
    domain: str = Field(default="time", description="Signal domain: 'time', 'freq', 'quefrency', 'index'")
    units: str = Field(default="volts", description="Physical units of the signal")

    @property
    def duration(self) -> float:
        """Signal duration in seconds (only meaningful for time-domain signals)."""
        if self.domain == "time" and self.sampling_rate > 0 and len(self.shape) > 0:
            return self.shape[0] / self.sampling_rate
        return 0.0

    @property
    def nyquist(self) -> float:
        """Nyquist frequency in Hz."""
        return self.sampling_rate / 2.0

    def assert_compatible(self, other: AbstractSignal) -> None:
        """Assert that two signals are compatible for element-wise operations.

        Raises:
            ValueError: If sampling rates or shapes don't match.
        """
        if self.sampling_rate != other.sampling_rate:
            raise ValueError(
                f"Sampling rate mismatch: {self.sampling_rate} vs {other.sampling_rate}"
            )
        if self.shape != other.shape:
            raise ValueError(
                f"Shape mismatch: {self.shape} vs {other.shape}"
            )

    def assert_domain(self, expected: str) -> None:
        """Assert that the signal is in the expected domain.

        Raises:
            ValueError: If the signal domain doesn't match.
        """
        if self.domain != expected:
            raise ValueError(
                f"Domain mismatch: expected '{expected}', got '{self.domain}'"
            )


class AbstractBeatPool(BaseModel):
    """Abstract state for accumulative beat detection / SQI pipelines.

    Models the evolving confidence state of a beat accumulator without
    storing any actual waveform data.
    """

    size: int = Field(default=0, ge=0, description="Number of beats accumulated so far")
    is_calibrated: bool = Field(default=False, description="Whether the pool has enough beats to be reliable")
    calibration_threshold: int = Field(default=50, description="Minimum beats required for calibration")

    def accumulate(self, new_beat_count: int) -> "AbstractBeatPool":
        """Return a new pool with additional beats accumulated.

        Args:
            new_beat_count: Number of new beats to add.

        Returns:
            Updated AbstractBeatPool with new size and calibration status.
        """
        new_size = self.size + new_beat_count
        return AbstractBeatPool(
            size=new_size,
            is_calibrated=new_size >= self.calibration_threshold,
            calibration_threshold=self.calibration_threshold,
        )
