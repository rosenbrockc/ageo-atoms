"""Auto-generated ghost witness functions for PCG simulation."""

from __future__ import annotations

try:
    from ageoa.ghost.abstract import AbstractSignal
except ImportError:
    pass

def witness_shannon_energy(signal: AbstractSignal) -> AbstractSignal:
    """Ghost witness for Shannon Energy calculation.
    
    Postconditions:
        - Output shape matches input.
        - Domain is 'envelope'.
    """
    signal.assert_domain("time")
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=signal.sampling_rate,
        domain="envelope",
        units="normalized",
    )

def witness_pcg_segmentation(envelope: AbstractSignal) -> tuple[AbstractSignal, AbstractSignal]:
    """Ghost witness for Cyclic Event Segmentation.
    
    Postconditions:
        - Returns (s1_indices, s2_indices).
    """
    envelope.assert_domain("envelope")
    return (
        AbstractSignal(
            shape=(0,),
            dtype="int64",
            sampling_rate=envelope.sampling_rate,
            domain="index",
            units="samples",
        ),
        AbstractSignal(
            shape=(0,),
            dtype="int64",
            sampling_rate=envelope.sampling_rate,
            domain="index",
            units="samples",
        )
    )
