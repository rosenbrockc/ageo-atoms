from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal
import networkx as nx  # type: ignore


def witness_dedispersionkernel(input_data: AbstractSignal, delay_table: AbstractSignal, dm_steps: AbstractSignal, time_downsample: AbstractSignal, down_ndata: AbstractSignal, nchans: AbstractSignal, shared_mem_size: AbstractSignal, block_dim_x: AbstractSignal) -> AbstractSignal:
    """Ghost witness for DedispersionKernel."""
    result = AbstractSignal(
        shape=input_data.shape,
        dtype="float64",
        sampling_rate=getattr(input_data, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result
