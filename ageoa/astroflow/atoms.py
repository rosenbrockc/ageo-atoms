"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_dedispersionkernel)  # type: ignore[name-defined]
@icontract.require(lambda input_data: input_data is not None, "input_data cannot be None")
@icontract.require(lambda delay_table: delay_table is not None, "delay_table cannot be None")
@icontract.require(lambda dm_steps: dm_steps is not None, "dm_steps cannot be None")
@icontract.require(lambda time_downsample: time_downsample is not None, "time_downsample cannot be None")
@icontract.require(lambda down_ndata: down_ndata is not None, "down_ndata cannot be None")
@icontract.require(lambda nchans: nchans is not None, "nchans cannot be None")
@icontract.require(lambda shared_mem_size: shared_mem_size is not None, "shared_mem_size cannot be None")
@icontract.require(lambda block_dim_x: block_dim_x is not None, "block_dim_x cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "DedispersionKernel output must not be None")
def dedispersionkernel(input_data: "np.ndarray[np.generic]", delay_table: "np.ndarray[np.generic]", dm_steps: int, time_downsample: int, down_ndata: int, nchans: int, shared_mem_size: int, block_dim_x: int) -> "np.ndarray[np.generic]":
    """Applies a dedispersion kernel to input data using a pre-computed delay table, transforming the data across different dispersion measure (DM) steps.

    Args:
        input_data: Raw observational data array.
        delay_table: Table of delays to apply for each channel and DM step.
        dm_steps: Number of dispersion measures to process.
        time_downsample: Factor by which to downsample the time series.
        down_ndata: The size of the output data array after downsampling.
        nchans: Number of frequency channels in the input data.
        shared_mem_size: Size of shared memory for the computational kernel (e.g., on a GPU).
        block_dim_x: The block dimension in the x-axis for the computational kernel.

    Returns:
        The transformed, dedispersed data.
    """
    raise NotImplementedError("Wire to original implementation")