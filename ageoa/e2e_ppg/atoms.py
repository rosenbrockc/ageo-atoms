"""Stateless atoms for E2E-PPG windowed processing."""

from __future__ import annotations

from typing import Any, Tuple

import icontract
import numpy as np

from ageoa.e2e_ppg.state_models import PPGState
from ageoa.ghost.abstract import AbstractArray
from ageoa.ghost.registry import register_atom


# Ghost Witness

def witness_ppg_process(new_samples: AbstractArray) -> AbstractArray:
    """Ghost witness for Sliding-Window Signal Quality Refinement."""
    return AbstractArray(shape=new_samples.shape, dtype="float64")


@register_atom(witness_ppg_process)
@icontract.require(lambda new_samples: new_samples.ndim == 1, "Samples must be 1D")
@icontract.ensure(lambda result, new_samples: result[0].shape == new_samples.shape, "Output length must match input window length")
def process_ppg(
    new_samples: np.ndarray[Any, Any],
    state: PPGState,
) -> Tuple[np.ndarray[Any, Any], PPGState]:
    """Process new signal samples with sliding-window quality refinement and deterministic fallback.
    """
    current_buffer = (state.buffer or []) + new_samples.tolist()
    max_size = 30 * (state.sampling_rate or 20)
    if len(current_buffer) > max_size:
        current_buffer = current_buffer[-max_size:]

    try:
        from ppg_sqa import sqa  # type: ignore

        sig = np.array(current_buffer)
        clean_ind, noisy_ind = sqa(sig, state.sampling_rate or 20, filter_signal=True)
        is_reliable = len(noisy_ind) == 0
    except ImportError:
        clean_ind, noisy_ind = [], []
        is_reliable = True

    if not is_reliable:
        try:
            from ppg_reconstruction import reconstruction  # type: ignore

            reconstructed_sig, clean_ind, noisy_ind = reconstruction(
                sig,
                clean_ind,
                noisy_ind,
                state.sampling_rate or 20,
                filter_signal=False,
            )
            current_buffer = reconstructed_sig.tolist()
            is_reliable = True
        except ImportError:
            pass

    next_state = state.model_copy(update={
        "buffer": current_buffer,
        "is_reliable": is_reliable,
        "clean_indices": clean_ind,
        "noisy_indices": noisy_ind,
    })

    if new_samples.size == 0:
        window = np.empty((0,), dtype=np.float64)
    else:
        window = np.asarray(current_buffer[-new_samples.size :], dtype=np.float64)

    return window, next_state
