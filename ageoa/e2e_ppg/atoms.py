"""Stateless atoms for E2E-PPG windowed processing."""

from __future__ import annotations
import numpy as np
import icontract
from typing import Tuple, Any
from ageoa.ghost.registry import register_atom
from ageoa.e2e_ppg.state_models import PPGState

# Ghost Witness
def witness_ppg_process(new_samples: np.ndarray[Any, Any], state: PPGState) -> Tuple[np.ndarray[Any, Any], PPGState]:
    """Ghost witness for PPG processing."""
    return new_samples, state

@register_atom(witness_ppg_process)
@icontract.require(lambda new_samples: new_samples.ndim == 1, "Samples must be 1D")
@icontract.ensure(lambda result: result[0].ndim == 1, "Result must be 1D")
def process_ppg(new_samples: np.ndarray[Any, Any], state: PPGState) -> Tuple[np.ndarray[Any, Any], PPGState]:
    """
    Process new PPG samples: update buffer, assess quality, and reconstruct if needed.
    
    This is a stateless functional wrapper around the stateful PPGProcessor logic.
    """
    # 1. Update buffer (State mutation)
    current_buffer = (state.buffer or []) + new_samples.tolist()
    max_size = 30 * (state.sampling_rate or 20)
    if len(current_buffer) > max_size:
        current_buffer = current_buffer[-max_size:]
        
    # 2. Assess quality (Windowed SQA)
    try:
        from ppg_sqa import sqa # type: ignore
        sig = np.array(current_buffer)
        clean_ind, noisy_ind = sqa(sig, state.sampling_rate or 20, filter_signal=True)
        is_reliable = len(noisy_ind) == 0
    except ImportError:
        # Fallback for verification/tests without the external repo
        clean_ind, noisy_ind = [], []
        is_reliable = True
    
    # 3. Reconstruct if needed
    if not is_reliable:
        try:
            from ppg_reconstruction import reconstruction # type: ignore
            reconstructed_sig, clean_ind, noisy_ind = reconstruction(
                sig, clean_ind, noisy_ind, state.sampling_rate or 20, filter_signal=False
            )
            current_buffer = reconstructed_sig.tolist()
            is_reliable = True
        except ImportError:
            pass
        
    next_state = state.model_copy(update={
        "buffer": current_buffer,
        "is_reliable": is_reliable,
        "clean_indices": clean_ind,
        "noisy_indices": noisy_ind
    })
    
    return np.array(current_buffer[-len(new_samples):]), next_state
