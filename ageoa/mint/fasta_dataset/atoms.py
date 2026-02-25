"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from pathlib import Path
from typing import Any, Callable, List, Tuple, TypeVar, cast

from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]
_F = TypeVar("_F", bound=Callable[..., Any])

def typed_register_atom(witness: Any) -> Callable[[_F], _F]:
    return cast(Callable[[_F], _F], register_atom(witness))

FastaDatasetState = Any

witness_dataset_state_initialization: Any
witness_dataset_length_query: Any
witness_dataset_item_retrieval: Any
witness_token_budget_batch_planning: Any

@register_atom(witness_dataset_state_initialization)
@icontract.require(lambda sequence_labels: sequence_labels is not None, "sequence_labels cannot be None")
@icontract.require(lambda sequence_strs: sequence_strs is not None, "sequence_strs cannot be None")
@icontract.require(lambda fasta_file: fasta_file is not None, "fasta_file cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "dataset_state_initialization output must not be None")
@typed_register_atom(witness_dataset_length_query)
    """Build an immutable dataset state (sequence_labels, sequence_strs) either from direct in-memory inputs or by loading/parsing a FASTA file.
def dataset_state_initialization(sequence_labels: Any, sequence_strs: Any, fasta_file: Any) -> FastaDatasetState:
    """Build an immutable dataset state from in-memory inputs or a FASTA file."""

@register_atom(witness_dataset_length_query)
@typed_register_atom(witness_dataset_item_retrieval)
@icontract.ensure(lambda result, **kwargs: result is not None, "dataset_length_query output must not be None")
def dataset_length_query(dataset_state: FastaDatasetState) -> int:
    """Return the number of labeled sequences in dataset_state.

    Args:
        dataset_state: Must contain sequence_labels.

    Returns:
        >= 0
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_dataset_item_retrieval)
@icontract.require(lambda dataset_state: dataset_state is not None, "dataset_state cannot be None")
@icontract.require(lambda idx: idx is not None, "idx cannot be None")
@typed_register_atom(witness_token_budget_batch_planning)
def dataset_item_retrieval(dataset_state: FastaDatasetState, idx: int) -> Tuple[str, str]:
    """Retrieve a single labeled sequence by index from immutable dataset_state.

    Args:
        dataset_state: Must contain sequence_labels and sequence_strs with equal length.
        idx: Index must be valid for dataset length.

    Returns:
        (label, sequence_str)
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_token_budget_batch_planning)
@icontract.require(lambda dataset_state: dataset_state is not None, "dataset_state cannot be None")
@icontract.require(lambda toks_per_batch: toks_per_batch is not None, "toks_per_batch cannot be None")
@icontract.require(lambda extra_toks_per_seq: extra_toks_per_seq is not None, "extra_toks_per_seq cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "token_budget_batch_planning output must not be None")
def token_budget_batch_planning(dataset_state: FastaDatasetState, toks_per_batch: int, extra_toks_per_seq: int) -> List[List[int]]:
    """Compute batch index groups under a token budget using sequence strings and per-sequence overhead.

    Args:
        dataset_state: Uses sequence_strs for length-aware batching.
        toks_per_batch: > 0
        extra_toks_per_seq: >= 0

    Returns:
        Each inner list is a batch of dataset indices satisfying token budget.
    """
    raise NotImplementedError("Wire to original implementation")