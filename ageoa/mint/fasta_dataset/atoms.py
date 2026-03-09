from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import icontract
from ageoa.ghost.registry import register_atom

from .witnesses import (
    witness_dataset_state_initialization,
    witness_dataset_length_query,
    witness_dataset_item_retrieval,
    witness_token_budget_batch_planning,
)


@register_atom(witness_dataset_state_initialization)
@icontract.require(lambda sequence_labels, sequence_strs: len(sequence_labels) == len(sequence_strs), "sequence_labels and sequence_strs must have equal length")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def dataset_state_initialization(
    sequence_labels: list[str], sequence_strs: list[str], fasta_file: str
) -> object:
    """Build an immutable dataset state from in-memory inputs or a FASTA file.

    Args:
        sequence_labels: List of sequence identifiers.
        sequence_strs: List of sequence strings.
        fasta_file: Path to FASTA file for loading.

    Returns:
        Immutable dataset state object.
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_dataset_length_query)
@icontract.require(lambda dataset_state: dataset_state is not None, "dataset_state cannot be None")
@icontract.ensure(lambda result: isinstance(result, int), "result must be int")
def dataset_length_query(dataset_state: object) -> int:
    """Return the number of labeled sequences in dataset_state.

    Args:
        dataset_state: Must contain sequence_labels.

    Returns:
        Number of sequences, >= 0.
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_dataset_item_retrieval)
@icontract.require(lambda idx: isinstance(idx, int), "idx must be int")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def dataset_item_retrieval(dataset_state: object, idx: int) -> tuple[str, str]:
    """Retrieve a single labeled sequence by index from dataset_state.

    Args:
        dataset_state: Must contain sequence_labels and sequence_strs.
        idx: Index must be valid for dataset length.

    Returns:
        Tuple of (label, sequence_str).
    """
    raise NotImplementedError("Wire to original implementation")


@register_atom(witness_token_budget_batch_planning)
@icontract.require(lambda toks_per_batch: isinstance(toks_per_batch, int), "toks_per_batch must be int")
@icontract.require(lambda extra_toks_per_seq: isinstance(extra_toks_per_seq, int), "extra_toks_per_seq must be int")
@icontract.ensure(lambda result: isinstance(result, list), "result must be list")
def token_budget_batch_planning(
    dataset_state: object, toks_per_batch: int, extra_toks_per_seq: int
) -> list[list[int]]:
    """Compute batch index groups under a token budget.

    Args:
        dataset_state: Uses sequence_strs for length-aware batching.
        toks_per_batch: Token budget per batch, > 0.
        extra_toks_per_seq: Per-sequence overhead tokens, >= 0.

    Returns:
        List of batches, each a list of dataset indices.
    """
    raise NotImplementedError("Wire to original implementation")
