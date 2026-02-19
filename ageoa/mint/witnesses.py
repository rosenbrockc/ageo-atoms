from __future__ import annotations
import torch
from ageoa.ghost.abstract import AbstractSignal, AbstractArray

def witness_multimer_collator(sequences: list[str], state: AbstractSignal) -> tuple[tuple[AbstractArray, AbstractArray], AbstractSignal]:
    """Ghost witness for Multimer Collator."""
    # Assume fixed max length or based on input sequences
    batch_size = len(sequences)
    # Estimate total length (concatenated chains)
    total_len = sum(len(s) + 2 for s in sequences) # +2 for cls/eos
    
    tokens = AbstractArray(shape=(batch_size, total_len), dtype="int64")
    chain_ids = AbstractArray(shape=(batch_size, total_len), dtype="int32")
    return (tokens, chain_ids), state

def witness_protein_transformer(tokens: AbstractArray, chain_ids: AbstractArray, state: AbstractSignal) -> tuple[AbstractArray, AbstractSignal]:
    """Ghost witness for Protein Transformer."""
    # Transformer output has same seq len, but adds embedding dim
    embed_dim = 1280 # ESM-2 large default
    representations = AbstractArray(shape=tokens.shape + (embed_dim,), dtype="float32")
    return representations, state

def witness_chain_level_contextualizer(representations: AbstractArray, tokens: AbstractArray, chain_ids: AbstractArray, state: AbstractSignal) -> tuple[AbstractArray, AbstractSignal]:
    """Ghost witness for Chain-level Contextualizer."""
    # Reduces sequence dimension to number of chains, concatenated along embed dim
    # Or just concatenated mean embeddings.
    # In MINTWrapper it returns torch.cat(mean_chain_outs, dim=-1)
    # If we have 2 chains, it returns (batch, 2 * embed_dim)
    n_chains = 2 # Heuristic for multimer
    embed_dim = representations.shape[-1]
    result = AbstractArray(shape=(representations.shape[0], n_chains * embed_dim), dtype="float32")
    return result, state
