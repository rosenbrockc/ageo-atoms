from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray


def witness_multimer_collator(sequences: list[str]) -> tuple[AbstractArray, AbstractArray]:
    """Ghost witness for Multimer Collator."""
    total_len = sum(len(s) + 2 for s in sequences)  # +2 for cls/eos
    n_chains = max(1, len(sequences))

    tokens = AbstractArray(shape=(1, total_len), dtype="int64")
    chain_ids = AbstractArray(shape=(1, total_len), dtype="int32", min_val=0, max_val=float(n_chains - 1))
    return tokens, chain_ids


def witness_protein_transformer(tokens: AbstractArray, chain_ids: AbstractArray) -> AbstractArray:
    """Ghost witness for Protein Transformer."""
    if tokens.shape != chain_ids.shape:
        raise ValueError(f"Tokens and chain_ids shape mismatch: {tokens.shape} vs {chain_ids.shape}")

    embed_dim = 1280
    return AbstractArray(shape=tokens.shape + (embed_dim,), dtype="float32")


def witness_chain_level_contextualizer(
    representations: AbstractArray,
    tokens: AbstractArray,
    chain_ids: AbstractArray,
) -> AbstractArray:
    """Ghost witness for Chain-level Contextualizer."""
    if representations.shape[:2] != tokens.shape:
        raise ValueError(
            f"Representations/tokens mismatch: {representations.shape} vs {tokens.shape}"
        )
    if tokens.shape != chain_ids.shape:
        raise ValueError(f"Tokens/chain_ids shape mismatch: {tokens.shape} vs {chain_ids.shape}")

    embed_dim = representations.shape[-1]
    n_chains = int(chain_ids.max_val) + 1 if chain_ids.max_val is not None else 1
    n_chains = max(1, n_chains)
    return AbstractArray(shape=(representations.shape[0], n_chains * embed_dim), dtype="float32")
