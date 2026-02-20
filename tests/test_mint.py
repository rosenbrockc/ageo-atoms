import torch

from ageoa.mint.atoms import (
    chain_level_contextualizer,
    multimer_collator,
    protein_transformer,
)
from ageoa.mint.state_models import MINTProcessingState


def test_mint_pipeline():
    sequences = ["MKTVRQERLKSIVVLGAGFVGSCYAFALNQ", "MLPLVVLGA"]
    state = MINTProcessingState(rng_seed=123)

    # 1. Multimer Collator
    (tokens, chain_ids), state = multimer_collator(sequences, state)
    assert tokens.ndim == 2
    assert chain_ids.ndim == 2
    assert tokens.shape == chain_ids.shape

    # 2. Protein Transformer (seeded stochastic boundary)
    representations, state = protein_transformer(tokens, chain_ids, state)
    assert representations.ndim == 3
    assert representations.shape[0] == tokens.shape[0]
    assert representations.shape[1] == tokens.shape[1]
    assert representations.shape[2] == 1280

    # 3. Chain-level Contextualizer
    contextualized, state = chain_level_contextualizer(representations, tokens, chain_ids, state)
    assert contextualized.ndim == 2
    assert contextualized.shape[0] == 1  # batch size
    assert contextualized.shape[1] == 2 * 1280  # 2 chains


def test_protein_transformer_seed_is_reproducible():
    sequences = ["AAAA", "BBBB"]
    (tokens, chain_ids), _ = multimer_collator(sequences, MINTProcessingState())

    base_state = MINTProcessingState(rng_seed=7, rng_counter=0)

    out1, _ = protein_transformer(tokens, chain_ids, base_state)
    out2, _ = protein_transformer(tokens, chain_ids, base_state)
    assert torch.equal(out1, out2)

    # Sequential call advances counter, so stream changes deterministically.
    out3, next_state = protein_transformer(tokens, chain_ids, base_state)
    out4, _ = protein_transformer(tokens, chain_ids, next_state)
    assert not torch.equal(out3, out4)

    # Recreate the advanced state and confirm reproducibility.
    recreated = MINTProcessingState(rng_seed=7, rng_counter=1)
    out5, _ = protein_transformer(tokens, chain_ids, recreated)
    assert torch.equal(out4, out5)


def test_chain_contextualizer_scales_with_chain_count():
    sequences = ["AA", "BB", "CC"]
    (tokens, chain_ids), state = multimer_collator(sequences, MINTProcessingState(rng_seed=1))
    representations, state = protein_transformer(tokens, chain_ids, state)

    contextualized, _ = chain_level_contextualizer(representations, tokens, chain_ids, state)
    assert contextualized.shape == (1, 3 * 1280)
