import torch
import pytest
from ageoa.mint.atoms import multimer_collator, protein_transformer, chain_level_contextualizer
from ageoa.mint.state_models import MINTProcessingState

def test_mint_pipeline():
    sequences = ["MKTVRQERLKSIVVLGAGFVGSCYAFALNQ", "MLPLVVLGA"]
    state = MINTProcessingState()
    
    # 1. Multimer Collator
    (tokens, chain_ids), state = multimer_collator(sequences, state)
    assert tokens.ndim == 2
    assert chain_ids.ndim == 2
    assert tokens.shape == chain_ids.shape
    
    # 2. Protein Transformer (Opaque boundary)
    representations, state = protein_transformer(tokens, chain_ids, state)
    assert representations.ndim == 3
    assert representations.shape[0] == tokens.shape[0]
    assert representations.shape[1] == tokens.shape[1]
    assert representations.shape[2] == 1280
    
    # 3. Chain-level Contextualizer
    contextualized, state = chain_level_contextualizer(representations, tokens, chain_ids, state)
    assert contextualized.ndim == 2
    assert contextualized.shape[0] == 1 # batch size
    assert contextualized.shape[1] == 2 * 1280 # 2 chains
