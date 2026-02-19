"""MINT atoms for multimeric protein sequence processing."""

from __future__ import annotations

import torch
import icontract
import numpy as np
from typing import List, Tuple, Any

from ageoa.ghost.registry import register_atom
from ageoa.mint.state_models import MINTProcessingState
from ageoa.mint.witnesses import (
    witness_multimer_collator,
    witness_protein_transformer,
    witness_chain_level_contextualizer,
)

# Mock/Simplified Alphabet for protein sequences
PROTEIN_ALPHABET = {
    "<pad>": 0, "<mask>": 1, "<cls>": 2, "<eos>": 3, "<unk>": 4,
    "A": 5, "C": 6, "D": 7, "E": 8, "F": 9, "G": 10, "H": 11, "I": 12,
    "K": 13, "L": 14, "M": 15, "N": 16, "P": 17, "Q": 18, "R": 19,
    "S": 20, "T": 21, "V": 22, "W": 23, "Y": 24
}

def tokenize_protein(seq: str) -> List[int]:
    return [PROTEIN_ALPHABET.get(aa, PROTEIN_ALPHABET["<unk>"]) for aa in seq.upper()]

@register_atom(witness_multimer_collator)
@icontract.require(lambda sequences: len(sequences) > 0, "At least one sequence required")
@icontract.ensure(lambda result: result[0][0].shape == result[0][1].shape, "Tokens and chain IDs must have same shape")
def multimer_collator(sequences: List[str], state: MINTProcessingState) -> Tuple[Tuple[torch.Tensor, torch.Tensor], MINTProcessingState]:
    """Pads, tokenizes, and concatenates multiple protein chains into a single multimeric representation."""
    # Simplified implementation of MINT CollateFn
    # We treat the input list as a single batch of multimeric chains for simplicity
    
    encoded_chains = []
    for seq in sequences:
        # Add cls and eos tokens per chain as per MINT convert logic
        tokens = [PROTEIN_ALPHABET["<cls>"]] + tokenize_protein(seq) + [PROTEIN_ALPHABET["<eos>"]]
        encoded_chains.append(torch.tensor(tokens, dtype=torch.long))
    
    # Pad each chain to its max length (not strictly necessary for concat but common)
    # However, MINT concatenates them: chains = torch.cat(chains, -1)
    
    tokens_cat = torch.cat(encoded_chains, dim=-1).unsqueeze(0) # Add batch dim
    
    # Create chain IDs
    chain_ids_list = []
    for i, chain in enumerate(encoded_chains):
        chain_ids_list.append(torch.full_like(chain, i, dtype=torch.int32))
    chain_ids_cat = torch.cat(chain_ids_list, dim=-1).unsqueeze(0)
    
    new_state = state.model_copy(update={
        "tokens": tokens_cat,
        "chain_ids": chain_ids_cat
    })
    
    return (tokens_cat, chain_ids_cat), new_state

@register_atom(witness_protein_transformer)
@icontract.require(lambda tokens: tokens.ndim == 2, "Tokens must be (Batch, Seq)")
@icontract.ensure(lambda result: result[0].ndim == 3, "Output must be (Batch, Seq, Embed)")
def protein_transformer(tokens: torch.Tensor, chain_ids: torch.Tensor, state: MINTProcessingState) -> Tuple[torch.Tensor, MINTProcessingState]:
    """Black-box transformer boundary (OpaqueAtom stub)."""
    # In a real scenario, this would call the ESM-2 model.
    # Here we simulate the output dimension.
    batch_size, seq_len = tokens.shape
    embed_dim = 1280
    
    # Placeholder for actual model inference
    representations = torch.randn(batch_size, seq_len, embed_dim)
    
    new_state = state.model_copy(update={
        "representations": representations
    })
    
    return representations, new_state

@register_atom(witness_chain_level_contextualizer)
@icontract.require(lambda representations, tokens: representations.shape[1] == tokens.shape[1], "Representations and tokens length mismatch")
def chain_level_contextualizer(representations: torch.Tensor, tokens: torch.Tensor, chain_ids: torch.Tensor, state: MINTProcessingState) -> Tuple[torch.Tensor, MINTProcessingState]:
    """Applies masking and per-chain mean pooling to extract sequence-level embeddings."""
    # Logic from MINTWrapper.forward
    
    # Create mask to exclude special tokens
    mask = (
        (~tokens.eq(PROTEIN_ALPHABET["<cls>"])) &
        (~tokens.eq(PROTEIN_ALPHABET["<eos>"])) &
        (~tokens.eq(PROTEIN_ALPHABET["<pad>"]))
    )
    
    max_chain_id = chain_ids.max().item()
    mean_chain_outs = []
    
    for chain_id in range(int(max_chain_id) + 1):
        chain_mask = (chain_ids == chain_id) & mask
        # Expand mask for representation pooling
        chain_mask_exp = chain_mask.unsqueeze(-1).expand_as(representations)
        
        # Masked representations for this chain
        masked_out = representations * chain_mask_exp
        sum_masked = masked_out.sum(dim=1)
        mask_counts = chain_mask.sum(dim=1, keepdim=True).float()
        
        # Avoid division by zero
        mask_counts[mask_counts == 0] = 1.0
        mean_out = sum_masked / mask_counts
        mean_chain_outs.append(mean_out)
        
    # Concatenate per-chain embeddings
    contextualized = torch.cat(mean_chain_outs, dim=-1)
    
    return contextualized, state
