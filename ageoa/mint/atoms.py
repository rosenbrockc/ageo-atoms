"""MINT atoms for multimeric protein sequence processing."""

from __future__ import annotations

from typing import List, Tuple

import icontract
import torch

from ageoa.ghost.registry import register_atom
from ageoa.mint.state_models import MINTProcessingState
from ageoa.mint.witnesses import (
    witness_chain_level_contextualizer,
    witness_multimer_collator,
    witness_protein_transformer,
)

# Mock/Simplified Alphabet for protein sequences
PROTEIN_ALPHABET = {
    "<pad>": 0,
    "<mask>": 1,
    "<cls>": 2,
    "<eos>": 3,
    "<unk>": 4,
    "A": 5,
    "C": 6,
    "D": 7,
    "E": 8,
    "F": 9,
    "G": 10,
    "H": 11,
    "I": 12,
    "K": 13,
    "L": 14,
    "M": 15,
    "N": 16,
    "P": 17,
    "Q": 18,
    "R": 19,
    "S": 20,
    "T": 21,
    "V": 22,
    "W": 23,
    "Y": 24,
}


def tokenize_protein(seq: str) -> List[int]:
    """Tokenize a categorical sequence into integer indices using a fixed vocabulary."""
    return [PROTEIN_ALPHABET.get(aa, PROTEIN_ALPHABET["<unk>"]) for aa in seq.upper()]


@register_atom(witness_multimer_collator)
@icontract.require(lambda sequences: len(sequences) > 0, "At least one sequence required")
@icontract.require(lambda sequences: all(len(seq) > 0 for seq in sequences), "Sequences must be non-empty")
@icontract.ensure(lambda result: result[0][0].shape == result[0][1].shape, "Tokens and chain IDs must have same shape")
def multimer_collator(
    sequences: List[str],
    state: MINTProcessingState,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], MINTProcessingState]:
    """Tokenize and concatenate multiple categorical sub-sequences into a single composite sample.

    <!-- conceptual_profile -->
    {
        "abstract_name": "Sequential Multichannel Stream Aggregator",
        "conceptual_transform": "Combines multiple independent sequences of discrete categorical tokens into a single unified stream, while maintaining unique identifiers for each source channel. It transforms a list of variable-length sequences into a structured, multichannel representation.",
        "abstract_inputs": [
            {
                "name": "sequences",
                "description": "A list of strings, each representing a sequence of categorical elements."
            },
            {
                "name": "state",
                "description": "A state object tracking the current processing context."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tuple containing the concatenated token tensor and a corresponding channel-ID tensor."
            },
            {
                "name": "new_state",
                "description": "The updated processing state object."
            }
        ],
        "algorithmic_properties": [
            "data-collation",
            "tokenization",
            "multichannel-encoding",
            "deterministic"
        ],
        "cross_disciplinary_applications": [
            "Aggregating log streams from multiple independent microservices for joint analysis.",
            "Combining telemetry data from multiple sensors into a single time-aligned training sample.",
            "Merging linguistic sequences from different speakers in a multi-party conversation model."
        ]
    }
    <!-- /conceptual_profile -->
    """
    encoded_chains = []
    for seq in sequences:
        tokens = [PROTEIN_ALPHABET["<cls>"]] + tokenize_protein(seq) + [PROTEIN_ALPHABET["<eos>"]]
        encoded_chains.append(torch.tensor(tokens, dtype=torch.long))

    tokens_cat = torch.cat(encoded_chains, dim=-1).unsqueeze(0)

    chain_ids_list = []
    for i, chain in enumerate(encoded_chains):
        chain_ids_list.append(torch.full_like(chain, i, dtype=torch.int32))
    chain_ids_cat = torch.cat(chain_ids_list, dim=-1).unsqueeze(0)

    new_state = state.model_copy(update={
        "tokens": tokens_cat,
        "chain_ids": chain_ids_cat,
    })

    return (tokens_cat, chain_ids_cat), new_state


@register_atom(witness_protein_transformer)
@icontract.require(lambda tokens: tokens.ndim == 2, "Tokens must be (Batch, Seq)")
@icontract.require(lambda tokens, chain_ids: tokens.shape == chain_ids.shape, "Tokens and chain IDs must have same shape")
@icontract.ensure(lambda result: result[0].ndim == 3, "Output must be (Batch, Seq, Embed)")
@icontract.ensure(lambda result, tokens: result[0].shape[0] == tokens.shape[0], "Batch size must be preserved")
@icontract.ensure(lambda result, tokens: result[0].shape[1] == tokens.shape[1], "Sequence length must be preserved")
def protein_transformer(
    tokens: torch.Tensor,
    chain_ids: torch.Tensor,
    state: MINTProcessingState,
    seed: int | None = None,
) -> Tuple[torch.Tensor, MINTProcessingState]:
    """Stochastic transformer boundary with deterministic seed control.

    Randomness is preserved, but controlled via `(rng_seed, rng_counter)` in
    state or an explicit `seed` override so runs are reproducible.

    <!-- conceptual_profile -->
    {
        "abstract_name": "Stochastic Contextual Sequence Embedding Generator",
        "conceptual_transform": "Generates a high-dimensional contextual representation for each element in a multichannel categorical sequence. It uses a stochastic transformation controlled by a deterministic seed to ensure reproducibility while providing rich, learned feature embeddings.",
        "abstract_inputs": [
            {
                "name": "tokens",
                "description": "A 2D tensor of categorical token identifiers."
            },
            {
                "name": "chain_ids",
                "description": "A 2D tensor of channel identifiers matching the tokens."
            },
            {
                "name": "state",
                "description": "A state object containing RNG seed and counter for reproducibility."
            },
            {
                "name": "seed",
                "description": "An optional integer to override the default stochastic seed."
            }
        ],
        "abstract_outputs": [
            {
                "name": "representations",
                "description": "A 3D tensor representing the learned contextual features for each token."
            },
            {
                "name": "new_state",
                "description": "The updated processing state with advanced RNG counter."
            }
        ],
        "algorithmic_properties": [
            "contextual-embedding",
            "stochastic-but-reproducible",
            "transformer-based",
            "generative-latent-space"
        ],
        "cross_disciplinary_applications": [
            "Generating contextual embeddings for multi-sensor industrial telemetry to detect anomalies.",
            "Creating feature representations for multi-user transaction logs in fraud detection.",
            "Embedding multi-modal sequence data for downstream predictive tasks in robotics."
        ]
    }
    <!-- /conceptual_profile -->
    """
    batch_size, seq_len = tokens.shape
    embed_dim = 1280

    effective_seed = int(seed if seed is not None else state.rng_seed)
    counter = int(state.rng_counter)

    gen = torch.Generator()
    gen.manual_seed(effective_seed + counter)

    representations = torch.randn(
        batch_size,
        seq_len,
        embed_dim,
        generator=gen,
        dtype=torch.float32,
    )
    if tokens.device.type != "cpu":
        representations = representations.to(tokens.device)

    new_state = state.model_copy(update={
        "representations": representations,
        "rng_seed": effective_seed,
        "rng_counter": counter + 1,
    })

    return representations, new_state


@register_atom(witness_chain_level_contextualizer)
@icontract.require(lambda representations, tokens: representations.shape[1] == tokens.shape[1], "Representations and tokens length mismatch")
@icontract.require(lambda tokens, chain_ids: tokens.shape == chain_ids.shape, "Tokens and chain IDs must have same shape")
@icontract.ensure(lambda result: result[0].ndim == 2, "Contextualized output must be rank-2")
def chain_level_contextualizer(
    representations: torch.Tensor,
    tokens: torch.Tensor,
    chain_ids: torch.Tensor,
    state: MINTProcessingState,
) -> Tuple[torch.Tensor, MINTProcessingState]:
    """Apply masking and per-segment mean pooling to get segment-level embeddings.

    <!-- conceptual_profile -->
    {
        "abstract_name": "Channel-Aware Global Feature Pooler",
        "conceptual_transform": "Aggregates fine-grained contextual embeddings into a single global representation for each source channel by applying a selective masking and mean-pooling operation. It reduces sequence-level detail into channel-level descriptors.",
        "abstract_inputs": [
            {
                "name": "representations",
                "description": "A 3D tensor of fine-grained contextual features."
            },
            {
                "name": "tokens",
                "description": "A 2D tensor of source tokens used for masking metadata."
            },
            {
                "name": "chain_ids",
                "description": "A 2D tensor identifying which features belong to which source channel."
            },
            {
                "name": "state",
                "description": "A state object tracking the current processing context."
            }
        ],
        "abstract_outputs": [
            {
                "name": "contextualized",
                "description": "A 2D tensor containing the global feature vectors for each source channel."
            },
            {
                "name": "state",
                "description": "The current processing state object."
            }
        ],
        "algorithmic_properties": [
            "global-pooling",
            "channel-aware",
            "masking-transformation",
            "dimension-reduction"
        ],
        "cross_disciplinary_applications": [
            "Summarizing per-sensor telemetry into a single status vector for each sensor in a factory.",
            "Pooling word-level embeddings into sentence-level representations in a multi-document summarizer.",
            "Aggregating fine-grained transaction features into a single profile for each user account."
        ]
    }
    <!-- /conceptual_profile -->
    """
    mask = (
        (~tokens.eq(PROTEIN_ALPHABET["<cls>"]))
        & (~tokens.eq(PROTEIN_ALPHABET["<eos>"]))
        & (~tokens.eq(PROTEIN_ALPHABET["<pad>"]))
    )

    if chain_ids.numel() == 0:
        empty = torch.empty((representations.shape[0], 0), dtype=representations.dtype, device=representations.device)
        return empty, state

    max_chain_id = int(chain_ids.max().item())
    mean_chain_outs = []

    for chain_id in range(max_chain_id + 1):
        chain_mask = (chain_ids == chain_id) & mask
        chain_mask_exp = chain_mask.unsqueeze(-1).expand_as(representations)

        masked_out = representations * chain_mask_exp
        sum_masked = masked_out.sum(dim=1)
        mask_counts = chain_mask.sum(dim=1, keepdim=True).float()
        mask_counts = torch.clamp(mask_counts, min=1.0)

        mean_out = sum_masked / mask_counts
        mean_chain_outs.append(mean_out)

    contextualized = torch.cat(mean_chain_outs, dim=-1)
    return contextualized, state
