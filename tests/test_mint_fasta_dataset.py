from __future__ import annotations

from auditlib import runtime_probes


_atoms = runtime_probes.safe_import_module("ageoa.mint.fasta_dataset.atoms")
dataset_state_initialization = _atoms.dataset_state_initialization
token_budget_batch_planning = _atoms.token_budget_batch_planning


def test_dataset_state_initialization_matches_upstream_constructor_shape() -> None:
    state = dataset_state_initialization(["seq_a", "seq_b"], ["ACGT", "TT"])
    assert state == {
        "sequence_labels": ["seq_a", "seq_b"],
        "sequence_strs": ["ACGT", "TT"],
    }


def test_token_budget_batch_planning_matches_upstream_length_sorted_batching() -> None:
    dataset_state = {
        "sequence_labels": ["seq_a", "seq_b", "seq_c"],
        "sequence_strs": ["ACGT", "TT", "GGG"],
    }
    assert token_budget_batch_planning(dataset_state, toks_per_batch=8, extra_toks_per_seq=1) == [[1, 2], [0]]


def test_token_budget_batch_planning_uses_upstream_default_overhead() -> None:
    dataset_state = {
        "sequence_labels": ["seq_a", "seq_b"],
        "sequence_strs": ["ACGT", "TT"],
    }
    assert token_budget_batch_planning(dataset_state, toks_per_batch=8) == [[1, 0]]
