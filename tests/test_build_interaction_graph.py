from ageoa.molecular_docking.build_interaction_graph.atoms import weighted_interaction_edge_derivation


def test_weighted_interaction_edge_derivation_matches_vendored_two_pair_surface() -> None:
    result = weighted_interaction_edge_derivation(("L0", "L1"), ("R0", "R1"))
    assert result == [
        (("L0", "R1"), ("L1", "R0")),
        (("L1", "R1"), ("L0", "R0")),
    ]
