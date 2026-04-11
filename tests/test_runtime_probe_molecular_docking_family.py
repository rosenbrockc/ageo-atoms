"""Focused runtime-probe coverage for molecular_docking family packets."""

from __future__ import annotations

from auditlib import runtime_probes


def _record(atom_name: str, module_import_path: str, wrapper_symbol: str) -> dict[str, object]:
    return {
        "atom_id": f"{atom_name}@ageoa/example.py:1",
        "atom_name": atom_name,
        "module_import_path": module_import_path,
        "module_path": "ageoa/example.py",
        "wrapper_symbol": wrapper_symbol,
        "wrapper_line": 1,
        "skeleton": False,
    }


def _assert_probe_passes(atom_name: str, module_import_path: str, wrapper_symbol: str) -> None:
    probe = runtime_probes.build_runtime_probe(_record(atom_name, module_import_path, wrapper_symbol))
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_molecular_docking_greedy_mapping_d12_family() -> None:
    for atom_name, module_path, wrapper_symbol in [
        (
            "ageoa.molecular_docking.greedy_mapping_d12.init_problem_context",
            "ageoa.molecular_docking.greedy_mapping_d12.atoms",
            "init_problem_context",
        ),
        (
            "ageoa.molecular_docking.greedy_mapping_d12.construct_mapping_state_via_greedy_expansion",
            "ageoa.molecular_docking.greedy_mapping_d12.atoms",
            "construct_mapping_state_via_greedy_expansion",
        ),
        (
            "ageoa.molecular_docking.greedy_mapping_d12.orchestrate_generation_and_validate",
            "ageoa.molecular_docking.greedy_mapping_d12.atoms",
            "orchestrate_generation_and_validate",
        ),
    ]:
        _assert_probe_passes(atom_name, module_path, wrapper_symbol)


def test_runtime_probe_passes_for_molecular_docking_build_interaction_graph_family() -> None:
    for atom_name, module_path, wrapper_symbol in [
        (
            "ageoa.molecular_docking.build_interaction_graph.pair_distance_compatibility_check",
            "ageoa.molecular_docking.build_interaction_graph.atoms",
            "pair_distance_compatibility_check",
        ),
        (
            "ageoa.molecular_docking.build_interaction_graph.weighted_interaction_edge_derivation",
            "ageoa.molecular_docking.build_interaction_graph.atoms",
            "weighted_interaction_edge_derivation",
        ),
        (
            "ageoa.molecular_docking.build_interaction_graph.networkx_weighted_graph_materialization",
            "ageoa.molecular_docking.build_interaction_graph.atoms",
            "networkx_weighted_graph_materialization",
        ),
    ]:
        _assert_probe_passes(atom_name, module_path, wrapper_symbol)


def test_runtime_probe_passes_for_molecular_docking_quantum_solver_d12_family() -> None:
    for atom_name, module_path, wrapper_symbol in [
        (
            "ageoa.molecular_docking.quantum_solver_d12.quantumsolverorchestrator",
            "ageoa.molecular_docking.quantum_solver_d12.atoms",
            "quantumsolverorchestrator",
        ),
        (
            "ageoa.molecular_docking.quantum_solver_d12.interactionboundscomputer",
            "ageoa.molecular_docking.quantum_solver_d12.atoms",
            "interactionboundscomputer",
        ),
        (
            "ageoa.molecular_docking.quantum_solver_d12.adiabaticpulseassembler",
            "ageoa.molecular_docking.quantum_solver_d12.atoms",
            "adiabaticpulseassembler",
        ),
        (
            "ageoa.molecular_docking.quantum_solver_d12.quantumcircuitsampler",
            "ageoa.molecular_docking.quantum_solver_d12.atoms",
            "quantumcircuitsampler",
        ),
        (
            "ageoa.molecular_docking.quantum_solver_d12.quantumsolutionextractor",
            "ageoa.molecular_docking.quantum_solver_d12.atoms",
            "quantumsolutionextractor",
        ),
    ]:
        _assert_probe_passes(atom_name, module_path, wrapper_symbol)


def test_runtime_probe_passes_for_molecular_docking_usage_equivalent_helpers() -> None:
    for atom_name, module_path, wrapper_symbol in [
        (
            "ageoa.molecular_docking.greedy_subgraph.greedy_maximum_subgraph",
            "ageoa.molecular_docking.greedy_subgraph",
            "greedy_maximum_subgraph",
        ),
        (
            "ageoa.molecular_docking.quantum_mwis_solver",
            "ageoa.molecular_docking.atoms",
            "quantum_mwis_solver",
        ),
        (
            "ageoa.molecular_docking.quantum_solver.quantumproblemdefinition",
            "ageoa.molecular_docking.quantum_solver.atoms",
            "quantumproblemdefinition",
        ),
        (
            "ageoa.molecular_docking.quantum_solver.adiabaticquantumsampler",
            "ageoa.molecular_docking.quantum_solver.atoms",
            "adiabaticquantumsampler",
        ),
        (
            "ageoa.molecular_docking.quantum_solver.solutionextraction",
            "ageoa.molecular_docking.quantum_solver.atoms",
            "solutionextraction",
        ),
        (
            "ageoa.molecular_docking.greedy_mapping.assemblestaticmappingcontext",
            "ageoa.molecular_docking.greedy_mapping.atoms",
            "assemblestaticmappingcontext",
        ),
        (
            "ageoa.molecular_docking.greedy_mapping.initializefrontierfromstartnode",
            "ageoa.molecular_docking.greedy_mapping.atoms",
            "initializefrontierfromstartnode",
        ),
        (
            "ageoa.molecular_docking.greedy_mapping.scoreandextendgreedycandidates",
            "ageoa.molecular_docking.greedy_mapping.atoms",
            "scoreandextendgreedycandidates",
        ),
        (
            "ageoa.molecular_docking.greedy_mapping.validatecurrentmapping",
            "ageoa.molecular_docking.greedy_mapping.atoms",
            "validatecurrentmapping",
        ),
        (
            "ageoa.molecular_docking.greedy_mapping.rungreedymappingpipeline",
            "ageoa.molecular_docking.greedy_mapping.atoms",
            "rungreedymappingpipeline",
        ),
        (
            "ageoa.molecular_docking.add_quantum_link.addquantumlink",
            "ageoa.molecular_docking.add_quantum_link",
            "addquantumlink",
        ),
        (
            "ageoa.molecular_docking.mwis_sa.to_qubo",
            "ageoa.molecular_docking.mwis_sa.atoms",
            "to_qubo",
        ),
    ]:
        _assert_probe_passes(atom_name, module_path, wrapper_symbol)


def test_runtime_probe_passes_for_molecular_docking_minimize_bandwidth_helper_groups() -> None:
    for atom_name, module_path, wrapper_symbol in [
        (
            "ageoa.molecular_docking.minimize_bandwidth.enumerate_threshold_based_permutations",
            "ageoa.molecular_docking.minimize_bandwidth.atoms",
            "enumerate_threshold_based_permutations",
        ),
        (
            "ageoa.molecular_docking.minimize_bandwidth.validate_square_matrix_shape",
            "ageoa.molecular_docking.minimize_bandwidth.atoms",
            "validate_square_matrix_shape",
        ),
        (
            "ageoa.molecular_docking.minimize_bandwidth.validate_symmetric_input",
            "ageoa.molecular_docking.minimize_bandwidth.atoms",
            "validate_symmetric_input",
        ),
        (
            "ageoa.molecular_docking.minimize_bandwidth.enforce_threshold_sparsity",
            "ageoa.molecular_docking.minimize_bandwidth.atoms",
            "enforce_threshold_sparsity",
        ),
        (
            "ageoa.molecular_docking.minimize_bandwidth.initialize_reduction_state",
            "ageoa.molecular_docking.minimize_bandwidth.atoms",
            "initialize_reduction_state",
        ),
        (
            "ageoa.molecular_docking.minimize_bandwidth.extract_final_permutation",
            "ageoa.molecular_docking.minimize_bandwidth.atoms",
            "extract_final_permutation",
        ),
        (
            "ageoa.molecular_docking.minimize_bandwidth.select_minimum_bandwidth_permutation",
            "ageoa.molecular_docking.minimize_bandwidth.atoms",
            "select_minimum_bandwidth_permutation",
        ),
        (
            "ageoa.molecular_docking.minimize_bandwidth.propose_greedy_permutation_step",
            "ageoa.molecular_docking.minimize_bandwidth.atoms",
            "propose_greedy_permutation_step",
        ),
        (
            "ageoa.molecular_docking.minimize_bandwidth.update_state_with_improvement_criterion",
            "ageoa.molecular_docking.minimize_bandwidth.atoms",
            "update_state_with_improvement_criterion",
        ),
        (
            "ageoa.molecular_docking.minimize_bandwidth.compute_absolute_weighted_index_distances",
            "ageoa.molecular_docking.minimize_bandwidth.atoms",
            "compute_absolute_weighted_index_distances",
        ),
        (
            "ageoa.molecular_docking.minimize_bandwidth.aggregate_maximum_distance_as_bandwidth",
            "ageoa.molecular_docking.minimize_bandwidth.atoms",
            "aggregate_maximum_distance_as_bandwidth",
        ),
        (
            "ageoa.molecular_docking.minimize_bandwidth.build_sparse_graph_view",
            "ageoa.molecular_docking.minimize_bandwidth.atoms",
            "build_sparse_graph_view",
        ),
        (
            "ageoa.molecular_docking.minimize_bandwidth.compute_symmetric_bandwidth_reducing_order",
            "ageoa.molecular_docking.minimize_bandwidth.atoms",
            "compute_symmetric_bandwidth_reducing_order",
        ),
        (
            "ageoa.molecular_docking.minimize_bandwidth.build_threshold_search_space",
            "ageoa.molecular_docking.minimize_bandwidth.atoms",
            "build_threshold_search_space",
        ),
    ]:
        _assert_probe_passes(atom_name, module_path, wrapper_symbol)
