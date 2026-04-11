"""Selector-friendly registry coverage for Phase B family probe lanes."""

from __future__ import annotations

from auditlib import runtime_probes


def test_runtime_probe_registry_contains_wave1_neurokit2() -> None:
    for atom_name in (
        "ageoa.neurokit2.averageqrstemplate",
        "ageoa.neurokit2.zhao2018hrvanalysis",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave1_rust_robotics() -> None:
    assert "ageoa.rust_robotics.bicycle_kinematic.evaluateandinvertdynamics" in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave1_belief_propagation() -> None:
    for atom_name in (
        "ageoa.belief_propagation.loopy_bp.initialize_message_passing_state",
        "ageoa.belief_propagation.loopy_bp.run_loopy_message_passing_and_belief_query",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave1_particle_filter_and_pasqal() -> None:
    for atom_name in (
        "ageoa.particle_filters.basic.resample_and_belief_projection",
        "ageoa.pasqal.docking.quantum_mwis_solver",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave2_kalman_filter() -> None:
    for atom_name in (
        "ageoa.kalman_filters.filter_rs.initializekalmanstatemodel",
        "ageoa.kalman_filters.static_kf.updatewithmeasurement",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave2_quant_engine() -> None:
    for atom_name in (
        "ageoa.quant_engine.calculate_ofi",
        "ageoa.quant_engine.execute_passive",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave2_advancedvi_and_iqe() -> None:
    for atom_name in (
        "ageoa.advancedvi.core.evaluate_log_probability_density",
        "ageoa.institutional_quant_engine.hawkes_process.hawkesprocesssimulator",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave2_conjugate_priors_and_small_mcmc() -> None:
    for atom_name in (
        "ageoa.conjugate_priors.beta_binom.posterior_randmodel",
        "ageoa.mcmc_foundational.kthohr_mcmc.de.build_de_transition_kernel",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave3_biosppy_ecg_packet() -> None:
    for atom_name in (
        "ageoa.biosppy.ecg.bandpass_filter",
        "ageoa.biosppy.ecg.heart_rate_computation_median_smoothed",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave3_biosppy_signal_packet() -> None:
    for atom_name in (
        "ageoa.biosppy.ppg_detectors.detect_signal_onsets_elgendi2013",
        "ageoa.biosppy.online_filter_v2.filterstep",
        "ageoa.biosppy.ecg_zz2018_d12.assemblezz2018sqi",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave3_pronto_packet() -> None:
    for atom_name in (
        "ageoa.pronto.backlash_filter.initializebacklashfilterstate",
        "ageoa.pronto.foot_contact.mode_snapshot_readout",
        "ageoa.pronto.dynamic_stance_estimator_d12.stanceestimation",
        "ageoa.pronto.state_estimator.update_state_estimate",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave3_mcmc_foundational_packet() -> None:
    for atom_name in (
        "ageoa.mcmc_foundational.kthohr_mcmc.aees.metropolishastingstransitionkernel",
        "ageoa.mcmc_foundational.mini_mcmc.hmc.initializehmcstate",
        "ageoa.mcmc_foundational.advancedhmc.trajectory.nutstransitionkernel",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave3_quantfin_packet() -> None:
    for atom_name in (
        "ageoa.quantfin.montecarlo.quick_sim_anti",
        "ageoa.quantfin.rng_skip_d12.randomword32",
        "ageoa.quantfin.tdma_solver_d12.tdmasolver",
        "ageoa.quantfin.monte_carlo_anti_d12.runsimulationanti",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave3_molecular_docking_packet() -> None:
    for atom_name in (
        "ageoa.molecular_docking.greedy_mapping_d12.init_problem_context",
        "ageoa.molecular_docking.quantum_solver_d12.quantumsolverorchestrator",
        "ageoa.molecular_docking.mwis_sa.to_qubo",
        "ageoa.molecular_docking.minimize_bandwidth.build_threshold_search_space",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave3_institutional_quant_engine_packet() -> None:
    for atom_name in (
        "ageoa.institutional_quant_engine.market_making_avellaneda",
        "ageoa.institutional_quant_engine.avellaneda_stoikov.initializemarketmakerstate",
        "ageoa.institutional_quant_engine.almgren_chriss_v2.optimalexecutiontrajectory",
        "ageoa.institutional_quant_engine.fractional_diff.fractional_differentiator",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave3_foundation_packet() -> None:
    for atom_name in (
        "ageoa.e2e_ppg.kazemi_peak_detection",
        "ageoa.e2e_ppg.reconstruction.windowed_signal_reconstruction",
        "ageoa.mint.axial_attention.rowselfattention",
        "ageoa.alphafold.invariant_point_attention",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave3_hftbacktest_and_ingest_packet() -> None:
    for atom_name in (
        "ageoa.hftbacktest.update_glft_coefficients",
        "ageoa.mint.encoding_dist_mat.encodedistancematrix",
        "ageoa.mint.fasta_dataset.dataset_state_initialization",
        "ageoa.mint.incremental_attention.enable_incremental_state_configuration",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS


def test_runtime_probe_registry_contains_wave2_conjugate_priors_and_small_mcmc_extended() -> None:
    for atom_name in (
        "ageoa.conjugate_priors.beta_binom.posterior_randmodel",
        "ageoa.mcmc_foundational.kthohr_mcmc.de.build_de_transition_kernel",
        "ageoa.mcmc_foundational.kthohr_mcmc.mala.mala_proposal_adjustment",
    ):
        assert atom_name in runtime_probes.PROBE_PLANS
