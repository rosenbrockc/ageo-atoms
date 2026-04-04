from __future__ import annotations

from types import SimpleNamespace

from auditlib import runtime_probes


def _record(atom_name: str, module_import_path: str = "ageoa.algorithms.search", wrapper_symbol: str = "binary_search") -> dict:
    return {
        "atom_id": f"{atom_name}@ageoa/example.py:1",
        "atom_name": atom_name,
        "module_import_path": module_import_path,
        "module_path": "ageoa/example.py",
        "wrapper_symbol": wrapper_symbol,
        "wrapper_line": 1,
        "skeleton": False,
    }


def test_runtime_probe_passes_for_safe_real_atom() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.algorithms.search.binary_search", "ageoa.algorithms.search", "binary_search")
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_skips_unsupported_atom() -> None:
    probe = runtime_probes.build_runtime_probe(_record("ageoa.example.unsupported", "ageoa.example", "atom"))
    assert probe["status"] == "not_applicable"
    assert probe["skip_reason"] == "unsupported_scope"
    assert probe["findings"] == ["RUNTIME_PROBE_SKIPPED"]


def test_runtime_probe_passes_for_numpy_fft() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.numpy.fft.fft", "ageoa.numpy.fft", "fft")
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_sparse_graph_laplacian() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.scipy.sparse_graph.graph_laplacian", "ageoa.scipy.sparse_graph", "graph_laplacian")
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_sklearn_image_grid_to_graph() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.sklearn.images.grid_to_graph", "ageoa.sklearn.images.atoms", "grid_to_graph")
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_numpy_fft_v2() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.numpy.fft_v2.forwardmultidimensionalfft",
            "ageoa.numpy.fft_v2.atoms",
            "forwardmultidimensionalfft",
        )
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_marks_sorting_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.algorithms.sorting.merge_sort", "ageoa.algorithms.sorting", "merge_sort")
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_passes_for_advancedvi_log_density() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.advancedvi.core.evaluate_log_probability_density",
            "ageoa.advancedvi.core",
            "evaluate_log_probability_density",
        )
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_hawkes_process_simulator() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.institutional_quant_engine.hawkes_process.hawkesprocesssimulator",
            "ageoa.institutional_quant_engine.hawkes_process",
            "hawkesprocesssimulator",
        )
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_advancedhmc_integrator_family() -> None:
    tempering_probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.mcmc_foundational.advancedhmc.integrator.temperingfactorcomputation",
            "ageoa.mcmc_foundational.advancedhmc.integrator.atoms",
            "temperingfactorcomputation",
        )
    )
    transition_probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.mcmc_foundational.advancedhmc.integrator.hamiltonianphasepointtransition",
            "ageoa.mcmc_foundational.advancedhmc.integrator.atoms",
            "hamiltonianphasepointtransition",
        )
    )

    assert tempering_probe["status"] == "pass"
    assert transition_probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in tempering_probe["findings"]
    assert "RUNTIME_PROBE_PASS" in transition_probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in tempering_probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in transition_probe["findings"]


def test_runtime_probe_passes_for_skyfield_family() -> None:
    angle_probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.skyfield.calculate_vector_angle",
            "ageoa.skyfield.atoms",
            "calculate_vector_angle",
        )
    )
    spherical_probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.skyfield.compute_spherical_coordinate_rates",
            "ageoa.skyfield.atoms",
            "compute_spherical_coordinate_rates",
        )
    )

    assert angle_probe["status"] == "pass"
    assert spherical_probe["status"] == "pass"
    assert angle_probe["parity_used"] is True
    assert spherical_probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in angle_probe["findings"]
    assert "RUNTIME_PROBE_PASS" in spherical_probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in angle_probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in spherical_probe["findings"]


def test_runtime_probe_passes_for_pulsar_family() -> None:
    for atom_name, symbol in [
        ("ageoa.pulsar.pipeline.delay_from_DM", "delay_from_DM"),
        ("ageoa.pulsar.pipeline.de_disperse", "de_disperse"),
        ("ageoa.pulsar.pipeline.fold_signal", "fold_signal"),
        ("ageoa.pulsar.pipeline.SNR", "SNR"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, "ageoa.pulsar.pipeline", symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_e2e_ppg_family() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.e2e_ppg.kazemi_peak_detection", "ageoa.e2e_ppg.atoms", "kazemi_peak_detection"),
        ("ageoa.e2e_ppg.ppg_reconstruction", "ageoa.e2e_ppg.atoms", "ppg_reconstruction"),
        ("ageoa.e2e_ppg.ppg_sqa", "ageoa.e2e_ppg.atoms", "ppg_sqa"),
        (
            "ageoa.e2e_ppg.template_matching.templatefeaturecomputation",
            "ageoa.e2e_ppg.template_matching",
            "templatefeaturecomputation",
        ),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_datadriven_discover_equations() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.datadriven.discover_equations",
            "ageoa.datadriven.atoms",
            "discover_equations",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_heston_path_sampler() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.institutional_quant_engine.heston_model.hestonpathsampler",
            "ageoa.institutional_quant_engine.heston_model",
            "hestonpathsampler",
        )
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_particle_filter_resample_projection() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.particle_filters.basic.resample_and_belief_projection",
            "ageoa.particle_filters.basic.atoms",
            "resample_and_belief_projection",
        )
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_pasqal_sub_graph_embedder() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.pasqal.docking.sub_graph_embedder",
            "ageoa.pasqal.docking",
            "sub_graph_embedder",
        )
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_pasqal_quantum_mwis_solver() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.pasqal.docking.quantum_mwis_solver",
            "ageoa.pasqal.docking",
            "quantum_mwis_solver",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_marks_pronto_leg_odometer_and_mode_readouts_as_usage_equivalent() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.pronto.foot_contact.mode_snapshot_readout", "ageoa.pronto.foot_contact.atoms", "mode_snapshot_readout"),
        ("ageoa.pronto.leg_odometer.velocitystatereadout", "ageoa.pronto.leg_odometer.atoms", "velocitystatereadout"),
        ("ageoa.pronto.leg_odometer.posequeryaccessors", "ageoa.pronto.leg_odometer.atoms", "posequeryaccessors"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_scipy_optimize_v2_shgo() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.scipy.optimize_v2.shgoglobaloptimization",
            "ageoa.scipy.optimize_v2.atoms",
            "shgoglobaloptimization",
        )
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_scipy_optimize_v2_differential_evolution() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.scipy.optimize_v2.differentialevolutionoptimization",
            "ageoa.scipy.optimize_v2.atoms",
            "differentialevolutionoptimization",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_marks_scipy_curve_fit_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.scipy.optimize.curve_fit",
            "ageoa.scipy.optimize",
            "curve_fit",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_char_func_option_integrand_helper() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.quantfin.char_func_option_d12.f",
            "ageoa.quantfin.char_func_option_d12.atoms",
            "f",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_char_func_option_family() -> None:
    for atom_name, symbol in [
        ("ageoa.quantfin.char_func_option_d12.cf", "cf"),
        ("ageoa.quantfin.char_func_option_d12.charfuncoption", "charfuncoption"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(
                atom_name,
                "ageoa.quantfin.char_func_option_d12.atoms",
                symbol,
            )
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_local_vol_d12_family() -> None:
    for atom_name, symbol in [
        ("ageoa.quantfin.local_vol_d12.var", "var"),
        ("ageoa.quantfin.local_vol_d12.localvol", "localvol"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(
                atom_name,
                "ageoa.quantfin.local_vol_d12.atoms",
                symbol,
            )
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_biosppy_ppg_detectors() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.ppg_detectors.detect_signal_onsets_elgendi2013", "detect_signal_onsets_elgendi2013"),
        ("ageoa.biosppy.ppg_detectors.detectonsetevents", "detectonsetevents"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.biosppy.ppg_detectors", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_biosppy_abp() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.biosppy.abp.audio_onset_detection",
            "ageoa.biosppy.abp",
            "audio_onset_detection",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_biosppy_ecg_module_wrappers() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.ecg.bandpass_filter", "bandpass_filter"),
        ("ageoa.biosppy.ecg.r_peak_detection", "r_peak_detection"),
        ("ageoa.biosppy.ecg.peak_correction", "peak_correction"),
        ("ageoa.biosppy.ecg.template_extraction", "template_extraction"),
        ("ageoa.biosppy.ecg.heart_rate_computation", "heart_rate_computation"),
        ("ageoa.biosppy.ecg.ssf_segmenter", "ssf_segmenter"),
        ("ageoa.biosppy.ecg.christov_segmenter", "christov_segmenter"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.biosppy.ecg", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_biosppy_threshold_based_asi() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.biosppy.ecg_detectors.thresholdbasedsignalsegmentation",
            "ageoa.biosppy.ecg_detectors",
            "thresholdbasedsignalsegmentation",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_pronto_backlash_filter_family() -> None:
    for atom_name, symbol in [
        ("ageoa.pronto.backlash_filter.initializebacklashfilterstate", "initializebacklashfilterstate"),
        ("ageoa.pronto.backlash_filter.updatealphaparameter", "updatealphaparameter"),
        ("ageoa.pronto.backlash_filter.updatecrossingtimemaximum", "updatecrossingtimemaximum"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.pronto.backlash_filter.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_biosppy_eda() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.eda.gamboa_segmenter", "gamboa_segmenter"),
        ("ageoa.biosppy.eda.eda_feature_extraction", "eda_feature_extraction"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.biosppy.eda", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_biosppy_emg_detectors() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.emg_detectors.detect_onsets_with_rest_aware_thresholds", "detect_onsets_with_rest_aware_thresholds"),
        ("ageoa.biosppy.emg_detectors.bonato_onset_detection", "bonato_onset_detection"),
        ("ageoa.biosppy.emg_detectors.threshold_based_onset_detection", "threshold_based_onset_detection"),
        ("ageoa.biosppy.emg_detectors.solnik_onset_detect", "solnik_onset_detect"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.biosppy.emg_detectors", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_biosppy_pcg() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.pcg.shannon_energy", "shannon_energy"),
        ("ageoa.biosppy.pcg.pcg_segmentation", "pcg_segmentation"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.biosppy.pcg", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_biosppy_online_filter_variants() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.biosppy.online_filter.filterstateinit", "ageoa.biosppy.online_filter.atoms", "filterstateinit"),
        ("ageoa.biosppy.online_filter.filterstep", "ageoa.biosppy.online_filter.atoms", "filterstep"),
        ("ageoa.biosppy.online_filter_codex.filterstateinit", "ageoa.biosppy.online_filter_codex.atoms", "filterstateinit"),
        ("ageoa.biosppy.online_filter_codex.filterstep", "ageoa.biosppy.online_filter_codex.atoms", "filterstep"),
        ("ageoa.biosppy.online_filter_v2.filterstateinit", "ageoa.biosppy.online_filter_v2.atoms", "filterstateinit"),
        ("ageoa.biosppy.online_filter_v2.filterstep", "ageoa.biosppy.online_filter_v2.atoms", "filterstep"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_biosppy_svm_proc_family() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.svm_proc.get_auth_rates", "get_auth_rates"),
        ("ageoa.biosppy.svm_proc.get_id_rates", "get_id_rates"),
        ("ageoa.biosppy.svm_proc.get_subject_results", "get_subject_results"),
        ("ageoa.biosppy.svm_proc.assess_classification", "assess_classification"),
        ("ageoa.biosppy.svm_proc.assess_runs", "assess_runs"),
        ("ageoa.biosppy.svm_proc.combination", "combination"),
        ("ageoa.biosppy.svm_proc.majority_rule", "majority_rule"),
        ("ageoa.biosppy.svm_proc.cross_validation", "cross_validation"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.biosppy.svm_proc.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_kalman_filter_families() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.kalman_filters.filter_rs.initializekalmanstatemodel", "ageoa.kalman_filters.filter_rs.atoms", "initializekalmanstatemodel"),
        ("ageoa.kalman_filters.filter_rs.predictlatentstateandcovariance", "ageoa.kalman_filters.filter_rs.atoms", "predictlatentstateandcovariance"),
        ("ageoa.kalman_filters.filter_rs.predictlatentstatesteadystate", "ageoa.kalman_filters.filter_rs.atoms", "predictlatentstatesteadystate"),
        ("ageoa.kalman_filters.filter_rs.evaluatemeasurementoracle", "ageoa.kalman_filters.filter_rs.atoms", "evaluatemeasurementoracle"),
        ("ageoa.kalman_filters.filter_rs.updateposteriorstateandcovariance", "ageoa.kalman_filters.filter_rs.atoms", "updateposteriorstateandcovariance"),
        ("ageoa.kalman_filters.filter_rs.updateposteriorstatesteadystate", "ageoa.kalman_filters.filter_rs.atoms", "updateposteriorstatesteadystate"),
        ("ageoa.kalman_filters.static_kf.initializelineargaussianstatemodel", "ageoa.kalman_filters.static_kf.atoms", "initializelineargaussianstatemodel"),
        ("ageoa.kalman_filters.static_kf.predictlatentstate", "ageoa.kalman_filters.static_kf.atoms", "predictlatentstate"),
        ("ageoa.kalman_filters.static_kf.updatewithmeasurement", "ageoa.kalman_filters.static_kf.atoms", "updatewithmeasurement"),
        ("ageoa.kalman_filters.static_kf.exposelatentmean", "ageoa.kalman_filters.static_kf.atoms", "exposelatentmean"),
        ("ageoa.kalman_filters.static_kf.exposecovariance", "ageoa.kalman_filters.static_kf.atoms", "exposecovariance"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_pronto_blip_filter_family() -> None:
    for atom_name, symbol in [
        ("ageoa.pronto.blip_filter.bandpass_filter", "bandpass_filter"),
        ("ageoa.pronto.blip_filter.r_peak_detection", "r_peak_detection"),
        ("ageoa.pronto.blip_filter.peak_correction", "peak_correction"),
        ("ageoa.pronto.blip_filter.template_extraction", "template_extraction"),
        ("ageoa.pronto.blip_filter.heart_rate_computation", "heart_rate_computation"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.pronto.blip_filter.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_mcmc_foundational_builder_family() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.mcmc_foundational.kthohr_mcmc.aees.metropolishastingstransitionkernel", "ageoa.mcmc_foundational.kthohr_mcmc.aees.atoms", "metropolishastingstransitionkernel"),
        ("ageoa.mcmc_foundational.kthohr_mcmc.aees.targetlogkerneloracle", "ageoa.mcmc_foundational.kthohr_mcmc.aees.atoms", "targetlogkerneloracle"),
        ("ageoa.mcmc_foundational.kthohr_mcmc.hmc.buildhmckernelfromlogdensityoracle", "ageoa.mcmc_foundational.kthohr_mcmc.hmc", "buildhmckernelfromlogdensityoracle"),
        ("ageoa.mcmc_foundational.kthohr_mcmc.rmhmc.buildrmhmctransitionkernel", "ageoa.mcmc_foundational.kthohr_mcmc.rmhmc", "buildrmhmctransitionkernel"),
        ("ageoa.mcmc_foundational.kthohr_mcmc.rwmh.constructrandomwalkmetropoliskernel", "ageoa.mcmc_foundational.kthohr_mcmc.rwmh", "constructrandomwalkmetropoliskernel"),
        ("ageoa.mcmc_foundational.mini_mcmc.hmc_llm.initializehmckernelstate", "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.atoms", "initializehmckernelstate"),
        ("ageoa.mcmc_foundational.mini_mcmc.hmc_llm.initializesamplerrng", "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.atoms", "initializesamplerrng"),
        ("ageoa.mcmc_foundational.mini_mcmc.hmc_llm.hamiltoniantransitionkernel", "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.atoms", "hamiltoniantransitionkernel"),
        ("ageoa.mcmc_foundational.mini_mcmc.nuts_llm.initializenutsstate", "ageoa.mcmc_foundational.mini_mcmc.nuts_llm.atoms", "initializenutsstate"),
        ("ageoa.mcmc_foundational.mini_mcmc.nuts_llm.runnutstransitions", "ageoa.mcmc_foundational.mini_mcmc.nuts_llm.atoms", "runnutstransitions"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_mini_mcmc_kernel_family() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.mcmc_foundational.mini_mcmc.hmc.initializehmcstate", "ageoa.mcmc_foundational.mini_mcmc.hmc.atoms", "initializehmcstate"),
        ("ageoa.mcmc_foundational.mini_mcmc.hmc.leapfrogproposalkernel", "ageoa.mcmc_foundational.mini_mcmc.hmc.atoms", "leapfrogproposalkernel"),
        ("ageoa.mcmc_foundational.mini_mcmc.hmc.metropolishmctransition", "ageoa.mcmc_foundational.mini_mcmc.hmc.atoms", "metropolishmctransition"),
        ("ageoa.mcmc_foundational.mini_mcmc.hmc.runsamplingloop", "ageoa.mcmc_foundational.mini_mcmc.hmc.atoms", "runsamplingloop"),
        ("ageoa.mcmc_foundational.mini_mcmc.nuts.nuts_recursive_tree_build", "ageoa.mcmc_foundational.mini_mcmc.nuts", "nuts_recursive_tree_build"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_pronto_state_readout_family() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.pronto.foot_contact.foot_sensing_state_update", "ageoa.pronto.foot_contact.atoms", "foot_sensing_state_update"),
        ("ageoa.pronto.inverse_schmitt.inverse_schmitt_trigger_transform", "ageoa.pronto.inverse_schmitt", "inverse_schmitt_trigger_transform"),
        ("ageoa.pronto.torque_adjustment.torqueadjustmentidentitystage", "ageoa.pronto.torque_adjustment", "torqueadjustmentidentitystage"),
        ("ageoa.pronto.yaw_lock.readrobotstandingstatus", "ageoa.pronto.yaw_lock.atoms", "readrobotstandingstatus"),
        ("ageoa.pronto.yaw_lock.readinitialjointangles", "ageoa.pronto.yaw_lock.atoms", "readinitialjointangles"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_pronto_dynamic_stance_d12_family() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.pronto.dynamic_stance_estimator.initializefilter", "ageoa.pronto.dynamic_stance_estimator.atoms", "initializefilter"),
        ("ageoa.pronto.dynamic_stance_estimator.predictstep", "ageoa.pronto.dynamic_stance_estimator.atoms", "predictstep"),
        ("ageoa.pronto.dynamic_stance_estimator.updatestep", "ageoa.pronto.dynamic_stance_estimator.atoms", "updatestep"),
        ("ageoa.pronto.dynamic_stance_estimator.querystance", "ageoa.pronto.dynamic_stance_estimator.atoms", "querystance"),
        ("ageoa.pronto.dynamic_stance_estimator_d12.stancestateinit", "ageoa.pronto.dynamic_stance_estimator_d12.atoms", "stancestateinit"),
        ("ageoa.pronto.dynamic_stance_estimator_d12.stanceestimation", "ageoa.pronto.dynamic_stance_estimator_d12.atoms", "stanceestimation"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_conjugate_and_small_mcmc_family() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.conjugate_priors.beta_binom.posterior_randmodel", "ageoa.conjugate_priors.beta_binom.atoms", "posterior_randmodel"),
        ("ageoa.conjugate_priors.beta_binom.posterior_randmodel_weighted", "ageoa.conjugate_priors.beta_binom.atoms", "posterior_randmodel_weighted"),
        ("ageoa.conjugate_priors.normal.normal_gamma_posterior_update", "ageoa.conjugate_priors.normal", "normal_gamma_posterior_update"),
        ("ageoa.mcmc_foundational.kthohr_mcmc.de.build_de_transition_kernel", "ageoa.mcmc_foundational.kthohr_mcmc.de", "build_de_transition_kernel"),
        ("ageoa.mcmc_foundational.kthohr_mcmc.mala.mala_proposal_adjustment", "ageoa.mcmc_foundational.kthohr_mcmc.mala", "mala_proposal_adjustment"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_institutional_quant_engine_stateful_helpers() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.institutional_quant_engine.kalman_filter.kalmanfilterinit", "ageoa.institutional_quant_engine.kalman_filter.atoms", "kalmanfilterinit"),
        ("ageoa.institutional_quant_engine.kalman_filter.kalmanmeasurementupdate", "ageoa.institutional_quant_engine.kalman_filter.atoms", "kalmanmeasurementupdate"),
        ("ageoa.institutional_quant_engine.queue_estimator.initializeorderstate", "ageoa.institutional_quant_engine.queue_estimator.atoms", "initializeorderstate"),
        ("ageoa.institutional_quant_engine.queue_estimator.updatequeueontrade", "ageoa.institutional_quant_engine.queue_estimator.atoms", "updatequeueontrade"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_institutional_quant_engine_script_wrappers() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.institutional_quant_engine.copula_dependence.simulate_copula_dependence", "ageoa.institutional_quant_engine.copula_dependence", "simulate_copula_dependence"),
        ("ageoa.institutional_quant_engine.dynamic_hedge.kalman_hedge_ratio", "ageoa.institutional_quant_engine.dynamic_hedge", "kalman_hedge_ratio"),
        ("ageoa.institutional_quant_engine.evt_model.fit_gpd_tail", "ageoa.institutional_quant_engine.evt_model", "fit_gpd_tail"),
        ("ageoa.institutional_quant_engine.supply_chain.propagate_supply_shock", "ageoa.institutional_quant_engine.supply_chain", "propagate_supply_shock"),
        ("ageoa.institutional_quant_engine.triangular_arbitrage.detect_triangular_arbitrage", "ageoa.institutional_quant_engine.triangular_arbitrage", "detect_triangular_arbitrage"),
        ("ageoa.institutional_quant_engine.wash_trade.detect_wash_trade_rings", "ageoa.institutional_quant_engine.wash_trade", "detect_wash_trade_rings"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_rust_robotics_bicycle_dynamics() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.rust_robotics.bicycle_kinematic.evaluateandinvertdynamics",
            "ageoa.rust_robotics.bicycle_kinematic.atoms",
            "evaluateandinvertdynamics",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_pronto_state_estimator_update() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.pronto.state_estimator.update_state_estimate",
            "ageoa.pronto.state_estimator",
            "update_state_estimate",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_quantfin_quick_sim_anti() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.quantfin.montecarlo.quick_sim_anti",
            "ageoa.quantfin.montecarlo",
            "quick_sim_anti",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_quantfin_d12_helpers() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.quantfin.local_vol_d12.allfort", "ageoa.quantfin.local_vol_d12.atoms", "allfort"),
        ("ageoa.quantfin.rng_skip_d12.addmod64", "ageoa.quantfin.rng_skip_d12.atoms", "addmod64"),
        ("ageoa.quantfin.rng_skip_d12.mulmod64", "ageoa.quantfin.rng_skip_d12.atoms", "mulmod64"),
        ("ageoa.quantfin.rng_skip_d12.powmod64", "ageoa.quantfin.rng_skip_d12.atoms", "powmod64"),
        ("ageoa.quantfin.rng_skip_d12.skip", "ageoa.quantfin.rng_skip_d12.atoms", "skip"),
        ("ageoa.quantfin.rng_skip_d12.split", "ageoa.quantfin.rng_skip_d12.atoms", "split"),
        ("ageoa.quantfin.rng_skip_d12.next", "ageoa.quantfin.rng_skip_d12.atoms", "next"),
        ("ageoa.quantfin.rng_skip_d12.randomdouble", "ageoa.quantfin.rng_skip_d12.atoms", "randomdouble"),
        ("ageoa.quantfin.rng_skip_d12.randomint", "ageoa.quantfin.rng_skip_d12.atoms", "randomint"),
        ("ageoa.quantfin.rng_skip_d12.randomint64", "ageoa.quantfin.rng_skip_d12.atoms", "randomint64"),
        ("ageoa.quantfin.rng_skip_d12.randomword64", "ageoa.quantfin.rng_skip_d12.atoms", "randomword64"),
        ("ageoa.quantfin.tdma_solver_d12.tdmasolver", "ageoa.quantfin.tdma_solver_d12.atoms", "tdmasolver"),
        ("ageoa.quantfin.tdma_solver_d12.cotraversevec", "ageoa.quantfin.tdma_solver_d12.atoms", "cotraversevec"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_additional_quantfin_helpers() -> None:
    for atom_name, module_path, symbol, requires_negative in [
        ("ageoa.quantfin.monte_carlo_anti_d12.avg", "ageoa.quantfin.monte_carlo_anti_d12.atoms", "avg", True),
        ("ageoa.quantfin.monte_carlo_anti_d12.maxstep", "ageoa.quantfin.monte_carlo_anti_d12.atoms", "maxstep", True),
        ("ageoa.quantfin.monte_carlo_anti_d12.insertcf", "ageoa.quantfin.monte_carlo_anti_d12.atoms", "insertcf", True),
        ("ageoa.quantfin.monte_carlo_anti_d12.insertcflist", "ageoa.quantfin.monte_carlo_anti_d12.atoms", "insertcflist", True),
        ("ageoa.quantfin.monte_carlo_anti_d12.runmc", "ageoa.quantfin.monte_carlo_anti_d12.atoms", "runmc", True),
        ("ageoa.quantfin.monte_carlo_anti_d12.runsimulation", "ageoa.quantfin.monte_carlo_anti_d12.atoms", "runsimulation", True),
        ("ageoa.quantfin.monte_carlo_anti_d12.runsimulationanti", "ageoa.quantfin.monte_carlo_anti_d12.atoms", "runsimulationanti", True),
        ("ageoa.quantfin.monte_carlo_anti_d12.quicksim", "ageoa.quantfin.monte_carlo_anti_d12.atoms", "quicksim", True),
        ("ageoa.quantfin.monte_carlo_anti_d12.quicksimanti", "ageoa.quantfin.monte_carlo_anti_d12.atoms", "quicksimanti", True),
        ("ageoa.quantfin.monte_carlo_anti_d12.simulatestate", "ageoa.quantfin.monte_carlo_anti_d12.atoms", "simulatestate", True),
        ("ageoa.quantfin.monte_carlo_anti_d12.runsim", "ageoa.quantfin.monte_carlo_anti_d12.atoms", "runsim", True),
        ("ageoa.quantfin.monte_carlo_anti_d12.evolve", "ageoa.quantfin.monte_carlo_anti_d12.atoms", "evolve", True),
        ("ageoa.quantfin.rng_skip_d12.randomword32", "ageoa.quantfin.rng_skip_d12.atoms", "randomword32", True),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        if requires_negative:
            assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_signal_and_reconstruction_helpers() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.pulsar_folding.dm_can_brute_force", "ageoa.pulsar_folding.atoms", "dm_can_brute_force"),
        ("ageoa.pulsar_folding.spline_bandpass_correction", "ageoa.pulsar_folding.atoms", "spline_bandpass_correction"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_fractional_diff_and_encoding_helpers() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.institutional_quant_engine.fractional_diff.fractional_differentiator", "ageoa.institutional_quant_engine.fractional_diff", "fractional_differentiator"),
        ("ageoa.mint.encoding_dist_mat.encodedistancematrix", "ageoa.mint.encoding_dist_mat", "encodedistancematrix"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_mint_fasta_dataset_family() -> None:
    for atom_name, symbol in [
        ("ageoa.mint.fasta_dataset.dataset_state_initialization", "dataset_state_initialization"),
        ("ageoa.mint.fasta_dataset.dataset_length_query", "dataset_length_query"),
        ("ageoa.mint.fasta_dataset.dataset_item_retrieval", "dataset_item_retrieval"),
        ("ageoa.mint.fasta_dataset.token_budget_batch_planning", "token_budget_batch_planning"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.mint.fasta_dataset.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_greedy_mapping_d12_family() -> None:
    for atom_name, symbol in [
        ("ageoa.molecular_docking.greedy_mapping_d12.init_problem_context", "init_problem_context"),
        ("ageoa.molecular_docking.greedy_mapping_d12.construct_mapping_state_via_greedy_expansion", "construct_mapping_state_via_greedy_expansion"),
        ("ageoa.molecular_docking.greedy_mapping_d12.orchestrate_generation_and_validate", "orchestrate_generation_and_validate"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.molecular_docking.greedy_mapping_d12.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_jax_advi_posterior_draws() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.jax_advi.optimize_advi.posteriordrawsampling",
            "ageoa.jax_advi.optimize_advi.atoms",
            "posteriordrawsampling",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_jax_advi_mean_field_fit() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.jax_advi.optimize_advi.meanfieldvariationalfit",
            "ageoa.jax_advi.optimize_advi.atoms",
            "meanfieldvariationalfit",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_advancedvi_gradient_oracle() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.advancedvi.core.gradient_oracle_evaluation",
            "ageoa.advancedvi.core",
            "gradient_oracle_evaluation",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_particle_filter_helpers() -> None:
    for atom_name, symbol in [
        ("ageoa.particle_filters.basic.filter_step_preparation_and_dispatch", "filter_step_preparation_and_dispatch"),
        ("ageoa.particle_filters.basic.particle_propagation_kernel", "particle_propagation_kernel"),
        ("ageoa.particle_filters.basic.likelihood_reweight_kernel", "likelihood_reweight_kernel"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.particle_filters.basic.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_belief_propagation_loopy_bp() -> None:
    for atom_name, symbol in [
        ("ageoa.belief_propagation.loopy_bp.initialize_message_passing_state", "initialize_message_passing_state"),
        ("ageoa.belief_propagation.loopy_bp.run_loopy_message_passing_and_belief_query", "run_loopy_message_passing_and_belief_query"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.belief_propagation.loopy_bp.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_kthohr_mcmc_dispatch() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.mcmc_foundational.kthohr_mcmc.mcmc_algos.dispatch_mcmc_algorithm",
            "ageoa.mcmc_foundational.kthohr_mcmc.mcmc_algos",
            "dispatch_mcmc_algorithm",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_marks_hftbacktest_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.hftbacktest.update_glft_coefficients", "ageoa.hftbacktest.atoms", "update_glft_coefficients")
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_marks_institutional_quant_engine_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.institutional_quant_engine.market_making_avellaneda",
            "ageoa.institutional_quant_engine.atoms",
            "market_making_avellaneda",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_passes_for_quant_engine_execution_helpers() -> None:
    for atom_name, symbol in [
        ("ageoa.quant_engine.calculate_ofi", "calculate_ofi"),
        ("ageoa.quant_engine.execute_vwap", "execute_vwap"),
        ("ageoa.quant_engine.execute_pov", "execute_pov"),
        ("ageoa.quant_engine.execute_passive", "execute_passive"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.quant_engine.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_institutional_quant_engine_portfolio_wrappers() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.institutional_quant_engine.avellaneda_stoikov.initializemarketmakerstate", "ageoa.institutional_quant_engine.avellaneda_stoikov.atoms", "initializemarketmakerstate"),
        ("ageoa.institutional_quant_engine.avellaneda_stoikov.computeinventoryadjustedquotes", "ageoa.institutional_quant_engine.avellaneda_stoikov.atoms", "computeinventoryadjustedquotes"),
        ("ageoa.institutional_quant_engine.hierarchical_risk_parity.compute_hrp_weights", "ageoa.institutional_quant_engine.hierarchical_risk_parity", "compute_hrp_weights"),
        ("ageoa.institutional_quant_engine.hierarchical_risk_parity.hrppipelinerun", "ageoa.institutional_quant_engine.hierarchical_risk_parity", "hrppipelinerun"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_almgren_chriss_v2_family() -> None:
    for atom_name, symbol in [
        ("ageoa.institutional_quant_engine.almgren_chriss_v2.riskaversioninit", "riskaversioninit"),
        ("ageoa.institutional_quant_engine.almgren_chriss_v2.optimalexecutiontrajectory", "optimalexecutiontrajectory"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.institutional_quant_engine.almgren_chriss_v2.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_almgren_chriss_generated_wrapper() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.institutional_quant_engine.almgren_chriss.computeoptimaltrajectory",
            "ageoa.institutional_quant_engine.almgren_chriss",
            "computeoptimaltrajectory",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_order_flow_and_pin_generated_wrappers() -> None:
    for atom_name, module_path, symbol in [
        (
            "ageoa.institutional_quant_engine.order_flow_imbalance.orderflowimbalanceevaluation",
            "ageoa.institutional_quant_engine.order_flow_imbalance",
            "orderflowimbalanceevaluation",
        ),
        (
            "ageoa.institutional_quant_engine.pin_model.pinlikelihoodevaluation",
            "ageoa.institutional_quant_engine.pin_model",
            "pinlikelihoodevaluation",
        ),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, module_path, symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_avellaneda_stoikov_d12_family() -> None:
    for atom_name, symbol in [
        ("ageoa.institutional_quant_engine.avellaneda_stoikov_d12.marketmakerstateinit", "marketmakerstateinit"),
        ("ageoa.institutional_quant_engine.avellaneda_stoikov_d12.optimalquotecalculation", "optimalquotecalculation"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.institutional_quant_engine.avellaneda_stoikov_d12.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_scipy_spatial_v2_family() -> None:
    for atom_name, symbol in [
        ("ageoa.scipy.spatial_v2.voronoitessellation", "voronoitessellation"),
        ("ageoa.scipy.spatial_v2.delaunaytriangulation", "delaunaytriangulation"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.scipy.spatial_v2.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_scipy_interpolate_v2_family() -> None:
    for atom_name, symbol in [
        ("ageoa.scipy.interpolate_v2.cubicsplinefit", "cubicsplinefit"),
        ("ageoa.scipy.interpolate_v2.rbfinterpolatorfit", "rbfinterpolatorfit"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.scipy.interpolate_v2.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_scipy_sparse_graph_v2_family() -> None:
    for atom_name, symbol in [
        ("ageoa.scipy.sparse_graph_v2.singlesourceshortestpath", "singlesourceshortestpath"),
        ("ageoa.scipy.sparse_graph_v2.allpairsshortestpath", "allpairsshortestpath"),
        ("ageoa.scipy.sparse_graph_v2.minimumspanningtree", "minimumspanningtree"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.scipy.sparse_graph_v2.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_astroflow_dedispersionkernel() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.astroflow.dedispersionkernel", "ageoa.astroflow.atoms", "dedispersionkernel")
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_build_interaction_graph_family() -> None:
    for atom_name, symbol in [
        (
            "ageoa.molecular_docking.build_interaction_graph.pair_distance_compatibility_check",
            "pair_distance_compatibility_check",
        ),
        (
            "ageoa.molecular_docking.build_interaction_graph.weighted_interaction_edge_derivation",
            "weighted_interaction_edge_derivation",
        ),
        (
            "ageoa.molecular_docking.build_interaction_graph.networkx_weighted_graph_materialization",
            "networkx_weighted_graph_materialization",
        ),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.molecular_docking.build_interaction_graph.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_quantum_solver_d12_family() -> None:
    for atom_name, symbol in [
        ("ageoa.molecular_docking.quantum_solver_d12.quantumsolverorchestrator", "quantumsolverorchestrator"),
        ("ageoa.molecular_docking.quantum_solver_d12.interactionboundscomputer", "interactionboundscomputer"),
        ("ageoa.molecular_docking.quantum_solver_d12.adiabaticpulseassembler", "adiabaticpulseassembler"),
        ("ageoa.molecular_docking.quantum_solver_d12.quantumcircuitsampler", "quantumcircuitsampler"),
        ("ageoa.molecular_docking.quantum_solver_d12.quantumsolutionextractor", "quantumsolutionextractor"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.molecular_docking.quantum_solver_d12.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_incremental_attention_configuration() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.mint.incremental_attention.enable_incremental_state_configuration",
            "ageoa.mint.incremental_attention",
            "enable_incremental_state_configuration",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_greedy_subgraph() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.molecular_docking.greedy_subgraph.greedy_maximum_subgraph",
            "ageoa.molecular_docking.greedy_subgraph",
            "greedy_maximum_subgraph",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_marks_quantum_mwis_solver_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.molecular_docking.quantum_mwis_solver",
            "ageoa.molecular_docking.atoms",
            "quantum_mwis_solver",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_marks_quantum_solver_helpers_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.molecular_docking.quantum_solver.quantumproblemdefinition", "quantumproblemdefinition"),
        ("ageoa.molecular_docking.quantum_solver.adiabaticquantumsampler", "adiabaticquantumsampler"),
        ("ageoa.molecular_docking.quantum_solver.solutionextraction", "solutionextraction"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.molecular_docking.quantum_solver.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_marks_greedy_mapping_helpers_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.molecular_docking.greedy_mapping.assemblestaticmappingcontext", "assemblestaticmappingcontext"),
        ("ageoa.molecular_docking.greedy_mapping.initializefrontierfromstartnode", "initializefrontierfromstartnode"),
        ("ageoa.molecular_docking.greedy_mapping.scoreandextendgreedycandidates", "scoreandextendgreedycandidates"),
        ("ageoa.molecular_docking.greedy_mapping.validatecurrentmapping", "validatecurrentmapping"),
        ("ageoa.molecular_docking.greedy_mapping.rungreedymappingpipeline", "rungreedymappingpipeline"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.molecular_docking.greedy_mapping.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_marks_add_quantum_link_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.molecular_docking.add_quantum_link.addquantumlink",
            "ageoa.molecular_docking.add_quantum_link",
            "addquantumlink",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_marks_mwis_to_qubo_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.molecular_docking.mwis_sa.to_qubo",
            "ageoa.molecular_docking.mwis_sa.atoms",
            "to_qubo",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_marks_threshold_permutation_enumeration_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.molecular_docking.minimize_bandwidth.enumerate_threshold_based_permutations",
            "ageoa.molecular_docking.minimize_bandwidth.atoms",
            "enumerate_threshold_based_permutations",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_marks_minimize_bandwidth_validation_helpers_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.molecular_docking.minimize_bandwidth.validate_square_matrix_shape", "validate_square_matrix_shape"),
        ("ageoa.molecular_docking.minimize_bandwidth.validate_symmetric_input", "validate_symmetric_input"),
        ("ageoa.molecular_docking.minimize_bandwidth.enforce_threshold_sparsity", "enforce_threshold_sparsity"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.molecular_docking.minimize_bandwidth.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_minimize_bandwidth_state_helpers_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.molecular_docking.minimize_bandwidth.initialize_reduction_state", "initialize_reduction_state"),
        ("ageoa.molecular_docking.minimize_bandwidth.extract_final_permutation", "extract_final_permutation"),
        ("ageoa.molecular_docking.minimize_bandwidth.select_minimum_bandwidth_permutation", "select_minimum_bandwidth_permutation"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.molecular_docking.minimize_bandwidth.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_minimize_bandwidth_greedy_loop_helpers_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.molecular_docking.minimize_bandwidth.propose_greedy_permutation_step", "propose_greedy_permutation_step"),
        ("ageoa.molecular_docking.minimize_bandwidth.update_state_with_improvement_criterion", "update_state_with_improvement_criterion"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.molecular_docking.minimize_bandwidth.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_minimize_bandwidth_numeric_helpers_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.molecular_docking.minimize_bandwidth.compute_absolute_weighted_index_distances", "compute_absolute_weighted_index_distances"),
        ("ageoa.molecular_docking.minimize_bandwidth.aggregate_maximum_distance_as_bandwidth", "aggregate_maximum_distance_as_bandwidth"),
        ("ageoa.molecular_docking.minimize_bandwidth.build_sparse_graph_view", "build_sparse_graph_view"),
        ("ageoa.molecular_docking.minimize_bandwidth.compute_symmetric_bandwidth_reducing_order", "compute_symmetric_bandwidth_reducing_order"),
        ("ageoa.molecular_docking.minimize_bandwidth.build_threshold_search_space", "build_threshold_search_space"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.molecular_docking.minimize_bandwidth.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_numpy_lexsort_v2_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.numpy.search_sort_v2.lexicographicindirectsort",
            "ageoa.numpy.search_sort_v2.atoms",
            "lexicographicindirectsort",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_marks_numpy_search_sort_v2_helpers_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.numpy.search_sort_v2.binarysearchinsertion", "binarysearchinsertion"),
        ("ageoa.numpy.search_sort_v2.partialsortpartition", "partialsortpartition"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.numpy.search_sort_v2.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_mint_axial_attention_helpers_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.mint.axial_attention.rowselfattention", "rowselfattention"),
        ("ageoa.mint.axial_attention.row_self_attention", "row_self_attention"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.mint.axial_attention", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_mint_top_level_attention_atoms_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.mint.axial_attention", "axial_attention"),
        ("ageoa.mint.rotary_positional_embeddings", "rotary_positional_embeddings"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.mint.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_alphafold_atoms_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.alphafold.invariant_point_attention", "invariant_point_attention"),
        ("ageoa.alphafold.equivariant_frame_update", "equivariant_frame_update"),
        ("ageoa.alphafold.coordinate_reconstruction", "coordinate_reconstruction"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.alphafold.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_e2e_ppg_windowed_reconstruction_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record(
            "ageoa.e2e_ppg.reconstruction.windowed_signal_reconstruction",
            "ageoa.e2e_ppg.reconstruction.atoms",
            "windowed_signal_reconstruction",
        )
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True


def test_runtime_probe_marks_biosppy_hamilton_detectors_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.ecg_detectors.hamilton_segmentation", "hamilton_segmentation"),
        ("ageoa.biosppy.ecg_detectors.hamilton_segmenter", "hamilton_segmenter"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.biosppy.ecg_detectors", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_biosppy_zz2018_main_wrappers_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.ecg_zz2018.calculatecompositesqi_zz2018", "calculatecompositesqi_zz2018"),
        ("ageoa.biosppy.ecg_zz2018.calculatebeatagreementsqi", "calculatebeatagreementsqi"),
        ("ageoa.biosppy.ecg_zz2018.calculatefrequencypowersqi", "calculatefrequencypowersqi"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.biosppy.ecg_zz2018.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_biosppy_zz2018_d12_wrappers_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.biosppy.ecg_zz2018_d12.assemblezz2018sqi", "assemblezz2018sqi"),
        ("ageoa.biosppy.ecg_zz2018_d12.computebeatagreementsqi", "computebeatagreementsqi"),
        ("ageoa.biosppy.ecg_zz2018_d12.computefrequencysqi", "computefrequencysqi"),
        ("ageoa.biosppy.ecg_zz2018_d12.computekurtosissqi", "computekurtosissqi"),
    ]:
        probe = runtime_probes.build_runtime_probe(
            _record(atom_name, "ageoa.biosppy.ecg_zz2018_d12.atoms", symbol)
        )
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_neurokit2_wrappers_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.neurokit2.averageqrstemplate", "averageqrstemplate"),
        ("ageoa.neurokit2.zhao2018hrvanalysis", "zhao2018hrvanalysis"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, "ageoa.neurokit2.atoms", symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_kazemi_wrapper_d12_wrappers_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.e2e_ppg.kazemi_wrapper_d12.normalizesignal", "normalizesignal"),
        ("ageoa.e2e_ppg.kazemi_wrapper_d12.wrapperevaluate", "wrapperevaluate"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, "ageoa.e2e_ppg.kazemi_wrapper_d12.atoms", symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True


def test_runtime_probe_marks_numpy_random_v2_family_as_usage_equivalent() -> None:
    for atom_name, symbol in [
        ("ageoa.numpy.random_v2.continuousmultivariatesampler", "continuousmultivariatesampler"),
        ("ageoa.numpy.random_v2.discreteeventsampler", "discreteeventsampler"),
        ("ageoa.numpy.random_v2.combinatoricssampler", "combinatoricssampler"),
    ]:
        probe = runtime_probes.build_runtime_probe(_record(atom_name, "ageoa.numpy.random_v2.atoms", symbol))
        assert probe["status"] == "pass"
        assert probe["parity_used"] is True
        assert "RUNTIME_PROBE_PASS" in probe["findings"]
        assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_marks_jfof_wrapper_as_usage_equivalent() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.jFOF.find_fof_clusters", "ageoa.jFOF.atoms", "find_fof_clusters")
    )
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_records_positive_failure(monkeypatch) -> None:
    atom_name = "ageoa.example.fail"
    monkeypatch.setitem(
        runtime_probes.PROBE_PLANS,
        atom_name,
        runtime_probes.ProbePlan(
            positive=runtime_probes.ProbeCase("always fails", lambda func: func(), None),
            negative=runtime_probes.ProbeCase("negative passes", lambda func: (_ for _ in ()).throw(ValueError("bad")), expect_exception=True),
        ),
    )
    monkeypatch.setattr(runtime_probes, "safe_import_module", lambda _: SimpleNamespace(atom=lambda: (_ for _ in ()).throw(RuntimeError("boom"))))
    probe = runtime_probes.build_runtime_probe(_record(atom_name, "ageoa.example", "atom"))
    assert probe["status"] == "fail"
    assert "RUNTIME_PROBE_FAIL" in probe["findings"]
    assert probe["positive_probe"]["exception_type"] == "RuntimeError"
