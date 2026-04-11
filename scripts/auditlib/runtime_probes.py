"""Compatibility wrapper for conservative deterministic runtime probes."""

from __future__ import annotations

import logging

from . import runtime_probes_core as _core
from .runtime_probe_assertions import (
    _assert_array,
    _assert_batch_plan,
    _assert_dataset_state,
    _assert_dict_keys,
    _assert_draw_bundle,
    _assert_finite_vector,
    _assert_float_int_pair,
    _assert_float_list,
    _assert_float_mask,
    _assert_int_pair,
    _assert_inventory_adjusted_quotes,
    _assert_market_maker_state,
    _assert_monotonic_index_array,
    _assert_nonincreasing_float_list,
    _assert_online_filter_init_state,
    _assert_online_filter_step_result,
    _assert_optimize_result_near,
    _assert_pair_of_arrays,
    _assert_pair_of_sorted_integer_arrays,
    _assert_permutation_list,
    _assert_positive_weights,
    _assert_profitable_cycles,
    _assert_quantum_solution_extractor,
    _assert_quantum_solver_orchestrator_result,
    _assert_scalar,
    _assert_search_space,
    _assert_shape,
    _assert_sorted_array,
    _assert_sparse_shape,
    _assert_state_snapshot,
    _assert_triple_of_arrays_matching_onsets,
    _assert_tuple,
    _assert_type,
    _assert_unit_interval_shape,
    _assert_value,
)
from .runtime_probe_plans import (
    get_advancedvi_and_iqe_probe_plans,
    get_belief_propagation_probe_plans,
    get_biosppy_probe_plans,
    get_conjugate_priors_and_small_mcmc_probe_plans,
    get_foundation_probe_plans,
    get_hftbacktest_and_ingest_probe_plans,
    get_kalman_filter_probe_plans,
    get_mcmc_foundational_probe_plans,
    get_molecular_docking_probe_plans,
    get_neurokit2_probe_plans,
    get_numpy_basic_probe_plans,
    get_numpy_fft_v2_probe_plans,
    get_numpy_search_sort_v2_probe_plans,
    get_particle_filter_and_pasqal_probe_plans,
    get_pronto_probe_plans,
    get_quant_engine_probe_plans,
    get_quantfin_probe_plans,
    get_rust_robotics_probe_plans,
    get_scipy_basic_probe_plans,
    get_scipy_optimize_v2_probe_plans,
    get_scipy_sparse_graph_probe_plans,
    get_scipy_stats_integrate_probe_plans,
    get_search_probe_plans,
    get_sorting_probe_plans,
)
from .runtime_probes_core import (
    ProbeCase,
    ProbePlan,
    assemble_probe_plans,
    get_probe_plans,
    install_ageoa_stub,
    install_package_stub,
    load_module_from_file,
    safe_import_module,
    set_probe_plans,
)

logger = logging.getLogger(__name__)


def _optional_probe_plans(loader):
    try:
        return loader()
    except Exception:
        logger.debug("Skipping runtime probe plan loader %s", getattr(loader, "__name__", repr(loader)), exc_info=True)
        return {}


PROBE_PLANS = assemble_probe_plans(
    get_search_probe_plans(),
    get_numpy_basic_probe_plans(),
    get_scipy_basic_probe_plans(),
    get_sorting_probe_plans(),
    get_scipy_sparse_graph_probe_plans(),
    get_scipy_stats_integrate_probe_plans(),
    get_numpy_fft_v2_probe_plans(),
    get_numpy_search_sort_v2_probe_plans(),
    get_scipy_optimize_v2_probe_plans(),
    get_foundation_probe_plans(),
    get_advancedvi_and_iqe_probe_plans(),
    get_belief_propagation_probe_plans(),
    get_quant_engine_probe_plans(),
    get_particle_filter_and_pasqal_probe_plans(),
    get_rust_robotics_probe_plans(),
    get_quantfin_probe_plans(),
    get_kalman_filter_probe_plans(),
    get_mcmc_foundational_probe_plans(),
    _optional_probe_plans(get_hftbacktest_and_ingest_probe_plans),
    get_molecular_docking_probe_plans(),
    _optional_probe_plans(get_neurokit2_probe_plans),
    get_biosppy_probe_plans(),
    get_pronto_probe_plans(),
    get_conjugate_priors_and_small_mcmc_probe_plans(),
)
set_probe_plans(PROBE_PLANS)


def build_runtime_probe(record):
    original_importer = _core.safe_import_module
    _core.safe_import_module = safe_import_module
    try:
        set_probe_plans(PROBE_PLANS)
        return _core.build_runtime_probe(record)
    finally:
        _core.safe_import_module = original_importer


def write_runtime_probe(record):
    original_importer = _core.safe_import_module
    _core.safe_import_module = safe_import_module
    try:
        set_probe_plans(PROBE_PLANS)
        return _core.write_runtime_probe(record)
    finally:
        _core.safe_import_module = original_importer
