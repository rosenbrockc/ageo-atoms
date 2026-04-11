"""Focused runtime-probe coverage for institutional_quant_engine family packets."""

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


def test_runtime_probe_passes_for_institutional_quant_engine_stateful_helpers() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.institutional_quant_engine.kalman_filter.kalmanfilterinit", "ageoa.institutional_quant_engine.kalman_filter.atoms", "kalmanfilterinit"),
        ("ageoa.institutional_quant_engine.kalman_filter.kalmanmeasurementupdate", "ageoa.institutional_quant_engine.kalman_filter.atoms", "kalmanmeasurementupdate"),
        ("ageoa.institutional_quant_engine.queue_estimator.initializeorderstate", "ageoa.institutional_quant_engine.queue_estimator.atoms", "initializeorderstate"),
        ("ageoa.institutional_quant_engine.queue_estimator.updatequeueontrade", "ageoa.institutional_quant_engine.queue_estimator.atoms", "updatequeueontrade"),
    ]:
        _assert_probe_passes(atom_name, module_path, symbol)


def test_runtime_probe_passes_for_institutional_quant_engine_script_wrappers() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.institutional_quant_engine.copula_dependence.simulate_copula_dependence", "ageoa.institutional_quant_engine.copula_dependence", "simulate_copula_dependence"),
        ("ageoa.institutional_quant_engine.dynamic_hedge.kalman_hedge_ratio", "ageoa.institutional_quant_engine.dynamic_hedge", "kalman_hedge_ratio"),
        ("ageoa.institutional_quant_engine.evt_model.fit_gpd_tail", "ageoa.institutional_quant_engine.evt_model", "fit_gpd_tail"),
        ("ageoa.institutional_quant_engine.supply_chain.propagate_supply_shock", "ageoa.institutional_quant_engine.supply_chain", "propagate_supply_shock"),
        ("ageoa.institutional_quant_engine.triangular_arbitrage.detect_triangular_arbitrage", "ageoa.institutional_quant_engine.triangular_arbitrage", "detect_triangular_arbitrage"),
        ("ageoa.institutional_quant_engine.wash_trade.detect_wash_trade_rings", "ageoa.institutional_quant_engine.wash_trade", "detect_wash_trade_rings"),
    ]:
        _assert_probe_passes(atom_name, module_path, symbol)


def test_runtime_probe_passes_for_institutional_quant_engine_market_making_wrapper() -> None:
    _assert_probe_passes(
        "ageoa.institutional_quant_engine.market_making_avellaneda",
        "ageoa.institutional_quant_engine.atoms",
        "market_making_avellaneda",
    )


def test_runtime_probe_passes_for_institutional_quant_engine_portfolio_wrappers() -> None:
    for atom_name, module_path, symbol in [
        ("ageoa.institutional_quant_engine.avellaneda_stoikov.initializemarketmakerstate", "ageoa.institutional_quant_engine.avellaneda_stoikov.atoms", "initializemarketmakerstate"),
        ("ageoa.institutional_quant_engine.avellaneda_stoikov.computeinventoryadjustedquotes", "ageoa.institutional_quant_engine.avellaneda_stoikov.atoms", "computeinventoryadjustedquotes"),
        ("ageoa.institutional_quant_engine.hierarchical_risk_parity.compute_hrp_weights", "ageoa.institutional_quant_engine.hierarchical_risk_parity", "compute_hrp_weights"),
        ("ageoa.institutional_quant_engine.hierarchical_risk_parity.hrppipelinerun", "ageoa.institutional_quant_engine.hierarchical_risk_parity", "hrppipelinerun"),
    ]:
        _assert_probe_passes(atom_name, module_path, symbol)


def test_runtime_probe_passes_for_institutional_quant_engine_almgren_chriss_v2_family() -> None:
    for atom_name, symbol in [
        ("ageoa.institutional_quant_engine.almgren_chriss_v2.riskaversioninit", "riskaversioninit"),
        ("ageoa.institutional_quant_engine.almgren_chriss_v2.optimalexecutiontrajectory", "optimalexecutiontrajectory"),
    ]:
        _assert_probe_passes(
            atom_name,
            "ageoa.institutional_quant_engine.almgren_chriss_v2.atoms",
            symbol,
        )


def test_runtime_probe_passes_for_institutional_quant_engine_almgren_chriss_generated_wrapper() -> None:
    _assert_probe_passes(
        "ageoa.institutional_quant_engine.almgren_chriss.computeoptimaltrajectory",
        "ageoa.institutional_quant_engine.almgren_chriss",
        "computeoptimaltrajectory",
    )


def test_runtime_probe_passes_for_institutional_quant_engine_order_flow_and_pin_generated_wrappers() -> None:
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
        _assert_probe_passes(atom_name, module_path, symbol)


def test_runtime_probe_passes_for_institutional_quant_engine_avellaneda_stoikov_d12_family() -> None:
    for atom_name, symbol in [
        ("ageoa.institutional_quant_engine.avellaneda_stoikov_d12.marketmakerstateinit", "marketmakerstateinit"),
        ("ageoa.institutional_quant_engine.avellaneda_stoikov_d12.optimalquotecalculation", "optimalquotecalculation"),
    ]:
        _assert_probe_passes(
            atom_name,
            "ageoa.institutional_quant_engine.avellaneda_stoikov_d12.atoms",
            symbol,
        )


def test_runtime_probe_passes_for_institutional_quant_engine_fractional_diff_helper() -> None:
    _assert_probe_passes(
        "ageoa.institutional_quant_engine.fractional_diff.fractional_differentiator",
        "ageoa.institutional_quant_engine.fractional_diff",
        "fractional_differentiator",
    )
