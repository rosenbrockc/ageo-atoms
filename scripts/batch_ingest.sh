#!/usr/bin/env bash
# Batch ingest all INTEREST.md entries at max_depth=12
# Alternates between gemini_cli and codex_cli
set -u

MATCHER=/Users/conrad/personal/ageo-matcher
ATOMS=/Users/conrad/personal/ageo-atoms
TP=$ATOMS/third_party
OUT=$ATOMS/ageoa
export AGEOM_INGESTER_MAX_DEPTH=12

LOG_DIR=$ATOMS/logs/ingest
mkdir -p "$LOG_DIR"

ingest() {
    local src="$1" cls="$2" out="$3" provider="$4"
    local name=$(basename "$out")
    if [ -f "$OUT/$out/COMPLETED.json" ]; then
        echo "SKIP $name (already done)"
        return 0
    fi
    echo "START $name (provider=$provider)"
    cd "$MATCHER" && ageom ingest "$src" --class "$cls" --output "$OUT/$out" \
        --llm-provider "$provider" --trace \
        >"$LOG_DIR/${name}.log" 2>&1
    local rc=$?
    echo "DONE  $name (rc=$rc)"
}

ingest_procedural() {
    local src="$1" cls="$2" out="$3"
    local name=$(basename "$out")
    if [ -f "$OUT/$out/COMPLETED.json" ]; then
        echo "SKIP $name (already done)"
        return 0
    fi
    echo "START $name (procedural)"
    cd "$MATCHER" && ageom ingest "$src" --class "$cls" --output "$OUT/$out" \
        --procedural --trace \
        >"$LOG_DIR/${name}.log" 2>&1
    local rc=$?
    echo "DONE  $name (rc=$rc)"
}

# --- BioSPPy (Python functions) ---
ingest "$TP/BioSPPy/biosppy/signals/ecg.py" ASI_segmenter biosppy/ecg_asi codex_cli &
ingest "$TP/BioSPPy/biosppy/signals/ecg.py" ZZ2018 biosppy/ecg_zz2018 gemini_cli &
ingest "$TP/BioSPPy/biosppy/signals/emg.py" solnik_onset_detector biosppy/emg_solnik codex_cli &
ingest "$TP/BioSPPy/biosppy/signals/emg.py" bonato_onset_detector biosppy/emg_bonato gemini_cli &
ingest "$TP/BioSPPy/biosppy/signals/emg.py" abbink_onset_detector biosppy/emg_abbink codex_cli &
ingest "$TP/BioSPPy/biosppy/signals/pcg.py" homomorphic_filter biosppy/pcg_homomorphic gemini_cli &
ingest "$TP/BioSPPy/biosppy/signals/ppg.py" find_onsets_kavsaoglu2016 biosppy/ppg_kavsaoglu codex_cli &
ingest "$TP/BioSPPy/biosppy/signals/ppg.py" find_onsets_elgendi2013 biosppy/ppg_elgendi gemini_cli &
ingest "$TP/BioSPPy/biosppy/signals/abp.py" find_onsets_zong2003 biosppy/abp_zong codex_cli &

# --- E2E-PPG (Python functions) ---
ingest "$TP/E2E-PPG/ppg_reconstruction.py" gan_rec e2e_ppg/gan_rec gemini_cli &
ingest "$TP/E2E-PPG/ppg_reconstruction.py" reconstruction e2e_ppg/reconstruction codex_cli &
ingest "$TP/E2E-PPG/ppg_sqa.py" heart_cycle_detection e2e_ppg/heart_cycle gemini_cli &
ingest "$TP/E2E-PPG/ppg_sqa.py" template_matching_features e2e_ppg/template_matching codex_cli &
ingest "$TP/E2E-PPG/kazemi_peak_detection.py" Wrapper_function e2e_ppg/kazemi_wrapper gemini_cli &

# --- IQE classes ---
ingest "$TP/Institutional-Quant-Engine/execution_hft/23_market_making_avellaneda.py" MarketMaker institutional_quant_engine/avellaneda_stoikov codex_cli &
ingest "$TP/Institutional-Quant-Engine/execution_hft/49_limit_order_queue_estimator.py" QueueTracker institutional_quant_engine/queue_estimator gemini_cli &

# --- IQE functions (will use extract_function fallback) ---
ingest "$TP/Institutional-Quant-Engine/execution_hft/17_optimal_execution_almgren.py" optimal_trajectory institutional_quant_engine/almgren_chriss codex_cli &
ingest "$TP/Institutional-Quant-Engine/execution_hft/28_order_flow_imbalance.py" calculate_ofi institutional_quant_engine/order_flow_imbalance gemini_cli &
ingest "$TP/Institutional-Quant-Engine/execution_hft/41_pin_informed_trading.py" pin_likelihood institutional_quant_engine/pin_model codex_cli &
ingest "$TP/Institutional-Quant-Engine/research_math/34_fractional_diff_stationarity.py" frac_diff institutional_quant_engine/fractional_diff gemini_cli &
ingest "$TP/Institutional-Quant-Engine/research_math/35_hawkes_process_arrival.py" simulate_hawkes institutional_quant_engine/hawkes_process codex_cli &

# --- IQE procedural (script-only, no clear entry function) ---
ingest_procedural "$TP/Institutional-Quant-Engine/risk_portfolio/16_hierarchical_risk_parity.py" HRP institutional_quant_engine/hierarchical_risk_parity &
ingest_procedural "$TP/Institutional-Quant-Engine/risk_portfolio/27_copula_dependence.py" CopulaDependence institutional_quant_engine/copula_dependence &
ingest_procedural "$TP/Institutional-Quant-Engine/risk_portfolio/24_tail_risk_evt.py" EVTModel institutional_quant_engine/evt_model &
ingest_procedural "$TP/Institutional-Quant-Engine/research_math/19_triangular_arbitrage_graph.py" TriangularArbitrage institutional_quant_engine/triangular_arbitrage &
ingest_procedural "$TP/Institutional-Quant-Engine/strategies/29_dynamic_hedge_kalman.py" DynamicHedge institutional_quant_engine/dynamic_hedge &
ingest_procedural "$TP/Institutional-Quant-Engine/strategies/47_supply_chain_propagation.py" SupplyChainShock institutional_quant_engine/supply_chain &
ingest_procedural "$TP/Institutional-Quant-Engine/ops_compliance/50_trade_surveillance_wash_graph.py" WashTradeDetector institutional_quant_engine/wash_trade &
ingest_procedural "$TP/Institutional-Quant-Engine/derivatives_pricing/30_stochastic_volatility_heston.py" HestonModel institutional_quant_engine/heston_model &

# --- Molecular-Docking (Python functions/classes) ---
ingest "$TP/Molecular-Docking/src/solver/quantum_solver_molecular.py" q_solver molecular_docking/quantum_solver gemini_cli &
ingest "$TP/Molecular-Docking/src/solver/optimiser.py" minimize_bandwidth molecular_docking/minimize_bandwidth codex_cli &
ingest "$TP/Molecular-Docking/src/solver/greedy_subgraph_vv.py" solve_breadthfirst_2_sol molecular_docking/greedy_subgraph gemini_cli &
ingest "$TP/Molecular-Docking/src/solver/greedy_lattice_mapping.py" GreedyMapping molecular_docking/greedy_mapping codex_cli &
ingest "$TP/Molecular-Docking/src/graph/mapping.py" add_quantum_link molecular_docking/add_quantum_link gemini_cli &
ingest "$TP/Molecular-Docking/src/graph/mapping.py" map_to_UDG molecular_docking/map_to_udg codex_cli &
ingest "$TP/Molecular-Docking/src/graph/interaction_graph.py" build_weighted_binding_interaction_graph molecular_docking/build_interaction_graph gemini_cli &
ingest "$TP/Molecular-Docking/src/graph/mapping.py" build_complementary_graph molecular_docking/build_complementary codex_cli &
ingest "$TP/Molecular-Docking/src/solver/mwis_SA.py" mwis_SA molecular_docking/mwis_sa gemini_cli &

# --- Mint (Python classes/functions) ---
ingest "$TP/mint/mint/rotary_embedding.py" RotaryEmbedding mint/rotary_embedding codex_cli &
ingest "$TP/mint/mint/multihead_attention.py" with_incremental_state mint/incremental_attention gemini_cli &
ingest "$TP/mint/mint/axial_attention.py" RowSelfAttention mint/axial_attention codex_cli &
ingest "$TP/mint/mint/data.py" FastaBatchedDataset mint/fasta_dataset gemini_cli &
ingest "$TP/mint/mint/modules.py" apc mint/apc_module codex_cli &
ingest "$TP/mint/downstream/TCR-Epitope/tcr-interface/teim_utils.py" encoding_dist_mat mint/encoding_dist_mat gemini_cli &

# --- Pronto (C++ classes via TreeSitter) ---
ingest "$TP/pronto/pronto_quadruped/src/DynamicStanceEstimator.cpp" DynamicStanceEstimator pronto/dynamic_stance_estimator codex_cli &
ingest "$TP/pronto/pronto_quadruped/src/FlexEstimator.cpp" FlexEstimator pronto/flex_estimator gemini_cli &
ingest "$TP/pronto/pronto_quadruped/src/LegOdometer.cpp" LegOdometer pronto/leg_odometer codex_cli &
ingest "$TP/pronto/pronto_utils/src/BacklashFilter.cpp" BacklashFilter pronto/backlash_filter gemini_cli &
ingest "$TP/pronto/pronto_utils/src/BlipFilter.cpp" BlipFilter pronto/blip_filter codex_cli &
ingest "$TP/pronto/pronto_utils/src/StateEstimator.cpp" StateEstimator pronto/state_estimator gemini_cli &
ingest "$TP/pronto/pronto_utils/src/InverseSchmittTrigger.cpp" InverseSchmittTrigger pronto/inverse_schmitt codex_cli &
ingest "$TP/pronto/pronto_utils/src/FootContactClassifier.cpp" FootContactClassifier pronto/foot_contact gemini_cli &
ingest "$TP/pronto/pronto_utils/src/YawLock.cpp" YawLock pronto/yaw_lock codex_cli &
ingest "$TP/pronto/pronto_utils/src/TorqueAdjustment.cpp" TorqueAdjustment pronto/torque_adjustment gemini_cli &

# --- Rust Robotics (Rust structs via TreeSitter) ---
ingest "$TP/rust_robotics/src/models/ground_vehicles/bicycle_kinematic.rs" Model rust_robotics/bicycle_kinematic codex_cli &
ingest "$TP/rust_robotics/src/models/humanoid/n_joint_arm2_d.rs" ModelD rust_robotics/n_joint_arm_2d gemini_cli &
ingest "$TP/rust_robotics/src/models/ground_vehicles/longitudinal_dynamics.rs" Model rust_robotics/longitudinal_dynamics codex_cli &

# --- Tempo.jl (Julia — procedural fallback, extract_function not supported) ---
ingest_procedural "$TP/Tempo.jl/src/convert.jl" find_year tempo_jl/find_year &
ingest_procedural "$TP/Tempo.jl/src/datetime.jl" find_month tempo_jl/find_month &
ingest_procedural "$TP/Tempo.jl/src/datetime.jl" jd2cal tempo_jl/jd2cal &
ingest_procedural "$TP/Tempo.jl/src/offset.jl" offset_tt2tdb tempo_jl/offsets &
ingest_procedural "$TP/Tempo.jl/src/offset.jl" offset_tt2tdbh tempo_jl/offsets &
ingest_procedural "$TP/Tempo.jl/src/scales.jl" apply_offsets tempo_jl/apply_offsets &
ingest_procedural "$TP/Tempo.jl/src/convert.jl" tai2utc tempo_jl/tai2utc &
ingest_procedural "$TP/Tempo.jl/src/convert.jl" utc2tai tempo_jl/utc2tai &

# NOTE: quantfin (Haskell .hs) and Pulsar_Folding (.ipynb) are skipped — unsupported formats.

echo "All ingestions launched. Waiting for completion..."
wait
echo "All ingestions complete."
