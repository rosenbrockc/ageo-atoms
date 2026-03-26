#!/usr/bin/env bash
# Retry failed ingestions: Gemini quota failures → codex, fixed pronto paths
set -u

MATCHER=/Users/conrad/personal/ageo-matcher
ATOMS=/Users/conrad/personal/ageo-atoms
TP=$ATOMS/third_party
OUT=$ATOMS/ageoa
export AGEOM_INGESTER_MAX_DEPTH=12

LOG_DIR=$ATOMS/logs/ingest
mkdir -p "$LOG_DIR"

# Concurrency limiter: max N jobs at a time
MAX_JOBS=4
job_count=0

wait_for_slot() {
    while [ "$job_count" -ge "$MAX_JOBS" ]; do
        wait -n 2>/dev/null || true
        job_count=$((job_count - 1))
    done
}

ingest() {
    local src="$1" cls="$2" out="$3" provider="$4"
    local name=$(basename "$out")
    # Remove old failed output so we get a fresh run
    rm -rf "$OUT/$out"
    echo "START $name (provider=$provider)"
    cd "$MATCHER" && ageom ingest "$src" --class "$cls" --output "$OUT/$out" \
        --llm-provider "$provider" --trace \
        >"$LOG_DIR/${name}.log" 2>&1
    local rc=$?
    echo "DONE  $name (rc=$rc)"
}

# --- Gemini quota failures → retry with alternating codex/gemini, 4 at a time ---

wait_for_slot
ingest "$TP/BioSPPy/biosppy/signals/pcg.py" homomorphic_filter biosppy/pcg_homomorphic codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/BioSPPy/biosppy/signals/ppg.py" find_onsets_elgendi2013 biosppy/ppg_elgendi gemini_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/E2E-PPG/ppg_reconstruction.py" gan_rec e2e_ppg/gan_rec codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/E2E-PPG/kazemi_peak_detection.py" Wrapper_function e2e_ppg/kazemi_wrapper gemini_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/Institutional-Quant-Engine/execution_hft/28_order_flow_imbalance.py" calculate_ofi institutional_quant_engine/order_flow_imbalance codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/Molecular-Docking/src/graph/interaction_graph.py" build_weighted_binding_interaction_graph molecular_docking/build_interaction_graph gemini_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/Molecular-Docking/src/solver/mwis_SA.py" mwis_SA molecular_docking/mwis_sa codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/Molecular-Docking/src/solver/quantum_solver_molecular.py" q_solver molecular_docking/quantum_solver gemini_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/mint/mint/multihead_attention.py" with_incremental_state mint/incremental_attention codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/mint/mint/data.py" FastaBatchedDataset mint/fasta_dataset gemini_cli &
job_count=$((job_count + 1))

# --- Rust (gemini quota failure) ---

wait_for_slot
ingest "$TP/rust_robotics/src/models/humanoid/n_joint_arm2_d.rs" ModelD rust_robotics/n_joint_arm_2d codex_cli &
job_count=$((job_count + 1))

# --- Pronto C++ (corrected paths → .hpp headers) ---

wait_for_slot
ingest "$TP/pronto/pronto_quadruped/include/pronto_quadruped/DynamicStanceEstimator.hpp" DynamicStanceEstimator pronto/dynamic_stance_estimator gemini_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/pronto/pronto_quadruped/include/pronto_quadruped/FlexEstimator.hpp" FlexEstimator pronto/flex_estimator codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/pronto/pronto_quadruped/include/pronto_quadruped/LegOdometer.hpp" LegOdometer pronto/leg_odometer gemini_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/pronto/pronto_utils/include/pronto_utils/backlash_filter.hpp" BacklashFilter pronto/backlash_filter codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/pronto/pronto_utils/include/pronto_utils/BlipFilter.hpp" BlipFilter pronto/blip_filter gemini_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/pronto/pronto_core/include/pronto_core/state_est.hpp" StateEstimator pronto/state_estimator codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/pronto/pronto_quadruped/include/pronto_quadruped/InverseSchmittTrigger.hpp" InverseSchmittTrigger pronto/inverse_schmitt gemini_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/pronto/pronto_biped_core/include/pronto_biped_core/foot_contact_classify.hpp" FootContactClassifier pronto/foot_contact codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/pronto/pronto_biped_core/include/pronto_biped_core/yawlock_common.hpp" YawLock pronto/yaw_lock gemini_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/pronto/pronto_utils/include/pronto_utils/torque_adjustment.hpp" TorqueAdjustment pronto/torque_adjustment codex_cli &
job_count=$((job_count + 1))

# --- greedy_subgraph: use --procedural since entry point is a class method ---
wait_for_slot
rm -rf "$OUT/molecular_docking/greedy_subgraph"
echo "START greedy_subgraph (procedural)"
cd "$MATCHER" && ageom ingest "$TP/Molecular-Docking/src/solver/greedy_subgraph_vv.py" \
    --class greedy_subgraph --output "$OUT/molecular_docking/greedy_subgraph" \
    --procedural --trace \
    >"$LOG_DIR/greedy_subgraph.log" 2>&1 &
echo "DONE  greedy_subgraph (rc=$?)"
job_count=$((job_count + 1))

echo "All retries launched. Waiting..."
wait
echo "All retries complete."
