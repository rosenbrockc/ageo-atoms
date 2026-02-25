#!/usr/bin/env bash
# Retry #2: all codex_cli, MAX_JOBS=2 to avoid quota issues
set -u

MATCHER=/Users/conrad/personal/ageo-matcher
ATOMS=/Users/conrad/personal/ageo-atoms
TP=$ATOMS/third_party
OUT=$ATOMS/ageoa
export AGEOM_INGESTER_MAX_DEPTH=12

LOG_DIR=$ATOMS/logs/ingest
mkdir -p "$LOG_DIR"

MAX_JOBS=2
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
    rm -rf "$OUT/$out"
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
    rm -rf "$OUT/$out"
    echo "START $name (procedural)"
    cd "$MATCHER" && ageom ingest "$src" --class "$cls" --output "$OUT/$out" \
        --procedural --trace \
        >"$LOG_DIR/${name}.log" 2>&1
    local rc=$?
    echo "DONE  $name (rc=$rc)"
}

# --- Gemini quota failures → all codex_cli now ---

wait_for_slot
ingest "$TP/pronto/pronto_biped_core/include/pronto_biped_core/yawlock_common.hpp" YawLock pronto/yaw_lock codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/pronto/pronto_quadruped/include/pronto_quadruped/LegOdometer.hpp" LegOdometer pronto/leg_odometer codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/Molecular-Docking/src/graph/interaction_graph.py" build_weighted_binding_interaction_graph molecular_docking/build_interaction_graph codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/pronto/pronto_quadruped/include/pronto_quadruped/InverseSchmittTrigger.hpp" InverseSchmittTrigger pronto/inverse_schmitt codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/E2E-PPG/kazemi_peak_detection.py" Wrapper_function e2e_ppg/kazemi_wrapper codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/mint/mint/data.py" FastaBatchedDataset mint/fasta_dataset codex_cli &
job_count=$((job_count + 1))

# --- mwis_SA: no class/function named mwis_SA, use procedural ---
wait_for_slot
ingest_procedural "$TP/Molecular-Docking/src/solver/mwis_SA.py" mwis_SA molecular_docking/mwis_sa &
job_count=$((job_count + 1))

echo "All retry2 launched. Waiting..."
wait
echo "All retry2 complete."
