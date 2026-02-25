#!/usr/bin/env bash
# Re-ingest MCMC repos using extract_function (LLM pipeline, not --procedural)
set -u

MATCHER=/Users/conrad/personal/ageo-matcher
ATOMS=/Users/conrad/personal/ageo-atoms
TP=$ATOMS/third_party
OUT=$ATOMS/ageoa/mcmc_foundational
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
    echo "START $name (class=$cls, provider=$provider)"
    cd "$MATCHER" && ageom ingest "$src" --class "$cls" --output "$OUT/$out" \
        --llm-provider "$provider" --trace \
        >"$LOG_DIR/mcmc_${name}.log" 2>&1
    local rc=$?
    echo "DONE  $name (rc=$rc)"
}

# === Task 1: mini-mcmc (Rust) — structs ===
wait_for_slot
ingest "$TP/mini-mcmc/src/hmc.rs" HMC mini_mcmc/hmc codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/mini-mcmc/src/nuts.rs" NUTS mini_mcmc/nuts gemini_cli &
job_count=$((job_count + 1))

# === Task 2: AdvancedHMC.jl (Julia) — functions → extract_function fallback ===
wait_for_slot
ingest "$TP/AdvancedHMC.jl/src/integrator.jl" step advancedhmc/integrator codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/AdvancedHMC.jl/src/trajectory.jl" transition advancedhmc/trajectory gemini_cli &
job_count=$((job_count + 1))

# === Task 3: kthohr/mcmc (C++) — use .ipp where available, .hpp otherwise ===
# .ipp files have implementations (function bodies); .hpp are declarations only
wait_for_slot
ingest "$TP/mcmc/include/mcmc/hmc.hpp" hmc kthohr_mcmc/hmc codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/mcmc/include/mcmc/nuts.ipp" nuts_build_tree kthohr_mcmc/nuts gemini_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/mcmc/include/mcmc/rwmh.hpp" rwmh kthohr_mcmc/rwmh codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/mcmc/include/mcmc/aees.ipp" single_step_mh kthohr_mcmc/aees gemini_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/mcmc/include/mcmc/de.hpp" de kthohr_mcmc/de codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/mcmc/include/mcmc/mala.ipp" mala_prop_adjustment kthohr_mcmc/mala gemini_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/mcmc/include/mcmc/rmhmc.hpp" rmhmc kthohr_mcmc/rmhmc codex_cli &
job_count=$((job_count + 1))

echo "All MCMC v2 re-ingestions launched. Waiting..."
wait
echo "All MCMC v2 re-ingestions complete."
