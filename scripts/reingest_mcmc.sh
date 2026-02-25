#!/usr/bin/env bash
# Re-ingest all three MCMC repos at max_depth=12
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
    echo "START $name (provider=$provider)"
    cd "$MATCHER" && ageom ingest "$src" --class "$cls" --output "$OUT/$out" \
        --llm-provider "$provider" --trace \
        >"$LOG_DIR/mcmc_${name}.log" 2>&1
    local rc=$?
    echo "DONE  $name (rc=$rc)"
}

ingest_procedural() {
    local src="$1" cls="$2" out="$3" provider="$4"
    local name=$(basename "$out")
    rm -rf "$OUT/$out"
    echo "START $name (procedural, provider=$provider)"
    cd "$MATCHER" && ageom ingest "$src" --class "$cls" --output "$OUT/$out" \
        --procedural --llm-provider "$provider" --trace \
        >"$LOG_DIR/mcmc_${name}.log" 2>&1
    local rc=$?
    echo "DONE  $name (rc=$rc)"
}

# === Task 1: mini-mcmc (Rust) — structs HMC and NUTS ===

wait_for_slot
ingest "$TP/mini-mcmc/src/hmc.rs" HMC mini_mcmc/hmc codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest "$TP/mini-mcmc/src/nuts.rs" NUTS mini_mcmc/nuts gemini_cli &
job_count=$((job_count + 1))

# === Task 2: AdvancedHMC.jl (Julia) — procedural since Julia uses tree-sitter ===

wait_for_slot
ingest_procedural "$TP/AdvancedHMC.jl/src/integrator.jl" AbstractIntegrator advancedhmc/integrator codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest_procedural "$TP/AdvancedHMC.jl/src/trajectory.jl" Trajectory advancedhmc/trajectory gemini_cli &
job_count=$((job_count + 1))

# === Task 3: kthohr/mcmc (C++) — procedural, flat function API ===

wait_for_slot
ingest_procedural "$TP/mcmc/include/mcmc/hmc.hpp" hmc kthohr_mcmc/hmc codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest_procedural "$TP/mcmc/include/mcmc/nuts.hpp" nuts kthohr_mcmc/nuts gemini_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest_procedural "$TP/mcmc/include/mcmc/rwmh.hpp" rwmh kthohr_mcmc/rwmh codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest_procedural "$TP/mcmc/include/mcmc/aees.hpp" aees kthohr_mcmc/aees gemini_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest_procedural "$TP/mcmc/include/mcmc/de.hpp" de kthohr_mcmc/de codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest_procedural "$TP/mcmc/include/mcmc/mala.hpp" mala kthohr_mcmc/mala gemini_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest_procedural "$TP/mcmc/include/mcmc/rmhmc.hpp" rmhmc kthohr_mcmc/rmhmc codex_cli &
job_count=$((job_count + 1))

wait_for_slot
ingest_procedural "$TP/mcmc/include/mcmc/mcmc_algos.hpp" mcmc_algos kthohr_mcmc/mcmc_algos gemini_cli &
job_count=$((job_count + 1))

echo "All MCMC re-ingestions launched. Waiting..."
wait
echo "All MCMC re-ingestions complete."
