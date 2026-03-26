#!/usr/bin/env bash
set -euo pipefail

export AGEOM_INGESTER_MAX_DEPTH=12
MATCHER=/Users/conrad/personal/ageo-matcher
TP=/Users/conrad/personal/ageo-atoms/third_party
OUT=/Users/conrad/personal/ageo-atoms/ageoa
LOGS=/Users/conrad/personal/ageo-atoms/logs/ingest

# Task 1: ParticleFilters.jl — BasicParticleFilter update function
echo "=== [1/3] ParticleFilters.jl: update ==="
cd "$MATCHER" && ageom ingest \
  "$TP/ParticleFilters.jl/src/basic.jl" \
  --class update \
  --output "$OUT/particle_filters/basic" \
  --llm-provider codex_cli \
  --trace \
  2>&1 | tee "$LOGS/particle_basic.log"

# Task 2a: filter-rs — KalmanFilter struct
echo "=== [2/3] filter-rs: KalmanFilter ==="
cd "$MATCHER" && ageom ingest \
  "$TP/filter-rs/src/kalman/kalman_filter.rs" \
  --class KalmanFilter \
  --output "$OUT/kalman_filters/filter_rs" \
  --llm-provider codex_cli \
  --trace \
  2>&1 | tee "$LOGS/kalman_filter_rs.log"

# Task 2b: kalman_filters crate — StaticKalmanFilter
echo "=== [3/3] kalman_filters: StaticKalmanFilter ==="
cd "$MATCHER" && ageom ingest \
  "$TP/kalman_filters/src/filter.rs" \
  --class StaticKalmanFilter \
  --output "$OUT/kalman_filters/static_kf" \
  --llm-provider codex_cli \
  --trace \
  2>&1 | tee "$LOGS/kalman_static_kf.log"

echo "=== All done ==="
