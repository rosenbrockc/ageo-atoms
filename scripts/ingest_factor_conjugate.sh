#!/usr/bin/env bash
set -euo pipefail

export AGEOM_INGESTER_MAX_DEPTH=12
MATCHER=/Users/conrad/personal/ageo-matcher
TP=/Users/conrad/personal/ageo-atoms/third_party
OUT=/Users/conrad/personal/ageo-atoms/ageoa
LOGS=/Users/conrad/personal/ageo-atoms/logs/ingest

mkdir -p "$OUT/belief_propagation" "$OUT/conjugate_priors" "$OUT/bayes_rs" "$LOGS"

# Task 1: Belief-Propagation — loopy_belief_propagation class
echo "=== [1/4] Belief-Propagation: loopy_belief_propagation ==="
cd "$MATCHER" && ageom ingest \
  "$TP/Belief-Propagation/belief_propagation.py" \
  --class loopy_belief_propagation \
  --output "$OUT/belief_propagation/loopy_bp" \
  --llm-provider codex_cli \
  --trace \
  2>&1 | tee "$LOGS/belief_propagation.log"

# Task 2a: ConjugatePriors.jl — posterior_canon in beta_binom.jl
echo "=== [2/4] ConjugatePriors.jl: posterior_canon (beta_binom) ==="
cd "$MATCHER" && ageom ingest \
  "$TP/ConjugatePriors.jl/src/beta_binom.jl" \
  --class posterior_canon \
  --output "$OUT/conjugate_priors/beta_binom" \
  --llm-provider codex_cli \
  --trace \
  2>&1 | tee "$LOGS/conjugate_beta_binom.log"

# Task 2b: ConjugatePriors.jl — posterior_canon in normal.jl
echo "=== [3/4] ConjugatePriors.jl: posterior_canon (normal) ==="
cd "$MATCHER" && ageom ingest \
  "$TP/ConjugatePriors.jl/src/normal.jl" \
  --class posterior_canon \
  --output "$OUT/conjugate_priors/normal" \
  --llm-provider gemini_cli \
  --trace \
  2>&1 | tee "$LOGS/conjugate_normal.log"

# Task 2c: bayes crate — Bernoulli conjugate fit
echo "=== [4/4] bayes crate: Bernoulli (conjugate fit) ==="
cd "$MATCHER" && ageom ingest \
  "$TP/bayes/src/prob/bernoulli.rs" \
  --class Bernoulli \
  --output "$OUT/bayes_rs/bernoulli" \
  --llm-provider codex_cli \
  --trace \
  2>&1 | tee "$LOGS/bayes_bernoulli.log"

echo "=== All done ==="
