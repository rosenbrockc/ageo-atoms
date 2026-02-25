#!/usr/bin/env bash
set -euo pipefail

export AGEOM_INGESTER_MAX_DEPTH=12
MATCHER=/Users/conrad/personal/ageo-matcher
TP=/Users/conrad/personal/ageo-atoms/third_party
OUT=/Users/conrad/personal/ageo-atoms/ageoa
LOGS=/Users/conrad/personal/ageo-atoms/logs/ingest

mkdir -p "$OUT/conjugate_priors/beta_binom" "$OUT/conjugate_priors/normal" "$OUT/bayes_rs/bernoulli"

# ConjugatePriors.jl — beta_binom (procedural with dummy class name)
echo "=== [1/3] ConjugatePriors.jl: beta_binom (procedural) ==="
cd "$MATCHER" && ageom ingest \
  "$TP/ConjugatePriors.jl/src/beta_binom.jl" \
  --class posterior_canon \
  --procedural \
  --output "$OUT/conjugate_priors/beta_binom" \
  --llm-provider codex_cli \
  --trace \
  2>&1 | tee "$LOGS/conjugate_beta_binom.log"

# ConjugatePriors.jl — normal.jl (extract posterior_canon function)
echo "=== [2/3] ConjugatePriors.jl: normal (posterior_canon) ==="
cd "$MATCHER" && ageom ingest \
  "$TP/ConjugatePriors.jl/src/normal.jl" \
  --class posterior_canon \
  --output "$OUT/conjugate_priors/normal" \
  --llm-provider codex_cli \
  --trace \
  2>&1 | tee "$LOGS/conjugate_normal.log"

# bayes crate — Bernoulli struct with conjugate fit
echo "=== [3/3] bayes crate: Bernoulli ==="
cd "$MATCHER" && ageom ingest \
  "$TP/bayes/src/prob/bernoulli.rs" \
  --class Bernoulli \
  --output "$OUT/bayes_rs/bernoulli" \
  --llm-provider codex_cli \
  --trace \
  2>&1 | tee "$LOGS/bayes_bernoulli.log"

echo "=== All done ==="
