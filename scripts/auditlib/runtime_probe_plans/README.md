# Runtime Probe Plan Contribution Guide

This package owns the family-scoped runtime probe registries. Phase B is about making family work parallel by default: one worker, one family module, no shared-file churn unless integration is required.

## Where New Plans Belong

Add new probe plans in the matching family module under `scripts/auditlib/runtime_probe_plans/`.

Examples:
- `biosppy.py`
- `mcmc_foundational.py`
- `quant_engine.py`
- `neurokit2.py`

If a family needs a helper that is only used by that family, keep it in the same module or a sibling helper file owned by that family.

## Helper Rules

- Family-local helpers stay with the family module that uses them.
- Shared helpers belong in `runtime_probes_core.py` or `runtime_probe_assertions.py`, but only if they are reused across multiple families.
- Do not move a one-off validator into shared code just because it is convenient for one worker.
- Do not add temporary helpers to `runtime_probes.py` or `runtime_probes_core.py` during family remediation.

## File Ownership

Family workers normally own only:
- `scripts/auditlib/runtime_probe_plans/<family>.py`
- focused tests for that family, when needed

Main-agent-only files during parallel remediation:
- `scripts/auditlib/runtime_probes.py`
- `scripts/auditlib/runtime_probes_core.py`
- `scripts/auditlib/runtime_probe_assertions.py`
- `scripts/auditlib/runtime_probe_plans/__init__.py`
- broad audit test files that touch many families

## Test Guidance

Prefer a family-specific selector before running the full audit suite.

Suggested pattern:
```bash
PYTHONPATH=/Users/conrad/personal/ageo-atoms/scripts \
/Users/conrad/personal/ageo-matcher/.venv/bin/python -m pytest ../ageo-atoms/tests/test_audit_runtime_probes.py -k '<family>' -q
```

If a family has a dedicated test file, run that first. Otherwise use the matching `test_runtime_probe_passes_for_<family>...` cases from `tests/test_audit_runtime_probes.py`.

## Current Worker Packets

### Wave 1
Low-coupling and already domain-local.

| Family module | Main selector | Notes |
| --- | --- | --- |
| `neurokit2.py` | `-k neurokit2` | Good for small isolated additions or wrapper parity fixes. |
| `rust_robotics.py` | `-k rust_robotics` | Keep dynamics helpers family-local. |
| `belief_propagation.py` | `-k belief_propagation` | Message-passing probes only. |
| `particle_filter_and_pasqal.py` | `-k particle_filter` or `-k pasqal` | Split by subfamily when assigning worker packets. |

### Wave 2
Moderate-size numerical families.

| Family module | Main selector | Notes |
| --- | --- | --- |
| `kalman_filter.py` | `-k kalman_filter` | State-update and estimator probes. |
| `quant_engine.py` | `-k quant_engine` | Shared state helpers should stay family-local unless reused. |
| `advancedvi_and_iqe.py` | `-k advancedvi` or `-k iqe` | Useful for separate worker packets if changes do not overlap. |
| `conjugate_priors_and_small_mcmc.py` | `-k conjugate_priors` or `-k small_mcmc` | Keep the smaller Bayesian helpers together. |

### Wave 3
Large and more coupled families.

| Family module | Main selector | Notes |
| --- | --- | --- |
| `biosppy.py` | `-k biosppy` | Split by ECG, PPG, ABP, EDA, EMG, and PCG subsections when practical. |
| `pronto.py` | `-k pronto` | Multiple wrapper families live here. |
| `molecular_docking.py` | `-k molecular_docking` | Keep domain-specific scoring helpers close to the family. |
| `quantfin.py` | `-k quantfin` | Higher fan-out, so keep packets narrow. |
| `hftbacktest_and_ingest.py` | `-k hftbacktest` or `-k ingest` | Avoid touching shared registry glue from worker lanes. |
| `mcmc_foundational.py` | `-k mcmc_foundational` or `-k advancedhmc` or `-k mini_mcmc` | Prefer one subsection per worker packet. |

### Integration Only
Already split or mostly stable. Do not assign as new Phase B worker packets unless there is a specific regression to fix.

| Family module | Main selector | Notes |
| --- | --- | --- |
| `search.py` | `-k binary_search` | Small shared baseline module. |
| `sorting.py` | `-k sorting` | Keep as a narrow integration target. |
| `numpy_basic.py` | `-k numpy` | Utility-style probes, usually not a worker packet. |
| `scipy_basic.py` | `-k scipy` | Utility-style probes, usually not a worker packet. |
| `scipy_sparse_graph.py` | `-k sparse_graph` | Already isolated enough for integration review only. |
| `scipy_stats_integrate.py` | `-k integrate` or `-k stats` | Keep for follow-up regressions, not routine packet work. |
| `numpy_fft_v2.py` | `-k fft_v2` | Already split and best handled as a small follow-up lane. |
| `numpy_search_sort_v2.py` | `-k search_sort_v2` | Usually a focused regression lane, not a new packet. |
| `scipy_optimize_v2.py` | `-k optimize_v2` | Follow-up lane only unless it regresses. |

## Registry And Integration

- `__init__.py` owns the import surface for family registries.
- `runtime_probes.py` and `runtime_probes_core.py` own shared assembly and execution behavior.
- Family workers should not edit registry assembly unless specifically assigned to an integration pass.
- If many family lanes land at once, batch registry-import updates in one main-agent pass.

## Practical Rule

If a change can be made without reading beyond one family module and its focused tests, it belongs in a worker packet. If it needs registry assembly, cross-family helper reuse, or broad runtime behavior, it stays with the main agent.
