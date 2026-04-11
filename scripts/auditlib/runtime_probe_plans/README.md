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
| `neurokit2.py` | `-k neurokit2` | Packet file: `tests/test_runtime_probe_neurokit2_family.py`. |
| `rust_robotics.py` | `-k rust_robotics` | Packet file: `tests/test_runtime_probe_rust_robotics_family.py`. |
| `belief_propagation.py` | `-k belief_propagation` | Packet file: `tests/test_runtime_probe_belief_propagation_family.py`. |
| `particle_filter_and_pasqal.py` | `-k particle_filter` or `-k pasqal` | Packet file: `tests/test_runtime_probe_particle_filter_and_pasqal_family.py`; mixed parity, so keep usage-equivalent helpers local to the packet file. |

### Wave 2
Moderate-size numerical families.

| Family module | Main selector | Notes |
| --- | --- | --- |
| `kalman_filter.py` | `-k kalman_filter` | Packet file: `tests/test_runtime_probe_kalman_filter_family.py`. |
| `quant_engine.py` | `-k quant_engine` | Packet file: `tests/test_runtime_probe_quant_engine_family.py`. |
| `advancedvi_and_iqe.py` | `-k advancedvi` or `-k iqe` | Packet file: `tests/test_runtime_probe_advancedvi_and_iqe_family.py`. |
| `conjugate_priors_and_small_mcmc.py` | `-k conjugate_priors` or `-k small_mcmc` | Packet file: `tests/test_runtime_probe_conjugate_priors_and_small_mcmc_family.py`. |

### Wave 3
Large and more coupled families.

| Family module | Main selector | Notes |
| --- | --- | --- |
| `biosppy.py` | `-k biosppy` | Packet files: `tests/test_runtime_probe_biosppy_ecg_family.py` and `tests/test_runtime_probe_biosppy_signal_family.py`. |
| `pronto.py` | `-k pronto` | Packet file: `tests/test_runtime_probe_pronto_family.py`. |
| `molecular_docking.py` | `-k molecular_docking` | Packet file: `tests/test_runtime_probe_molecular_docking_family.py`. |
| `quantfin.py` | `-k quantfin` | Packet file: `tests/test_runtime_probe_quantfin_family.py`. |
| `hftbacktest_and_ingest.py` | `-k hftbacktest` or `-k ingest` | Packet file: `tests/test_runtime_probe_hftbacktest_and_ingest_family.py`. |
| `mcmc_foundational.py` | `-k mcmc_foundational` or `-k advancedhmc` or `-k mini_mcmc` | Packet file: `tests/test_runtime_probe_mcmc_foundational_family.py`. |
| `foundation.py` | `-k e2e_ppg` or `-k mint` or `-k alphafold` | Packet file: `tests/test_runtime_probe_foundation_family.py` for the remaining mixed foundation-style surfaces. |

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

## Residual Audit-Only Cases

Keep these in the broad audit suite unless they regress or grow large enough to justify a dedicated packet:
- small baseline probes in `search.py`, `sorting.py`, `numpy_basic.py`, `scipy_basic.py`, and `scipy_sparse_graph.py`
- narrow optimizer and vectorized helper lanes in `numpy_fft_v2.py`, `numpy_search_sort_v2.py`, and `scipy_optimize_v2.py`
- standalone or two-case probes such as `skyfield`, `pulsar`, `jax_advi`, `astroflow`, `datadriven`, and small generated wrappers

Practical rule: add a dedicated packet file only when the family meaningfully benefits from isolated worker ownership or from a selector-friendly verification target.

## Practical Rule

If a change can be made without reading beyond one family module and its focused tests, it belongs in a worker packet. If it needs registry assembly, cross-family helper reuse, or broad runtime behavior, it stays with the main agent.
