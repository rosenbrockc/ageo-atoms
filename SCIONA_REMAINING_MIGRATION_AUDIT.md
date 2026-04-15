# Sciona Remaining Migration Audit

As of 2026-04-12, the following atom artifacts or provider-boundary tasks remain outstanding relative to the original `sciona-atoms*` migration plans.

This audit counts:
- atom families and leaves still present only in `ageoa`
- plan-level provider-boundary work that is still matcher-local

This audit does not count:
- shared support code such as `ghost`, `__pycache__`, `.partial`, package aggregators, or repo metadata
- legacy copies in `ageoa` when a corresponding provider-owned home already exists

## Summary

The atom-family split is mostly complete.

What is effectively done:
- general/core atom families, including the `numpy` and `scipy` utility surface, now live in `sciona-atoms`
- fintech atom migration is complete for the originally planned families
- robotics atom migration is complete for the originally planned `rust_robotics` and `pronto` surface
- ML atom migration is complete for the currently intended split

What remains is concentrated in:
- the deeper signal backlog
- the advanced molecular-docking remainder
- the two specialist physics families
- provider-boundary cleanup for expansion atoms and signal family assets

## Remaining By Provider

### General / Core

#### Remaining plan-level provider-boundary work
- move matcher-local general expansion runtimes/registries into `sciona-atoms`
- add provider-owned `src/sciona/atoms/expansion/...` and `src/sciona/probes/expansion/...` if these families are meant to live outside matcher
- finish source-of-truth cleanup for general family assets and matcher compatibility shims
- review whether any remaining matcher-local heuristic/tunable artifacts should become provider-owned

#### Atom-family migration already complete and not counted as remaining
- `conjugate_priors/beta_binom`
- `jax_advi/optimize_advi`
- `algorithms/graph.py`
- `algorithms/search.py`
- `kalman_filters/static_kf`
- `advancedvi`
- `bayes_rs`
- `belief_propagation/loopy_bp`
- `mcmc_foundational/*` current migrated set
- `particle_filters/basic`
- `kalman_filters/filter_rs`
- `algorithmic/divide_and_conquer/sorting`
- general `numpy/*`
- general `scipy/*`

### Signal

#### Remaining biosppy backlog
- `biosppy/ecg_segmenters_deep`
- `biosppy/ecg_zz2018`
- `biosppy/ecg_zz2018_d12`
- `biosppy/online_filter`
- `biosppy/online_filter_codex`
- `biosppy/online_filter_sonnet`
- `biosppy/online_filter_v2`
- `biosppy/svm`
- `biosppy/svm_codex`
- `biosppy/svm_proc`

#### Remaining e2e_ppg backlog
- `e2e_ppg/gan_reconstruction.py`
- `e2e_ppg/heart_cycle.py`
- `e2e_ppg/template_matching.py`

#### Remaining plan-level provider-boundary work
- move signal-specific expansion runtimes/registries out of matcher:
  - `signal_event_rate`
  - `signal_filter`
  - `signal_transform`
  - `signal_detect_measure`
  - likely `graph_signal_processing`
- make `sciona-atoms-signal` the single source of truth for signal family assets
- add the missing provider-owned `signal_detect_measure` family asset
- remove or intentionally retain duplicate `signal_event_rate` assets outside the signal provider

#### Already migrated and not counted as remaining
- `biosppy/ecg`
- `heartpy`
- `neurokit2`
- `e2e_ppg/reconstruction`
- `e2e_ppg/kazemi_wrapper`
- `e2e_ppg/kazemi_wrapper_d12`
- `signal_event_rate` heuristic asset exists in the signal provider, but source-of-truth cleanup is still incomplete

### Bio

#### Remaining molecular_docking backlog
- `molecular_docking/mwis_sa`
- `molecular_docking/quantum_solver`
- `molecular_docking/quantum_solver_d12`

#### Already migrated and not counted as remaining
- `alphafold`
- `hPDB`
- `mint/*` current migrated set
- `molecular_docking/build_interaction_graph`
- `molecular_docking/greedy_mapping`
- `molecular_docking/greedy_mapping_d12`
- `molecular_docking/build_complementary`
- `molecular_docking/add_quantum_link`
- `molecular_docking/greedy_subgraph`
- `molecular_docking/map_to_udg`
- `molecular_docking/minimize_bandwidth`

### Fintech

No confirmed remaining atom backlog from the original provider plan.

#### Already migrated and not counted as remaining
- `quant_engine`
- `quantfin`
- `hftbacktest`
- `institutional_quant_engine/*` planned execution, pricing, and risk leaves
- `institutional_quant_engine/almgren_chriss.py`

### ML

No confirmed remaining atom backlog from the current split audit.

#### Remaining plan-level review
- decide whether any matcher-local `neural_network` expansion support should move to `sciona-atoms-ml`

#### Current migrated surface includes
- `datadriven`
- `sklearn/images`

### Physics

#### Remaining families
- `jFOF`
- `pasqal`

#### Remaining plan-level cleanup
- final boundary cleanup for any broad math/state-estimation helpers discovered while migrating those families

#### Already migrated and not counted as remaining
- `tempo_jl/*` current migrated set
- `skyfield`
- `astroflow`
- `pulsar`
- `pulsar_folding`

### Robotics

No confirmed remaining atom backlog from the original provider plan.

#### Already migrated and not counted as remaining
- `rust_robotics/*` planned surface, including:
  - `bicycle_kinematic`
  - `longitudinal_dynamics`
  - `n_joint_arm_2d`
- `pronto/*` planned surface, including:
  - backlash / blip filters
  - dynamic stance estimators
  - EKF / smoother wrappers
  - foot contact / leg odometer / yaw lock

## Residual Risk Notes

The remaining backlog is not uniform.

Lower-risk residuals:
- signal subfamilies that are structurally similar to already migrated wrappers but still need provider packaging
- `molecular_docking/mwis_sa` if it proves to be another self-contained leaf

Higher-risk residuals:
- biosppy remainder because several subfamilies are alternate-codegen or legacy variants
- `e2e_ppg/gan_reconstruction.py`, `heart_cycle.py`, `template_matching.py` because they are more dependency-sensitive than the already migrated wrappers
- `molecular_docking/quantum_solver` and `quantum_solver_d12` because they likely carry more runtime assumptions
- `jFOF` and `pasqal` because they are specialist physics families without the clean pilot-style slice pattern used in the earlier waves
- expansion-family migration because it is a provider-boundary cleanup across matcher and provider repos, not just a leaf copy

## Recommended Reading Order

To resume migration work efficiently:
1. finish the remaining signal atom backlog
2. finish the remaining bio and physics family backlogs in parallel
3. close the provider-boundary cleanup for signal/general expansion atoms and signal family assets
4. finish the ML expansion-support review if it is still intended to move
