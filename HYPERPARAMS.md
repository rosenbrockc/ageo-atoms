# Hyperparameter Audit Plan

This document describes how to audit the atoms in `ageo-atoms` for safe, meaningful hyperparameter exposure to downstream optimizers such as `ageom optimize`.

## Goal

Produce a repository-wide, manually reviewed inventory of:
- which atoms have meaningful tunable parameters
- which parameters are safe to expose
- valid domains and defaults for those parameters
- which atoms should never be tuned

The output of this audit should support a general optimization framework, not benchmark-specific tuning hacks.

## Scope

This plan covers atoms under:
- [ageoa](/Users/conrad/personal/ageo-atoms/ageoa)

Representative families currently present include:
- signal / biosignals
- probabilistic inference
- filtering / state estimation
- finance / quant
- astronomy / ephemeris
- robotics / controls
- numerical scipy / numpy wrappers

## Principle

Do not infer tunable parameters mechanically from function signatures.

Manual review is required because many arguments are:
- required data inputs
- structural invariants
- library plumbing
- implementation details that should stay fixed

Only expose parameters that:
- materially affect algorithm behavior
- have stable semantics across datasets
- can be bounded safely
- do not invalidate the atom contract

## Deliverables

### 1. Audit manifest

Create a machine-readable manifest, for example:
- `data/hyperparams/manifest.json`

Generate a query index from that manifest:
- `data/hyperparams/manifest.sqlite`

Each atom entry should record:
- module path
- atom name
- family
- audit status
- tunable parameters
- blocked / unsafe parameters
- reviewer notes
- range/source provenance

### 2. Parameter schema per atom

For each approved tunable parameter, record:
- `name`
- `kind`: `int | float | categorical | bool`
- `default`
- `min_value`
- `max_value`
- `step`
- `log_scale`
- `choices`
- `constraints`
- `semantic_role`
- `safe_to_optimize`
- `reason`
- `range_source`
- `source_reference`
- `source_confidence`

### 3. Audit coverage report

Create a summary report showing:
- total atoms audited
- atoms with tunables
- atoms with no safe tunables
- atoms blocked pending deeper review
- coverage by family

## Audit Categories

Each atom should be assigned one of:
- `approved`
  safe tunables identified
- `fixed`
  atom should not expose tunables
- `blocked`
  needs deeper domain review
- `deprecated`
  not worth adding tuning support

## Review Checklist

Use source evidence in this order:
1. the wrapper in `ageoa/.../atoms.py`
2. the original implementation in `third_party/`
3. official API documentation / docstrings
4. cited papers / method docs
5. targeted web search for documented heuristics, defaults, warnings, and safe ranges

Do not approve a parameter range from wrapper intuition alone if the upstream code or docs define the intended usage.

For each atom:
1. Read the implementation in `atoms.py` or the module wrapper.
2. Read the corresponding implementation in `third_party/` when available.
3. Read witnesses and nearby docs if needed.
4. Read official docs and cited papers for default heuristics and parameter semantics.
5. Use targeted web search if the local sources do not clearly justify safe ranges.
6. Identify behavior-shaping arguments.
7. Separate true hyperparameters from:
   - inputs
   - outputs
   - state variables
   - internal constants that preserve correctness
8. Decide whether the atom is:
   - worth tuning
   - safe to tune
   - too brittle or too opaque
9. If tunable, define a bounded domain only when primary sources justify it.
10. Record provenance for every approved range and default.
11. Record the result in the manifest.

## Recommended Family Order

Start with families most likely to benefit from tuning and most likely to be used in end-to-end optimization.

### Phase 1: Signal / biosignal families

Priority paths:
- [ageoa/biosppy](/Users/conrad/personal/ageo-atoms/ageoa/biosppy)
- [ageoa/neurokit2](/Users/conrad/personal/ageo-atoms/ageoa/neurokit2)
- [ageoa/heartpy](/Users/conrad/personal/ageo-atoms/ageoa/heartpy)
- [ageoa/e2e_ppg](/Users/conrad/personal/ageo-atoms/ageoa/e2e_ppg)
- [ageoa/scipy/signal.py](/Users/conrad/personal/ageo-atoms/ageoa/scipy/signal.py)
- [ageoa/scipy/signal_v2](/Users/conrad/personal/ageo-atoms/ageoa/scipy/signal_v2)

Likely tunables:
- filter cutoffs
- filter order
- window sizes
- threshold multipliers
- refractory periods
- smoothing windows

### Phase 2: State estimation / filtering

Priority paths:
- [ageoa/kalman_filters](/Users/conrad/personal/ageo-atoms/ageoa/kalman_filters)
- [ageoa/particle_filters](/Users/conrad/personal/ageo-atoms/ageoa/particle_filters)
- [ageoa/rust_robotics](/Users/conrad/personal/ageo-atoms/ageoa/rust_robotics)

Likely tunables:
- process noise scales
- measurement noise scales
- resampling thresholds
- gating thresholds

### Phase 3: Probabilistic inference

Priority paths:
- [ageoa/mcmc_foundational](/Users/conrad/personal/ageo-atoms/ageoa/mcmc_foundational)
- [ageoa/advancedvi](/Users/conrad/personal/ageo-atoms/ageoa/advancedvi)
- [ageoa/jax_advi](/Users/conrad/personal/ageo-atoms/ageoa/jax_advi)
- [ageoa/conjugate_priors](/Users/conrad/personal/ageo-atoms/ageoa/conjugate_priors)

Likely tunables:
- step sizes
- mass matrix / adaptation settings
- particle counts
- optimization learning rates
- convergence tolerances

These need especially careful review because some parameters affect stability more than task quality.

### Phase 4: Numerical wrappers

Priority paths:
- [ageoa/scipy](/Users/conrad/personal/ageo-atoms/ageoa/scipy)
- [ageoa/numpy](/Users/conrad/personal/ageo-atoms/ageoa/numpy)

Many of these will likely be `fixed`, not `approved`, because they are general-purpose library wrappers rather than algorithm-level tunables.

### Phase 5: Domain-specific long tail

Examples:
- finance
- astronomy
- docking
- pulsar
- molecular

These should be audited after the higher-value families above.

## Family-Specific Heuristics

### Good hyperparameter candidates

- thresholds
- window sizes
- cutoff frequencies
- smoothing strengths
- search depth / iteration limits
- regularization scales
- tolerances

### Usually not good candidates

- raw input arrays
- timestamps
- identifiers
- file paths
- booleans used only for API mode selection unless semantically important
- values that merely toggle implementation plumbing

### High-risk candidates

These require extra review:
- parameters that can violate mathematical assumptions
- parameters that change output type/shape
- parameters that invalidate witness logic
- categorical choices that should instead be modeled as separate atoms

## Suggested Repo Additions

Recommended files:
- `data/hyperparams/manifest.json`
- `data/hyperparams/manifest.sqlite`
- `scripts/build_hyperparams_manifest.py`
- `scripts/report_hyperparams.py`
- `tests/test_hyperparams_manifest.py`

Source-of-truth rule:
- edit `manifest.json` through reviewed audit work
- regenerate `manifest.sqlite` from the JSON manifest
- do not hand-edit the SQLite file

Recommended script behaviors:
- enumerate atom modules under `ageoa/`
- join audit annotations with filesystem inventory
- flag unaudited atoms
- generate coverage summaries

## Manual Audit Workflow

For each family:
1. Enumerate atoms.
2. Triage obvious `fixed` atoms.
3. Review implementation and witnesses.
4. Draft candidate parameter schema.
5. Mark unsafe or blocked parameters explicitly.
6. Add audit record.
7. Run validation on manifest format.

## Acceptance Criteria

The audit is complete enough for optimizer integration when:
- every atom in `ageoa/` has an audit status
- every approved tunable parameter has a bounded schema
- blocked atoms are explicitly labeled with reasons
- a coverage report exists by family

## Non-Goals

- auto-generating tunables from signatures
- tuning everything in one pass
- exposing every optional argument
- adding benchmark-specific defaults to atom metadata
- inventing search bounds without upstream support

## Recommended First Milestone

Complete a full audit for:
- `biosppy`
- `neurokit2`
- `heartpy`
- `scipy.signal`

That gives the downstream optimizer a high-value first set of tunable atoms for signal-processing tasks while establishing the general audit process for the rest of the repo.
