# Sciona Remaining Migration Execution Plan

This plan covers the remaining work after the completed first-wave provider split and the completed general `numpy` / `scipy` utility migration.

Goal:
- finish provider extraction from `ageoa`
- close the remaining provider-boundary cleanup left in matcher
- use parallel work where ownership is disjoint
- keep verification scoped to import-smoke and provider-level aggregate tests unless a slice requires deeper runtime validation

## Operating Constraints

- Each provider repo is an independent write target.
- Parallelization is best done by provider, not by file, to avoid merge conflicts and to keep verification simple.
- The remaining backlog is skewed toward heavier or more awkward families, so batch size should shrink as risk rises.
- Existing unrelated edits in `ageoa` must not be reverted during planning or execution.
- Worker writes should continue to use `/tmp` scratch repos, with reviewed copy-back into the real provider repos.

## Parallelization Model

### Safe to run in parallel
- one worker or execution lane per provider repo
- read-only exploration in parallel across any number of repos
- verification in parallel across distinct provider repos
- one boundary-cleanup lane in matcher only if it does not overlap with another active write lane

### Avoid parallelizing together
- multiple writers against the same provider repo in the same wave
- mixed heavy-runtime slices in the same repo unless they are already known to import cleanly
- cross-cutting refactors that change naming, packaging, or manifests simultaneously in multiple repos without a clear owner

## Recommended Waves

## Wave 1: Remaining provider atom families

### Signal
- migrate `biosppy/ecg_segmenters_deep`
- migrate `biosppy/ecg_zz2018`
- migrate `biosppy/ecg_zz2018_d12`
- migrate `biosppy/online_filter`
- migrate `biosppy/online_filter_codex`
- migrate `biosppy/online_filter_sonnet`
- migrate `biosppy/online_filter_v2`
- migrate `biosppy/svm`
- migrate `biosppy/svm_codex`
- migrate `biosppy/svm_proc`
- migrate `e2e_ppg/gan_reconstruction.py`
- migrate `e2e_ppg/heart_cycle.py`
- migrate `e2e_ppg/template_matching.py`

### Bio
- migrate `molecular_docking/mwis_sa`
- migrate `molecular_docking/quantum_solver`
- migrate `molecular_docking/quantum_solver_d12`

### Physics
- migrate `jFOF`
- migrate `pasqal`

### Parallelization
- run three lanes in parallel:
  - signal lane
  - bio lane
  - physics lane

### Verification
- targeted import smokes first
- then rerun the touched provider aggregate suites

## Wave 2: Signal / general provider-boundary cleanup

### Signal provider ownership
- move signal-specific expansion runtimes/registries out of matcher
- add provider-owned packages or assets for:
  - `signal_event_rate`
  - `signal_filter`
  - `signal_transform`
  - `signal_detect_measure`
  - likely `graph_signal_processing`
- move the missing `signal_detect_measure` family asset into `sciona-atoms-signal`
- remove duplicate or transitional signal family assets outside the signal provider once matcher compatibility is preserved

### General provider ownership
- move matcher-local general expansion runtimes/registries into `sciona-atoms`
- add provider-owned `src/sciona/atoms/expansion/...` and `src/sciona/probes/expansion/...` if these remain in scope
- finish family-asset source-of-truth cleanup and compatibility-shim reduction

### Parallelization
- keep this as two coordinated lanes:
  - signal-boundary lane
  - general-boundary lane
- do not mix either lane with another writer against the same repos

### Verification
- matcher:
  - registry resolution
  - family asset loading
  - compatibility-path coverage
- providers:
  - import smokes
  - probe record resolution
  - asset presence / metadata checks

## Wave 3: ML expansion-support decision

### ML
- review whether matcher-local `neural_network` expansion support is genuinely ML-owned
- if yes, migrate it to `sciona-atoms-ml`
- if no, explicitly mark it as matcher-local and close the plan item

### Parallelization
- can run in parallel with read-only verification elsewhere
- keep it separate from Wave 2 if it touches matcher packaging

## Execution Order Recommendation

Recommended global order:
1. Wave 1 with signal, bio, and physics in parallel
2. Wave 2 after the remaining provider atom backlog is reduced
3. Wave 3 last, unless the ML review blocks another matcher-boundary decision

Rationale:
- The remaining repo-local atom backlog is now cleanly separable by provider.
- The boundary-cleanup work is more cross-cutting and should happen after the heavy provider-local leaves are reduced.
- The ML expansion-support question is a policy/boundary decision, not a simple leaf migration.

## Worker / Permission Strategy

### Recommended workflow
1. refresh `/tmp` scratch clones for each touched provider repo
2. assign one worker per provider repo
3. have workers edit only their scratch repo
4. review worker output locally
5. copy approved changes back into the real provider repo with escalated commands
6. run provider verification in the real repo

### If worker writes are unreliable
Use sub-agents for:
- read-only sizing
- dependency/risk inspection
- test selection recommendations

Use direct execution for:
- scratch-lane rescue when a worker stalls
- cross-repo writes back into the real provider repos
- git initialization
- pushes
- any command that needs filesystem escalation outside the main writable root

### Important permission note
If a task needs writes outside the current writable root, escalation is needed on the actual writing command.

That means:
- spawning a worker first does not by itself grant broader filesystem access
- if the worker must write outside the sandbox, the worker would still need to request escalation for its own write command
- the practical reliable path remains:
  - worker writes in `/tmp`
  - main agent reviews
  - main agent copies back with escalation

## Completion Criteria

The migration is complete when:
- every intended atom family in `ageoa` has a corresponding home in one of the sibling `sciona-atoms*` repos or has been explicitly designated out of scope
- provider-boundary work called out in the original plans is either migrated or explicitly retained in matcher by decision
- each touched provider repo has passing import-smoke coverage for the migrated slices
- aggregate provider smoke suites pass for each touched provider
- `ageoa` is left only with intentionally retained support/runtime material, not provider-owned atom families
