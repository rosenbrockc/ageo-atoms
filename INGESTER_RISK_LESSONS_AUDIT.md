# Ingester Risk-Lessons Audit

This document audits whether the main lessons learned during `ageo-atoms`
trust-debt remediation have been pushed back into `../ageo-matcher`.

It is not a rehash of the hardening plans. It is a cross-repo check:

- what `ageo-atoms` had to repair manually
- what the matcher now covers directly
- what remains only partially absorbed
- what is intentionally outside the ingester boundary

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Audit Result

Short version:

- the main **code-generation lessons** from the risk audit have been absorbed
  into the ingester
- the ingester now covers the important architectural failures that repeatedly
  produced risky atoms
- the main remaining gaps are **rollout breadth**, **cross-repo traceability**,
  and **audit-only concerns** that do not belong entirely inside the matcher

## Coverage Matrix

| Lesson from trust-debt remediation | `ageo-atoms` evidence | Matcher coverage status | Matcher evidence | Remaining gap |
| --- | --- | --- | --- | --- |
| Wrappers drift from upstream signatures | [ecg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/ecg_detectors.py), [ppg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/ppg_detectors.py), [emg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/emg_detectors.py) | `covered` | [emitter.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/emitter.py), [test_ingester_emitter.py](/Users/conrad/personal/ageo-matcher/tests/test_ingester_emitter.py) | smoke/regression breadth can still grow, but the core signature hardening landed |
| Wrappers guess the wrong return shape or unwrap results incorrectly | [ecg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/ecg_detectors.py), [images/atoms.py](/Users/conrad/personal/ageo-atoms/ageoa/sklearn/images/atoms.py) | `partially_covered` | [chunker.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/chunker.py), [emitter.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/emitter.py), [return_shapes.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/return_shapes.py), [test_ingester_return_shapes.py](/Users/conrad/personal/ageo-matcher/tests/test_ingester_return_shapes.py) | conservative fallback is in place, but explicit structured-return knowledge is still intentionally narrow |
| Witness / decorator emission can be syntactically valid but semantically wrong | repaired witness/decorator issues during remediation, especially detector families | `covered` | [emitter.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/emitter.py), [test_ingester_emitter.py](/Users/conrad/personal/ageo-matcher/tests/test_ingester_emitter.py) | no major architecture gap remains here |
| Package scoping was too fragmented, producing one-symbol package sprawl | [sklearn/images](/Users/conrad/personal/ageo-atoms/ageoa/sklearn/images), [tempo_jl/offsets](/Users/conrad/personal/ageo-atoms/ageoa/tempo_jl/offsets) | `partially_covered` | [cli.py](/Users/conrad/personal/ageo-matcher/sciona/cli.py), [ingest_cmds.py](/Users/conrad/personal/ageo-matcher/sciona/commands/ingest_cmds.py), [monitor.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/monitor.py), [test_ingest_output_scope.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_output_scope.py) | grouped publication is first-class now, but true multi-target grouped ingest is still not a finished production path |
| Bad outputs need deterministic rejection before they land in `ageo-atoms` | [runtime_probes.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/runtime_probes.py), [test_audit_runtime_probes.py](/Users/conrad/personal/ageo-atoms/tests/test_audit_runtime_probes.py) | `partially_covered` | [smoke.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/smoke.py), [monitor.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/monitor.py), [test_ingest_smoke.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_smoke.py) | the smoke gate works, but matcher-side allowlist coverage is still much smaller than the repo audit surface |
| Learned failures should become matcher regressions, not just repo memory | [INGESTER_ROLLOUT_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_ROLLOUT_PLAN.md), [SKLEARN_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/SKLEARN_EXECUTION_PLAN.md) | `partially_covered` | [regression_harness.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/regression_harness.py), [test_ingest_regression_harness.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_regression_harness.py), [tests/fixtures/ingest_regression/sklearn_grouped_images/source.py](/Users/conrad/personal/ageo-matcher/tests/fixtures/ingest_regression/sklearn_grouped_images/source.py), [tests/fixtures/ingest_regression/detector_structured_output/source.py](/Users/conrad/personal/ageo-matcher/tests/fixtures/ingest_regression/detector_structured_output/source.py) | there is now a real regression corpus, but it is still small relative to the breadth of repairs performed in `ageo-atoms` |
| Decomposition must reject bad child graphs and fake orchestration cycles | cyclic `grid_to_graph` failure during sklearn ingestion | `covered` | [chunker.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/chunker.py), [test_ingester_chunker.py](/Users/conrad/personal/ageo-matcher/tests/test_ingester_chunker.py) | no immediate gap |

## Lessons That Are Covered Well

### 1. Signature fidelity is now matcher responsibility

This was one of the biggest recurring manual cleanup themes in
`ageo-atoms`. The ingester now preserves:

- positional-only markers
- keyword-only structure
- `*args`
- `**kwargs`
- default-preserving public signatures

Primary matcher evidence:

- [emitter.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/emitter.py)
- [test_ingester_emitter.py](/Users/conrad/personal/ageo-matcher/tests/test_ingester_emitter.py)

Conclusion:

- this lesson is absorbed at the architecture level
- future drift here should be treated as a regression, not expected cleanup

### 2. Witness and decorator emission is no longer a soft spot

The ingester now emits symbol-based witness registration and handles variadic
surfaces more carefully, including witness-side tuple/dict surrogates instead
of fake scalar assumptions.

Primary matcher evidence:

- [emitter.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/emitter.py)
- [test_ingester_emitter.py](/Users/conrad/personal/ageo-matcher/tests/test_ingester_emitter.py)

Conclusion:

- this lesson is absorbed
- remaining witness issues would likely be family-specific rather than a known
  global emitter defect

### 3. Grouped output is now a real matcher concept

The ingester no longer assumes one-symbol-per-output-dir as the only sane
publication mode. `family` scope is explicit and recorded in monitor artifacts.

Primary matcher evidence:

- [cli.py](/Users/conrad/personal/ageo-matcher/sciona/cli.py)
- [ingest_cmds.py](/Users/conrad/personal/ageo-matcher/sciona/commands/ingest_cmds.py)
- [monitor.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/monitor.py)
- [test_ingest_output_scope.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_output_scope.py)

Primary `ageo-atoms` evidence:

- [sklearn/images](/Users/conrad/personal/ageo-atoms/ageoa/sklearn/images)
- [tempo_jl/offsets](/Users/conrad/personal/ageo-atoms/ageoa/tempo_jl/offsets)

Conclusion:

- the repo-level scoping lesson is absorbed
- the remaining gap is rollout depth, not missing architecture

### 4. The ingester now rejects some obviously bad outputs before publication

The smoke gate is now integrated into ingest-time publication flow and recorded
in ingest monitor state.

Primary matcher evidence:

- [smoke.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/smoke.py)
- [ingest_cmds.py](/Users/conrad/personal/ageo-matcher/sciona/commands/ingest_cmds.py)
- [monitor.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/monitor.py)
- [test_ingest_smoke.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_smoke.py)

Conclusion:

- the lesson is absorbed in principle
- the remaining issue is limited allowlist breadth

## Lessons That Are Only Partially Covered

### 1. Structured return handling is fixed conservatively, not comprehensively

The most important correction landed: the ingester no longer broadly invents
dict-field or tuple-slot extraction. It defaults to conservative passthrough,
and explicit structured extraction is now allowlisted.

That is the correct direction. But the explicit knowledge layer is still tiny.

Current matcher evidence:

- [return_shapes.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/return_shapes.py)

Current scope:

- one allowlisted detector case only: `PeakDetector.detect`

Conclusion:

- this lesson is only partially absorbed
- the dangerous broad heuristic was removed, which was the important fix
- but the repo still holds manual repairs whose structured-return knowledge is
  not yet mirrored matcher-side

### 2. Smoke coverage is still much narrower than the audit oracle

The matcher smoke layer now covers:

- grouped sklearn image helpers
- narrow numerical canaries like `fft`

But `ageo-atoms` runtime/parity coverage is much broader, especially for
families like BioSPPy detectors.

Primary `ageo-atoms` evidence:

- [runtime_probes.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/runtime_probes.py)

Primary matcher evidence:

- [smoke.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/smoke.py)
- [test_ingest_smoke.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_smoke.py)

Conclusion:

- the lesson is only partially absorbed
- the architecture exists, but matcher-side probe coverage still trails the
  repo-local acceptance oracle by a wide margin

### 3. Regression coverage exists, but not yet at the same resolution as the repair history

The curated regression harness is a major improvement. It now carries grouped
family and detector-structured-output cases, which directly reflect important
recent lessons.

Primary matcher evidence:

- [regression_harness.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/regression_harness.py)
- [test_ingest_regression_harness.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_regression_harness.py)

Conclusion:

- this lesson is partially absorbed
- the matcher has stopped relying purely on memory
- but there is still no explicit one-to-one trace from many major `ageo-atoms`
  repair families to dedicated matcher regression fixtures

## Lessons That Are Intentionally Outside The Ingester Boundary

These are real trust lessons from the `ageo-atoms` audit, but they are not
fully matcher problems.

### 1. Review-basis and provenance completeness

The risk audit penalizes missing review basis, weak provenance, and incomplete
authoritative-source capture. Those are real trust issues, but they are only
partially ingester-adjacent.

Primary `ageo-atoms` evidence:

- [AUDIT_INGEST.md](/Users/conrad/personal/ageo-atoms/AUDIT_INGEST.md)
- [data/audit_reviews](/Users/conrad/personal/ageo-atoms/data/audit_reviews)

Conclusion:

- not a missing matcher feature
- this belongs mostly to the audit/review system and metadata pipeline

### 2. Broad parity coverage

The risk audit also penalizes missing parity evidence. That is a repo quality
goal, but it should not be copied wholesale into ingest-time matcher logic.

Conclusion:

- matcher-side smoke probes should stay narrow
- parity remains primarily an `ageo-atoms` validation concern

### 3. Module CDGs and upstream mapping inventory

Many trust-debt repairs were simple but important metadata cleanups:

- add module-level CDGs
- add upstream entries in [atom_manifest.yml](/Users/conrad/personal/ageo-atoms/scripts/atom_manifest.yml)

Those are real lessons, but they do not map cleanly to matcher behavior.

Conclusion:

- not a missing ingester refinement
- these remain repository-maintenance concerns

## Remaining Gaps Worth Addressing

These are the concrete areas where the audit still says “not fully absorbed.”

### Gap 1: Cross-repo traceability is weak

There is still no durable index answering:

- which `ageo-atoms` repair families caused which matcher change
- which matcher regression case protects each learned lesson

Recommendation:

- add a small crosswalk doc or JSON manifest mapping major repaired families to
  matcher tests and smoke probes

### Gap 2: Matcher smoke coverage trails the repair surface

The smoke gate architecture is present, but breadth is still small relative to
the families that were manually paid down in `ageo-atoms`.

Recommendation:

- selectively extend matcher smoke coverage to a few more deterministic
  detector-style and numerical families
- do not try to mirror the whole `ageo-atoms` runtime probe library

### Gap 3: Structured-return knowledge remains very narrow

This is acceptable by design today, but it means some repaired structured
wrapper families still do not have matcher-native knowledge behind them.

Recommendation:

- add only a few more explicit allowlisted cases where:
  - the repo already proved the shape by repair and tests
  - the return fields are stable and unambiguous

### Gap 4: Grouped output is first-class, but grouped ingest ergonomics still lag

The matcher now publishes grouped outputs correctly, but there is still a
practical gap between:

- “I can publish a grouped family”
- “the default ingest workflow naturally produces the grouped family I want”

Recommendation:

- if grouped family ingest becomes a common path, add higher-level matcher
  support for intentionally building a family package from several targets

## Overall Conclusion

The ingester refinement work did what it needed to do.

The main architectural lessons learned from risky atoms are now present in
`../ageo-matcher`:

- signature fidelity
- conservative output handling
- witness/decorator hardening
- grouped output scope
- ingest-time smoke validation
- curated regression coverage
- narrow explicit structured-return knowledge

What remains is not another major hardening phase. It is mostly:

- rollout breadth
- more explicit cross-repo traceability
- a few deliberately narrow follow-on allowlists and smoke cases

So the audit result is:

- **core lessons absorbed**
- **rollout still incomplete**
- **no new major ingester architecture gap discovered**
