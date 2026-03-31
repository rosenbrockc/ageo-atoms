# Ingester Final Coverage Audit

This document is the closeout audit for the ingester gap-closure phases that
followed the original
[INGESTER_RISK_LESSONS_AUDIT.md](/Users/conrad/personal/ageo-atoms/INGESTER_RISK_LESSONS_AUDIT.md).

It answers a narrower question than the original audit:

- after Phases 1 through 5, which rollout gaps were actually reduced
- which areas are now effectively covered
- which areas remain intentionally narrow rather than truly missing

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Result

The ingester refinement and gap-closure program is now in a good stopping
state.

Summary:

- the original architecture lessons remain covered
- the main rollout gaps were materially reduced
- no new major matcher architecture gap was discovered
- the remaining items are mostly intentional narrowness, not missing design

## What Closed Since The Original Audit

The original audit identified five remaining gap areas:

1. cross-repo traceability
2. smoke coverage breadth
3. regression corpus breadth
4. structured-return allowlist breadth
5. grouped-ingest ergonomics

All five now have concrete follow-on coverage.

## Before / After Matrix

| Gap area | Original status | Current status | Evidence | Residual gap |
| --- | --- | --- | --- | --- |
| Cross-repo lesson traceability | `missing` | `resolved` | [INGESTER_LESSON_CROSSWALK.md](/Users/conrad/personal/ageo-atoms/INGESTER_LESSON_CROSSWALK.md) | Crosswalk will still need refreshes if new major repair families appear |
| Smoke coverage breadth | `thin canary layer` | `improved but still intentionally narrow` | [smoke.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/smoke.py), [test_ingest_smoke.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_smoke.py) | Coverage is still curated, not parity-scale |
| Regression corpus breadth | `small` | `materially improved` | [regression_harness.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/regression_harness.py), [test_ingest_regression_harness.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_regression_harness.py), [tempo_grouped_offsets](/Users/conrad/personal/ageo-matcher/tests/golden/ingest_regression/tempo_grouped_offsets) | Corpus is still selective rather than exhaustive |
| Structured-return allowlist breadth | `single case only` | `still narrow, but less so` | [return_shapes.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/return_shapes.py), [test_ingester_return_shapes.py](/Users/conrad/personal/ageo-matcher/tests/test_ingester_return_shapes.py) | Explicit allowlist remains intentionally small |
| Grouped-ingest ergonomics | `publication metadata only` | `improved` | [cli.py](/Users/conrad/personal/ageo-matcher/sciona/cli.py), [ingest_cmds.py](/Users/conrad/personal/ageo-matcher/sciona/commands/ingest_cmds.py), [monitor.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/monitor.py), [test_ingest_output_scope.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_output_scope.py) | No true grouped merge / multi-target ingest workflow yet |

## Current Coverage Assessment

### 1. Cross-repo traceability is now covered

This was the cleanest gap to close. The crosswalk now gives later planners and
reviewers a durable mapping from repaired `ageo-atoms` families to matcher
coverage.

Primary evidence:

- [INGESTER_LESSON_CROSSWALK.md](/Users/conrad/personal/ageo-atoms/INGESTER_LESSON_CROSSWALK.md)

Assessment:

- `resolved`

### 2. Smoke coverage is now meaningfully better

The matcher smoke layer now covers more than sklearn grouped images and narrow
numerical canaries. It also includes a bounded BioSPPy detector slice:

- ECG:
  - `hamilton_segmentation`
  - `hamilton_segmenter`
- PPG:
  - `detect_signal_onsets_elgendi2013`
  - `detectonsetevents`
- EMG:
  - `detect_onsets_with_rest_aware_thresholds`
  - `threshold_based_onset_detection`

Primary evidence:

- [smoke.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/smoke.py)
- [test_ingest_smoke.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_smoke.py)

Assessment:

- `improved`
- still intentionally curated rather than broad

### 3. Regression coverage is no longer limited to the original canaries

The curated regression suite now includes:

- grouped sklearn images
- detector structured output
- grouped tempo offsets

Primary evidence:

- [regression_harness.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/regression_harness.py)
- [test_ingest_regression_harness.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_regression_harness.py)
- [tests/fixtures/ingest_regression/tempo_grouped_offsets/source.py](/Users/conrad/personal/ageo-matcher/tests/fixtures/ingest_regression/tempo_grouped_offsets/source.py)
- [tests/golden/ingest_regression/tempo_grouped_offsets](/Users/conrad/personal/ageo-matcher/tests/golden/ingest_regression/tempo_grouped_offsets)

Assessment:

- `improved`
- still selective rather than exhaustive

### 4. Structured-return knowledge remains intentionally narrow, but no longer single-case

The matcher now has two explicit detector-like dict-return cases:

- `PeakDetector.detect`
- `OnsetDetector.detect_events`

Primary evidence:

- [return_shapes.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/return_shapes.py)
- [test_ingester_return_shapes.py](/Users/conrad/personal/ageo-matcher/tests/test_ingester_return_shapes.py)

Assessment:

- `improved`
- still intentionally narrow
- still correctly conservative for everything else

### 5. Grouped-ingest ergonomics are safer now

The matcher now has an explicit replacement guard for family-scope output:

- new flag: `--allow-family-replace`
- default behavior: fail early if a family-scope target dir already contains
  canonical published artifacts
- monitor/status output records:
  - `existing_family_output`
  - `existing_family_artifacts`
  - `allow_family_replace`
  - `family_publication_mode`

Primary evidence:

- [cli.py](/Users/conrad/personal/ageo-matcher/sciona/cli.py)
- [ingest_cmds.py](/Users/conrad/personal/ageo-matcher/sciona/commands/ingest_cmds.py)
- [monitor.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/monitor.py)
- [test_ingest_output_scope.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_output_scope.py)

Assessment:

- `improved`
- safer operator path now exists
- no true merge or multi-target grouped ingest yet

## Re-evaluated Family Examples

### `sklearn/images`

Current matcher-side support now includes:

- grouped output scope handling
- grouped smoke coverage
- grouped regression coverage

Assessment:

- `partially_covered`, but solidly represented

### `tempo_jl/offsets`

Current matcher-side support now includes:

- grouped regression coverage via `tempo_grouped_offsets`
- grouped output safety/guardrail coverage through family replacement checks

Current missing piece:

- no dedicated matcher smoke probe for this grouped family

Assessment:

- moved from `uncovered` to `partially_covered`

## What Still Remains Intentionally Narrow

These are not failures. They are places where the current system is
deliberately conservative.

### 1. Smoke coverage is curated, not audit-scale

The matcher should not attempt to replicate the full
[runtime_probes.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/runtime_probes.py)
surface.

Remaining state:

- acceptable
- future expansion should stay selective

### 2. Structured-return knowledge is explicit, not inferred

This is the correct design. The remaining limitation is breadth, not shape.

Remaining state:

- acceptable
- future additions should stay case-by-case

### 3. Grouped ingest is safer, but not merged

The matcher now protects operators from silent grouped-family replacement, but
it still does not offer:

- merge semantics for existing grouped files
- a full multi-target grouped-ingest workflow

Remaining state:

- intentionally deferred
- not a blocker for the current refinement program

## Overall Conclusion

The gap-closure work succeeded.

The current state is:

- core ingester hardening complete
- rollout gaps materially reduced
- grouped-family safety improved
- detector-family smoke coverage improved
- grouped-family regression coverage improved
- structured-return coverage improved
- no new major design gap found

The remaining open items are:

- deliberately narrow smoke breadth
- deliberately narrow structured-return breadth
- lack of true merge-style grouped ingest

Those are future refinement opportunities, not unresolved failures in the
current ingester program.
