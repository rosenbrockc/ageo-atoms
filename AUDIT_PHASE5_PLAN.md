# Phase 5 Deterministic Semantic Checks Plan

This document is the concrete implementation plan for Phase 5 from
[AUDIT_INGEST.md](/Users/conrad/personal/ageo-atoms/AUDIT_INGEST.md):
deterministic semantic checks.

It is written to survive process restarts. A planning or coding agent should be
able to resume from this file plus the existing audit artifacts without relying
on prior chat context.

## Goal

Implement conservative, rerunnable semantic checks that surface suspicious atoms
for review without claiming semantic equivalence.

Phase 5 should add deterministic evidence for:

- return fidelity
- state fidelity
- generated-noun risk
- runtime probe coverage
- semantic aggregation

This phase should strengthen the audit portfolio and reduce how much Phase 3 is
dominated by missing-evidence reasons.

## Current Baseline

Available inputs:

- manifest:
  [audit_manifest.json](/Users/conrad/personal/ageo-atoms/data/audit_manifest.json)
- structural report:
  [structural_report.json](/Users/conrad/personal/ageo-atoms/data/audit/structural_report.json)
- risk report:
  [risk_report.json](/Users/conrad/personal/ageo-atoms/data/audit/risk_report.json)
- review queue:
  [review_queue.csv](/Users/conrad/personal/ageo-atoms/data/audit/review_queue.csv)
- signature evidence:
  [data/audit/evidence/](/Users/conrad/personal/ageo-atoms/data/audit/evidence)
- acceptability results:
  [data/audit/results/](/Users/conrad/personal/ageo-atoms/data/audit/results)
- draft review records:
  [data/audit_reviews/](/Users/conrad/personal/ageo-atoms/data/audit_reviews)

Implemented already:

- signature fidelity:
  [audit_signature_fidelity.py](/Users/conrad/personal/ageo-atoms/scripts/audit_signature_fidelity.py)
- acceptability scoring:
  [audit_acceptability.py](/Users/conrad/personal/ageo-atoms/scripts/audit_acceptability.py)
- structural aggregation:
  [audit_structural.py](/Users/conrad/personal/ageo-atoms/scripts/audit_structural.py)
- risk triage:
  [audit_risk.py](/Users/conrad/personal/ageo-atoms/scripts/audit_risk.py)

Current limitation:

- no runtime probes
- no return-fidelity analysis
- no state-fidelity analysis
- no generated-noun analysis
- no semantic aggregator joining those checks back into the manifest

## Deliverables

Required code:

- `scripts/auditlib/runtime_probes.py`
- `scripts/auditlib/return_fidelity.py`
- `scripts/auditlib/state_fidelity.py`
- `scripts/auditlib/generated_nouns.py`
- `scripts/auditlib/semantics.py`
- `scripts/audit_runtime_probes.py`
- `scripts/audit_return_fidelity.py`
- `scripts/audit_state_fidelity.py`
- `scripts/audit_generated_nouns.py`
- `scripts/audit_semantics.py`

Files likely to modify:

- `scripts/auditlib/acceptability.py`
- `scripts/auditlib/models.py`
- `scripts/auditlib/paths.py`
- possibly `scripts/audit_risk.py`
  if risk should consume new semantic evidence after aggregation

Required generated artifacts:

- `data/audit/probes/<atom>.json`
- `data/audit/evidence/<atom>.json`
  extended with new sections or sibling semantic evidence files
- `data/audit/semantic_report.json`
- updated `data/audit_manifest.json`
- updated `data/audit_scores.csv`

## Scope Boundaries

Phase 5 is about deterministic checks only.

Phase 5 must not:

- promote atoms to `trusted`
- replace review records
- make unconstrained internet-backed judgments
- attempt to prove semantic equivalence

Phase 5 may:

- assign or update `semantic_status`
- assign or update `developer_semantics_status`
- add or remove `required_actions`
- cap or lower acceptability based on new evidence
- reduce evidence-gap-driven risk by adding real runtime or fidelity signals

## Semantic Evidence Model

For each atom, Phase 5 should populate deterministic evidence under these
categories:

- `signature_fidelity`
- `runtime_probe`
- `return_fidelity`
- `state_fidelity`
- `generated_nouns`

Each category should emit:

- `status`
- `findings`
- `notes`
- `source_refs`

Allowed statuses:

- `pass`
- `partial`
- `fail`
- `unknown`
- `not_applicable`

## Finding Code Families

Use stable finding code prefixes so they group cleanly:

Runtime:

- `RUNTIME_IMPORT_FAIL`
- `RUNTIME_PROBE_PASS`
- `RUNTIME_PROBE_FAIL`
- `RUNTIME_CONTRACT_NEGATIVE_PASS`
- `RUNTIME_CONTRACT_NEGATIVE_FAIL`
- `RUNTIME_PROBE_SKIPPED`
- `RUNTIME_NOT_IMPLEMENTED`

Return fidelity:

- `RETURN_FABRICATED_ATTRIBUTE`
- `RETURN_IGNORES_UPSTREAM_VALUE`
- `RETURN_DERIVED_ARTIFACT_UNDOCUMENTED`
- `RETURN_UNKNOWN`

State fidelity:

- `STATE_FABRICATED_FIELD`
- `STATE_REHYDRATION_MISSING`
- `STATE_QUERY_MUTATION_CONFUSION`
- `STATE_UNKNOWN`

Generated nouns:

- `NOUN_UNDOCUMENTED_OUTPUT`
- `NOUN_UNDOCUMENTED_STATE`
- `NOUN_LOW_UPSTREAM_ALIGNMENT`
- `NOUN_ALLOWLISTED_DERIVATION`

Semantic aggregation:

- `SEMANTIC_RUNTIME_SUPPORT_PRESENT`
- `SEMANTIC_RUNTIME_SUPPORT_MISSING`
- `SEMANTIC_FIDELITY_FAIL`
- `SEMANTIC_REVIEW_REQUIRED`

## Runtime Probe Strategy

The runtime probe system must be conservative.

### Initial Safe Scope

Phase 5 should start with safe probeable atoms only:

- pure Python wrappers
- small ndarray/scalar inputs
- no obvious I/O or network behavior
- no known expensive iterative pipelines
- no FFI-backed atoms unless there is a stable, tiny fixture and import path

Safe initial families likely include:

- `algorithms`
- selected `numpy`
- selected `scipy`
- selected `biosppy` signal primitives

### Probe Inputs

Use this priority order:

1. existing parity fixtures
2. existing `scripts/generate_base_inputs.py`
3. deterministic tiny synthetic inputs from a new helper registry

Every probe run should record whether it was:

- executed
- skipped
- failed

and why.

### Runtime Probe Outputs

Each probe artifact should contain:

- `atom_id`
- `probe_status`
- `positive_probe`
- `negative_probe`
- `parity_used`
- `skip_reason`
- `exception_type`
- `exception_message`

## Return Fidelity Strategy

The return-fidelity pass should be AST-based and conservative.

Target signals:

- wrapper calls an upstream function or method and then returns unrelated
  attributes
- wrapper ignores a concrete upstream return value
- wrapper returns attributes that are never written in the wrapper body
- wrapper returns tuple/object structures not reflected in docstring or state

Heuristics:

1. identify upstream call assignment targets
2. identify returned expressions
3. compare returned names/attributes to values written before return
4. flag suspicious attribute reads after method calls

Do not attempt full dataflow or alias analysis in v1.

## State Fidelity Strategy

The state-fidelity pass should focus on stateful wrappers only.

Target signals:

- state model fields not observed in wrapper writes
- wrappers that construct state updates without upstream anchors
- inference/predict wrappers that appear to require fitted fields but do not
  rehydrate them
- query-like methods that appear to mutate or training-like methods that appear
  read-only in misleading ways

Heuristics:

1. detect state arguments and state return positions
2. detect `.model_copy(update=...)` and similar update patterns
3. compare updated fields against wrapper assignments and upstream anchors
4. detect method names like `predict`, `transform`, `query`, `get`, `read`
5. detect training names like `fit`, `train`, `update`, `initialize`

## Generated-Noun Strategy

Generated nouns are deterministic proxy signals for hallucinated APIs.

Target signals:

- output/state names that do not occur in:
  - upstream symbol names
  - vendored source tokens
  - docstring text
  - allowlisted derived abstractions

Required support files:

- `data/audit/allowlists/generated_nouns.json`
- optionally:
  - `data/audit/allowlists/state_fields.json`
  - `data/audit/allowlists/decomposition_aliases.json`

Phase 5 initial implementation may start with only
`generated_nouns.json`.

## Semantic Aggregator

`scripts/audit_semantics.py` should be the single rollup layer for the new
Phase 5 evidence.

Responsibilities:

1. load manifest
2. load structural report
3. load all semantic evidence categories
4. derive:
   - `semantic_status`
   - `developer_semantics_status`
   - additional `required_actions`
5. write `data/audit/semantic_report.json`
6. optionally refresh acceptability using the new semantic evidence

### Rollup Rules

Suggested v1 rules:

- `fail` if runtime probe fails on a safe positive path or if return/state
  fidelity emits a fabricated/fail code
- `partial` if only warnings, skips, or unknowns exist
- `pass` if runtime support exists and no major fidelity/noun issues appear
- `unknown` if no semantic evidence exists

## Manifest Integration Rules

After Phase 5 aggregation, each manifest row should be updated with:

- `runtime_status`
- `semantic_status`
- `developer_semantics_status`
- `required_actions`
- `blocking_findings`

Phase 5 should not overwrite structural or review-derived fields except where a
new semantic failure clearly adds blockers/actions.

## Acceptability Integration

Phase 5 should reduce duplication in the acceptability scorer.

Required follow-on:

- `scripts/auditlib/acceptability.py` should consume Phase 5 semantic evidence
  and the structural report instead of relying only on local heuristics

At minimum:

- runtime dimension should use runtime probe artifacts
- fidelity dimension should use return/state/noun findings

## Risk Integration

Phase 5 may optionally refresh Phase 3 risk outputs after semantic evidence is
written.

Suggested behavior:

- keep `audit_risk.py` separate
- if semantic evidence is added, rerun `audit_risk.py`
- risk reasons like `RISK_NO_RUNTIME_EVIDENCE` should naturally drop where
  probes now exist

## Test Plan

Add focused tests for:

1. runtime probe pass/skip/fail behavior on synthetic atoms
2. return-fidelity suspicious attribute detection
3. state-fidelity detection on a small stateful wrapper fixture
4. generated-noun flagging and allowlist behavior
5. semantic aggregation rollup
6. acceptability integration with semantic evidence

Recommended test files:

- `tests/test_audit_runtime_probes.py`
- `tests/test_audit_return_fidelity.py`
- `tests/test_audit_state_fidelity.py`
- `tests/test_audit_generated_nouns.py`
- `tests/test_audit_semantics.py`

Prefer compact synthetic fixtures and temporary files over depending on the
full repo where possible.

## Execution Order

Implement in this order:

1. add path constants and allowlist directory support
2. implement runtime probe helpers and CLI
3. implement return-fidelity pass and CLI
4. implement state-fidelity pass and CLI
5. implement generated-noun pass and allowlist file
6. implement semantic aggregator
7. integrate acceptability with the new evidence
8. add tests
9. regenerate semantic artifacts
10. rerun risk triage if needed

## Idempotent Command Set

An agent resuming this phase should rerun:

```bash
python scripts/build_audit_manifest.py
python scripts/validate_audit_manifest.py
python scripts/audit_structural.py
python scripts/audit_signature_fidelity.py
python scripts/audit_risk.py
python scripts/audit_runtime_probes.py
python scripts/audit_return_fidelity.py
python scripts/audit_state_fidelity.py
python scripts/audit_generated_nouns.py
python scripts/audit_semantics.py
python scripts/audit_acceptability.py
pytest -q tests/test_audit_runtime_probes.py tests/test_audit_return_fidelity.py tests/test_audit_state_fidelity.py tests/test_audit_generated_nouns.py tests/test_audit_semantics.py
```

If a prerequisite artifact is stale or missing, regenerate it first.

## Exit Criteria

Phase 5 is complete when:

1. runtime probe artifacts exist for safe probeable atoms
2. return/state/generated-noun evidence exists for the portfolio or an
   explicitly documented subset
3. `data/audit/semantic_report.json` exists
4. manifest rows reflect semantic/runtime status from deterministic evidence
5. acceptability consumes the new semantic evidence
6. targeted tests pass

## Known Non-Goals

This phase does not:

- create or complete human review records
- perform broad CI gating
- promote atoms to `trusted`
- guarantee semantic correctness

Those belong to later phases.

## Agent Handoff Template

If an implementing agent stops mid-phase, it should leave:

- current commit and branch
- files created or modified
- which semantic evidence categories are implemented
- whether `data/audit/semantic_report.json` is fresh
- whether runtime probes are partial or full-scope
- tests run
- current blockers

If no handoff is available, the next agent should assume artifacts may be stale
and rerun the idempotent command set.

