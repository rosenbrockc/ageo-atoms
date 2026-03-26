# Audit Execution Plan

This document is the restart-safe execution plan for the repository audit
defined in [AUDIT_INGEST.md](/Users/conrad/personal/ageo-atoms/AUDIT_INGEST.md).

It is written for planning agents that may stop and restart. Every phase below
states:

- objective
- current status
- inputs and dependencies
- concrete deliverables
- implementation tasks
- idempotent commands to rerun
- exit criteria
- handoff notes for the next agent

This file should be treated as the operational source of truth for planning and
sequencing. The existing
[AUDIT_TOOL_IMPLEMENTATION.md](/Users/conrad/personal/ageo-atoms/AUDIT_TOOL_IMPLEMENTATION.md)
remains the tool-design document.

## Current Baseline

Implemented already:

- inventory/manifest builder:
  [build_audit_manifest.py](/Users/conrad/personal/ageo-atoms/scripts/build_audit_manifest.py)
- signature fidelity pass:
  [audit_signature_fidelity.py](/Users/conrad/personal/ageo-atoms/scripts/audit_signature_fidelity.py)
- deterministic acceptability scorer:
  [audit_acceptability.py](/Users/conrad/personal/ageo-atoms/scripts/audit_acceptability.py)
- shared audit library:
  [scripts/auditlib/](/Users/conrad/personal/ageo-atoms/scripts/auditlib)
- generated portfolio artifacts:
  [audit_manifest.json](/Users/conrad/personal/ageo-atoms/data/audit_manifest.json),
  [audit_scores.csv](/Users/conrad/personal/ageo-atoms/data/audit_scores.csv)

Not implemented yet:

- normalized Phase 2 structural report
- runtime probes
- deterministic return/state/generated-noun checks
- semantic aggregator
- audit progress reporter
- review-record workflow
- remediation workflow
- CI/promotion gating

## Global Rules

These rules apply to every phase:

1. All audit scripts must be rerunnable and idempotent.
2. Inventory and structural tools must not depend on importing `ageoa` as a
   package; they must work on a broken repo.
3. Generated artifacts belong under `data/` and must be derivable from source.
4. Every phase must leave behind durable artifacts that let the next agent
   resume work without reconstructing state from memory.
5. A deterministic tool may lower or cap trust, but may not grant `trusted`.
6. Every new finding code must be documented in code and used consistently.

## Status Sources Of Truth

Agents should consult these files first after a restart:

- [AUDIT_INGEST.md](/Users/conrad/personal/ageo-atoms/AUDIT_INGEST.md)
- [AUDIT_TOOL_IMPLEMENTATION.md](/Users/conrad/personal/ageo-atoms/AUDIT_TOOL_IMPLEMENTATION.md)
- [AUDIT_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/AUDIT_EXECUTION_PLAN.md)
- [audit_manifest.json](/Users/conrad/personal/ageo-atoms/data/audit_manifest.json)
- [audit_scores.csv](/Users/conrad/personal/ageo-atoms/data/audit_scores.csv)

If those disagree, this precedence order should apply:

1. `AUDIT_INGEST.md` for policy
2. `AUDIT_EXECUTION_PLAN.md` for sequencing
3. generated `data/` artifacts for current repo state
4. `AUDIT_TOOL_IMPLEMENTATION.md` for tool design details

## Phase 1: Inventory And Manifest

Objective:

- maintain a complete repository-wide index of public atoms and required audit
  metadata

Current status:

- partially implemented
- manifest exists, but still needs schema completion and a durable manifest
  validation pass

Primary inputs:

- `ageoa/**/*.py`
- `scripts/atom_manifest.yml`
- `tests/fixtures/`
- local sibling artifacts beside each atom

Primary outputs:

- `data/audit_manifest.json`
- optional CSV mirror

Missing from current implementation:

- explicit stochastic/procedural flags
- authoritative source fields
- manifest validation report
- stronger source-kind classification
- clearer distinction between inventory facts and derived judgments

Implementation tasks:

1. Split manifest fields into:
   - discovered facts
   - derived heuristics
   - audit results
2. Add missing schema fields required by `AUDIT_INGEST.md`:
   - `authoritative_sources`
   - `stateful_kind`
   - `stochastic`
   - `procedural`
   - `references_status`
   - `review_basis_at`
3. Add manifest validation:
   - required keys present
   - `atom_id` uniqueness
   - path existence checks
   - source-kind enum validation
4. Add a summary block:
   - atom count
   - family counts
   - risk-tier counts
   - unmapped upstream count
5. Emit machine-readable manifest validation errors under `data/audit/`

Idempotent commands:

```bash
python scripts/build_audit_manifest.py
python - <<'PY'
import json
data = json.load(open('data/audit_manifest.json'))
print(data['schema_version'], len(data['atoms']))
PY
```

Exit criteria:

- every public atom has exactly one manifest row
- no duplicate `atom_id`
- schema validation passes
- manifest captures all required Phase 1 fields or explicitly marks them
  `unknown`

Handoff notes:

- Phase 2 should consume the manifest instead of rediscovering atoms
- if discovery rules change, regenerate the full manifest before any downstream
  audits

## Phase 2: Structural Bulk Audit

Objective:

- normalize static structural checks into a repository-wide structural report

Current status:

- not implemented as a unified report
- inputs exist:
  [audit.py](/Users/conrad/personal/ageo-atoms/scripts/audit.py) and
  [type_and_isomorphism_audit.py](/Users/conrad/personal/ageo-atoms/scripts/type_and_isomorphism_audit.py)

Primary inputs:

- `data/audit_manifest.json`
- `scripts/audit.py`
- `scripts/type_and_isomorphism_audit.py`
- `../ageo-matcher/scripts/verify_atoms_repo.py`

Primary outputs:

- `data/audit/structural_report.json`
- `data/audit/structural_findings.csv`
- manifest updates for:
  - `structural_status`
  - `blocking_findings`
  - `required_actions`

Missing from current implementation:

- orchestration wrapper
- finding-code normalization
- per-atom structural finding rollup
- manifest integration
- progress summary

Implementation tasks:

1. Create `scripts/audit_structural.py`
2. Wrap each structural source as a deterministic collector:
   - `scripts/audit.py`
   - `verify_atoms_repo.py`
   - `type_and_isomorphism_audit.py`
3. Define normalized finding codes such as:
   - `STRUCT_PARSE_FAIL`
   - `STRUCT_REGISTER_MISSING`
   - `STRUCT_WITNESS_MISSING`
   - `STRUCT_IMPORT_HEAVY`
   - `STRUCT_DOCSTRING_PLACEHOLDER`
   - `STRUCT_STUB_PUBLIC_API`
   - `STRUCT_CDG_INVALID`
4. Map raw tool output to those codes
5. Join findings back to `atom_id`
6. Derive `structural_status` with consistent rules:
   - `fail` for parse, registration, witness, CDG, or stub failures
   - `partial` for placeholder docs or weaker hygiene issues
   - `pass` otherwise
7. Emit repo summary:
   - counts by finding code
   - counts by family
   - counts by severity

Idempotent commands:

```bash
python scripts/build_audit_manifest.py
python scripts/audit_structural.py
python - <<'PY'
import json
data = json.load(open('data/audit/structural_report.json'))
print(data['summary'])
PY
```

Exit criteria:

- every manifest row has a structural result
- all structural findings are normalized to stable codes
- the report is machine-readable and can be diffed across runs

Handoff notes:

- Phase 3 should consume structural findings to improve risk-tier assignment
- Phase 5 acceptability must read structural results from the structural report,
  not reimplement the logic independently

## Phase 3: Semantic Risk Triage

Objective:

- prioritize atoms by likelihood of semantic failure

Current status:

- partially implemented
- current `risk_tier` is a coarse heuristic in the manifest

Primary inputs:

- manifest facts
- structural report
- signature fidelity evidence
- parity coverage

Primary outputs:

- improved `risk_tier`
- `risk_reasons`
- `data/audit/risk_report.json`

Implementation tasks:

1. Replace coarse `risk_tier` logic with a scored heuristic model
2. Record reasons for every high-risk label
3. Distinguish:
   - stateful risk
   - generated-wrapper risk
   - fidelity risk
   - evidence-gap risk
4. Add per-family risk summaries
5. Expose a sorted review queue

Idempotent commands:

```bash
python scripts/build_audit_manifest.py
python scripts/audit_signature_fidelity.py
python scripts/audit_risk.py
```

Exit criteria:

- every atom has `risk_tier` and `risk_reasons`
- high-risk portfolio is meaningfully smaller than full-repo inventory
- risk signals are explainable from persisted evidence

Handoff notes:

- Phase 4 and Phase 7 should start from the high-risk queue

## Phase 4: Semantic Review Rubric

Objective:

- create durable, structured human/model review records for semantic judgments

Current status:

- not implemented

Primary inputs:

- manifest
- structural report
- fidelity evidence
- runtime probe evidence
- authoritative source references

Primary outputs:

- `data/audit_reviews/<atom_id>.json` or `.md`

Implementation tasks:

1. Define review record schema
2. Include required sections:
   - upstream symbols
   - authoritative sources
   - line references
   - semantic verdict
   - developer-semantics verdict
   - limitations
   - required follow-up
3. Add templates for reviewer use
4. Add a validator for review records

Idempotent commands:

```bash
python scripts/init_audit_reviews.py --atom-id <atom_id>
python scripts/validate_audit_reviews.py
```

Exit criteria:

- review template exists
- review records validate
- records can be joined back to manifest rows

Handoff notes:

- no atom can become `trusted` without a valid review record

## Phase 5: Deterministic Semantic Checks

Objective:

- automate conservative semantic signals that catch suspicious atoms

Current status:

- partially implemented
- signature fidelity exists
- return/state/generated-noun/runtime-probe checks do not

Primary inputs:

- manifest
- upstream mappings
- vendored source
- runtime probe fixtures

Primary outputs:

- per-atom evidence files under `data/audit/evidence/`
- updated `semantic_status`

Implementation tasks:

1. Keep `audit_signature_fidelity.py`
2. Add `audit_runtime_probes.py`
3. Add `audit_return_fidelity.py`
4. Add `audit_state_fidelity.py`
5. Add `audit_generated_nouns.py`
6. Add `audit_semantics.py` as the aggregator

Idempotent commands:

```bash
python scripts/audit_signature_fidelity.py
python scripts/audit_runtime_probes.py
python scripts/audit_return_fidelity.py
python scripts/audit_state_fidelity.py
python scripts/audit_generated_nouns.py
python scripts/audit_semantics.py
```

Exit criteria:

- deterministic semantic evidence exists per atom
- `semantic_status` and `developer_semantics_status` are derived from persisted
  evidence

Handoff notes:

- keep all semantic checks conservative; they should surface risk, not assert
  semantic equivalence

## Phase 6: Parity And Usage Tests

Objective:

- expand parity/usage evidence into systematic behavioral coverage

Current status:

- partially implemented in the repo already
- not integrated into audit coverage accounting

Primary inputs:

- fixtures in `tests/fixtures/`
- generated parity tests
- family-specific tests

Primary outputs:

- parity coverage report
- per-atom `parity_test_status`
- recommended missing-fixture queue

Implementation tasks:

1. inventory current parity fixtures and map them to `atom_id`
2. mark atoms by coverage level:
   - no parity evidence
   - positive-path only
   - positive and negative path
   - parity/usage-equivalence present
3. generate a parity coverage summary
4. create missing-fixture backlog files by family

Idempotent commands:

```bash
python scripts/generate_parity_tests.py
pytest -q tests/test_parity.py
python scripts/report_parity_coverage.py
```

Exit criteria:

- parity coverage is measurable per atom
- missing parity evidence is visible in reports

Handoff notes:

- Phase 9 gating should consume the parity coverage report

## Phase 7: Human / Model Review Pass

Objective:

- perform actual semantic review over the prioritized queue

Current status:

- not implemented as a workflow

Primary inputs:

- risk report
- structural report
- semantic evidence
- review templates

Primary outputs:

- validated review records
- final review-ready queue

Implementation tasks:

1. define review batching rules
2. review high-risk atoms first
3. update manifest rows from validated review records
4. track review throughput and backlog

Idempotent commands:

```bash
python scripts/report_audit_progress.py
python scripts/validate_audit_reviews.py
```

Exit criteria:

- every high-risk atom has a review status
- reviewed atoms have validated records

Handoff notes:

- do not mix review verdicts with deterministic ceilings; keep both

## Phase 8: Repository-Wide Remediation

Objective:

- fix or clearly downgrade problematic atoms in priority order

Current status:

- not implemented as a program

Primary inputs:

- scored portfolio
- validated review records
- structural report

Primary outputs:

- remediation queue
- fix PRs or local commits
- updated audit records

Implementation tasks:

1. build remediation queue from:
   - `broken`
   - `misleading`
   - high-risk unknown
2. track owner, status, and target artifact for each fix
3. require every fix to update audit evidence and tests

Idempotent commands:

```bash
python scripts/report_audit_progress.py
python scripts/build_remediation_queue.py
```

Exit criteria:

- remediation queue exists
- fixes can be traced back to findings and audit records

Handoff notes:

- remediation should always follow from findings, never from vague “cleanup”

## Phase 9: Ongoing Gating

Objective:

- make semantic quality part of normal ingest and merge flow

Current status:

- not implemented

Primary inputs:

- structural report
- semantic report
- parity coverage
- validated review records

Primary outputs:

- CI checks
- promotion rules
- merge/release gates

Implementation tasks:

1. add CI target for manifest and structural audit generation
2. block regressions on structural failures
3. require review records before promotion to `trusted`
4. require parity evidence before promotion from
   `acceptable_with_limits` to `trusted`

Idempotent commands:

```bash
python scripts/build_audit_manifest.py
python scripts/audit_structural.py
python scripts/audit_semantics.py
python scripts/report_audit_progress.py
```

Exit criteria:

- CI can regenerate and validate audit artifacts
- promotion rules are enforced by code, not by convention

Handoff notes:

- Phase 9 should be last; do not lock the repo behind gates until the upstream
  reporting pipeline is stable

## Cross-Phase Dependency Map

Dependencies:

- Phase 1 unlocks all later phases
- Phase 2 depends on Phase 1
- Phase 3 depends on Phases 1 and 2
- Phase 4 depends on Phases 1, 2, 3, and 5
- Phase 5 depends on Phase 1, and should start before Phase 7
- Phase 6 can proceed in parallel after Phase 1
- Phase 7 depends on Phases 3, 4, and 5
- Phase 8 depends on Phases 2, 5, and 7
- Phase 9 depends on stable outputs from Phases 2, 5, 6, and 7

Parallelizable work:

- Phase 5 and Phase 6 can run in parallel once Phase 1 is stable
- review record templating in Phase 4 can start while deterministic checks are
  still being completed

## Agent Restart Checklist

When a planning agent restarts, it should do exactly this:

1. open `AUDIT_INGEST.md`
2. open `AUDIT_EXECUTION_PLAN.md`
3. inspect `data/audit_manifest.json`
4. inspect `data/audit_scores.csv`
5. inspect `scripts/auditlib/`
6. identify the current target phase
7. rerun the target phase’s idempotent command set
8. compare outputs against the target phase’s exit criteria

If the outputs fail validation, the agent should fix the producing phase before
moving forward.

