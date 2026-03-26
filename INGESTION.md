# Atom Ingestion Guide

This document describes the current ingestion contract for `ageoa` as it
exists today.

The repo no longer treats ingestion as "generate wrappers and stop." An ingest
is only considered acceptable when:

- the stable atom artifacts are placed correctly in `ageoa/`
- the transient run byproducts are excluded from version control
- the deterministic audit toolchain has been rerun
- the resulting atom portfolio lands in an acceptable risk/review state

The matcher-side producer is `sciona ingest` in `../ageo-matcher`. The
repo-local consumer is the deterministic audit pipeline under `scripts/`.

## Current Repository Shape

New generated atoms are expected to live in per-atom directories:

```text
ageoa/
  <domain>/
    __init__.py
    <atom_name>/
      __init__.py
      atoms.py
      witnesses.py
      state_models.py          # only when stateful
      cdg.json
      matches.json             # optional but commonly retained
      references.json          # optional, curated/reference metadata
      uncertainty.json         # optional, measured later
```

Legacy atoms still exist and remain valid. Those may use:

- `ageoa/<domain>/<module>.py`
- `ageoa/<domain>/<stem>_witnesses.py`
- `ageoa/<domain>/<stem>_cdg.json`

The audit tooling explicitly supports both shapes. New ingest work should use
the per-atom directory layout unless there is a strong reason not to.

## Stable vs Ephemeral Artifacts

### Commit These

- `atoms.py`
- `witnesses.py`
- `state_models.py` when required
- `cdg.json`
- legacy `*_cdg.json` for grandfathered atoms
- `matches.json` when produced and worth preserving
- `references.json` when scholarly attribution exists
- `uncertainty.json` when perturbation analysis has been run
- package `__init__.py` updates

### Do Not Commit These

These are runtime monitor/telemetry artifacts, not canonical repo state:

- `.ingest_status.json`
- `COMPLETED.json`
- `FAILED.json`
- `trace.jsonl`
- `shared_context_metrics.json`
- local ingest logs under `logs/`
- local browser/debug captures such as `.playwright-mcp/`

The repo `.gitignore` reflects this policy.

## What `sciona ingest` Produces

`sciona ingest` still emits operational sidecars during a run, but only a
subset of its output is considered durable repo state.

Stable products:

- `atoms.py`
- `witnesses.py`
- `state_models.py` when state is hoisted
- `cdg.json`
- `matches.json` when match results are available

Operational sidecars:

- `.ingest_status.json`
- `COMPLETED.json`
- `FAILED.json`
- `trace.jsonl`
- `shared_context_metrics.json`

The operational sidecars are useful while debugging or monitoring a run, but
they are not part of the committed atom contract anymore.

## Ingestion Procedure

1. Run `sciona ingest` from `../ageo-matcher` with an output directory that
   already matches the target `ageoa/<domain>/<atom_name>` layout.
2. Keep the stable outputs listed above.
3. Update the relevant package `__init__.py` exports.
4. If the atom has known scholarly sources, update `references.json`.
5. If the atom has empirical perturbation data, keep or add
   `uncertainty.json`.
6. Rerun the deterministic audit pipeline described below.
7. Do not treat CLI success alone as acceptance; the audit outputs are the
   authoritative repo-local judgment.

## Wrapper Contract

The generated Python wrappers must still satisfy the core wrapper contract.

### `atoms.py`

- Every public atom must be fully type annotated.
- No public `Any`, `*args`, or `**kwargs`.
- Every public atom must be registered with `@register_atom(...)`.
- Every public atom must have meaningful `@icontract.require` and
  `@icontract.ensure` coverage.
- Stateful wrappers must use the generated state model rather than hidden
  ambient state.
- Skeleton behavior is allowed only when it is explicit and honest.

### `witnesses.py`

- Witnesses accept and return abstract ghost types only.
- Witnesses are pure and side-effect free.
- Witness shape/dtype/domain propagation should be semantically faithful.

### `state_models.py`

- Include only the state that must survive across calls/windows.
- Do not hide required constructor/configuration state in undocumented fields.

### `cdg.json`

- Must be structurally valid.
- Must describe the actual decomposition emitted by the wrapper set.
- New ingests should use `cdg.json`.
- Legacy `*_cdg.json` remains acceptable only for already-existing atoms.

## `matches.json`

`matches.json` is now treated as a retained handoff artifact rather than a
throwaway byproduct.

Why it matters:

- it captures matcher grounding output used by downstream synthesis flows
- it is not duplicated anywhere else in this repo
- it is harder to reproduce exactly than the monitor sidecars

It is not required for every atom, but if matcher-side grounding produced it,
keeping it is preferred.

## `references.json`

`references.json` is not part of the ingest engine itself, but it is part of
the repository-level atom quality contract.

Why it matters:

- it stores scholarly attribution and reference provenance per atom
- it feeds `data/hyperparams/manifest.json`
- it feeds `data/hyperparams/manifest.sqlite`
- it strengthens review/audit work

Use:

- `python scripts/add_reference.py ...`
- `python scripts/build_references.py`

## `uncertainty.json`

`uncertainty.json` is also outside the core ingest step, but it is a durable
post-ingest artifact when it exists.

Why it matters:

- it stores empirical perturbation analysis results
- it is consumed by matcher-side uncertainty loading
- it is validated by the local audit tooling

Generate it with:

```bash
python scripts/measure_uncertainty.py --atom <atom_name>
python scripts/measure_uncertainty.py --domain <domain>
```

## Deterministic Audit Workflow

The deterministic audit pipeline is the current repo-local acceptance layer.

### Phase 1: Inventory + manifest

```bash
python scripts/build_audit_manifest.py
python scripts/validate_audit_manifest.py
```

This rebuilds the committed [audit_manifest.json](/Users/conrad/personal/ageo-atoms/data/audit_manifest.json)
and validates schema/state.

### Phase 2: Structural audit

```bash
python scripts/audit_structural.py
```

This normalizes structural findings across wrappers, witnesses, CDGs, and repo
verification.

### Phase 3: Deterministic semantic evidence

```bash
python scripts/audit_signature_fidelity.py
python scripts/audit_runtime_probes.py
python scripts/audit_return_fidelity.py
python scripts/audit_state_fidelity.py
python scripts/audit_generated_nouns.py
python scripts/audit_semantics.py
python scripts/audit_acceptability.py
python scripts/audit_risk.py
python scripts/report_parity_coverage.py
```

For single-atom work, use the `--atom-id` options where available after
refreshing the audit manifest.

### Phase 4: Review workflow

```bash
python scripts/init_audit_reviews.py --priority review_now --limit 10
python scripts/validate_audit_reviews.py
python scripts/apply_audit_reviews.py --only-completed
python scripts/report_audit_progress.py
```

Automation can score and triage atoms, but it does not promote an atom to
`trusted` on its own.

## Acceptance Rules

An ingest is ready to keep only when all of the following are true:

- the wrapper files are structurally valid
- the CDG is present and valid
- the atom is represented correctly in `data/audit_manifest.json`
- the deterministic semantic checks do not classify it as broken/misleading
- the risk queue and review state are acceptable for the intended change

High-risk atoms need review before they should be treated as trusted.

## What the Audit Manifest Is For

[audit_manifest.json](/Users/conrad/personal/ageo-atoms/data/audit_manifest.json) is now part of the
repo's committed metadata. It records:

- inventory facts
- structural rollups
- semantic rollups
- risk tier / review queue placement
- provenance and review-related fields

Generated reports in `data/audit/` remain local build outputs and are not the
canonical committed source of truth.

## Recommended Minimal Loop for One New Atom

```bash
# 1. Ingest
cd ../ageo-matcher
sciona ingest <source> --class <ClassName> --output ../ageo-atoms/ageoa/<domain>/<atom_name>

# 2. Back in ageo-atoms, refresh deterministic audit state
cd ../ageo-atoms
python scripts/build_audit_manifest.py
python scripts/validate_audit_manifest.py
python scripts/audit_structural.py
python scripts/audit_signature_fidelity.py
python scripts/audit_runtime_probes.py
python scripts/audit_return_fidelity.py
python scripts/audit_state_fidelity.py
python scripts/audit_generated_nouns.py
python scripts/audit_semantics.py
python scripts/audit_acceptability.py
python scripts/audit_risk.py
```

Add `references.json` and `uncertainty.json` when applicable.

## Do Not Rely On

- `COMPLETED.json` as proof that an atom is acceptable
- `trace.jsonl` as the only provenance record
- matcher CLI summary text as the final judgment
- old flat-layout assumptions for new generated atoms

The deterministic audit outputs now carry the repo-local quality judgment.
