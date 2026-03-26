# Ingestion Agent Prompt

Primary task: use `sciona ingest` from `../ageo-matcher` to produce a new atom
or refresh an existing one in `ageoa/`.

Secondary task: leave the repo in a state that passes the deterministic audit
pipeline in this repo. Ingestion is not complete just because the matcher CLI
ran successfully.

## What Counts As Success

A successful ingest now means all of the following are true:

- the stable atom artifacts are written to the correct `ageoa/` location
- transient ingest monitor files are not committed
- the atom is represented correctly in `data/audit_manifest.json`
- the deterministic audit tools do not leave the atom in a broken or
  misleading state
- the resulting risk/review state is acceptable for the change

## Current Output Layout

New generated atoms should use a per-atom directory:

```text
ageoa/
  <domain>/
    __init__.py
    <atom_name>/
      __init__.py
      atoms.py
      witnesses.py
      state_models.py      # only when stateful
      cdg.json
      matches.json         # optional matcher grounding output
      references.json      # optional curated scholarly metadata
      uncertainty.json     # optional post-ingest perturbation analysis
```

Legacy layouts still exist and remain valid:

- `ageoa/<domain>/<module>.py`
- `ageoa/<domain>/<stem>_witnesses.py`
- `ageoa/<domain>/<stem>_cdg.json`

For new work, prefer the per-atom directory layout unless there is a specific
reason to preserve a legacy shape.

## `sciona ingest`

Run ingestion from `../ageo-matcher`:

```bash
sciona ingest <source_file> --class <ClassName> --output <ageoa_target_dir> [options]
```

Typical options:

- `--procedural` for deterministic extraction without LLM chunking
- `--llm-provider <provider>`
- `--llm-model <model>`
- `--trace` when you need debug traces during development

Example:

```bash
cd ../ageo-matcher
sciona ingest path/to/source.py \
  --class MyPipeline \
  --output ../ageo-atoms/ageoa/my_domain/my_pipeline
```

Use an output path that already matches the final repo location. Do not ingest
to a scratch directory and treat that scratch tree as the committed artifact.

## Stable vs Ephemeral Files

### Keep And Commit

- `atoms.py`
- `witnesses.py`
- `state_models.py` when needed
- `cdg.json`
- legacy `*_cdg.json` where already in use
- `matches.json` when matcher grounding produced it
- `references.json` when the atom has source/reference metadata
- `uncertainty.json` when perturbation analysis has been run
- `__init__.py` updates

### Do Not Commit

These are operational byproducts, not durable repo state:

- `.ingest_status.json`
- `COMPLETED.json`
- `FAILED.json`
- `trace.jsonl`
- `shared_context_metrics.json`
- local `logs/`
- local `.playwright-mcp/`

If these appear during ingestion, leave them ignored or remove them from the
commit. They are not part of the atom contract.

## Generated Wrapper Contract

### `atoms.py`

- Every public atom must be fully type annotated.
- No public `Any`, `*args`, or `**kwargs`.
- Every public atom must be decorated with `@register_atom(...)`.
- Every public atom must have meaningful `@icontract.require` and
  `@icontract.ensure` coverage.
- Skeleton wrappers are acceptable only when they are explicit and honest.
- Stateful wrappers must use the generated state model instead of hidden
  ambient state.

### Decorator ordering

`@icontract.require` decorators run bottom-up. Keep the safest checks closest
to `def`:

1. `isinstance` checks
2. dimensional/shape checks
3. finiteness or more expensive checks
4. `@register_atom(...)` outermost

### `witnesses.py`

- Witnesses accept and return abstract ghost types only.
- Witnesses must be pure and side-effect free.
- Witnesses must propagate shape/dtype/domain semantics faithfully.
- Each public atom must have its own witness.

### `state_models.py`

- Only include state that truly must persist across calls/windows.
- Do not smuggle configuration or hidden mutable state into undocumented
  fields.

### `cdg.json`

- Must be structurally valid.
- Must describe the actual wrapper decomposition.
- New ingests should emit `cdg.json`.
- Legacy `*_cdg.json` stays valid for older atoms already using that shape.

## Additional Durable Sidecars

### `matches.json`

Keep it when produced. It is a real matcher handoff artifact and is not
centralized elsewhere in this repo.

### `references.json`

Use it for scholarly attribution and provenance metadata. It strengthens
reviewability and feeds the hyperparams/reference metadata build.

### `uncertainty.json`

Keep it when perturbation analysis has been run. It stores empirical
uncertainty work that is not otherwise preserved in the repo.

## Deterministic Audit Workflow

After ingesting or updating atoms, rerun the repo-local audit pipeline from
`ageo-atoms`.

### Required baseline

```bash
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
python scripts/report_parity_coverage.py
```

For focused work, use `--atom-id` on the per-atom audit scripts where
supported after refreshing the manifest.

### Review workflow

When the resulting atom or portfolio requires review:

```bash
python scripts/init_audit_reviews.py --priority review_now --limit 10
python scripts/validate_audit_reviews.py
python scripts/apply_audit_reviews.py --only-completed
python scripts/report_audit_progress.py
```

Deterministic tools can score, classify, and triage atoms, but they do not
promote atoms to `trusted` on their own.

## Agent Procedure

1. Choose the target output path under `ageoa/<domain>/<atom_name>`.
2. Run `sciona ingest` from `../ageo-matcher`.
3. Keep stable artifacts, discard or ignore monitor sidecars.
4. Update package exports in the relevant `__init__.py` files.
5. Add or preserve `references.json` when the atom has authoritative sources.
6. Add or preserve `uncertainty.json` when perturbation analysis exists.
7. Run the deterministic audit workflow in this repo.
8. Inspect the manifest, semantic status, acceptability, and risk outcome.
9. Do not treat the ingest as finished until the audit state is acceptable.

## Acceptance Checklist

```text
Artifacts:
  [ ] Stable files are in the correct ageoa/ location
  [ ] No monitor sidecars are being committed
  [ ] Package exports are updated

Wrapper quality:
  [ ] Public atoms are typed, registered, and contracted
  [ ] Witnesses use abstract types only
  [ ] CDG is present and structurally valid

Deterministic audit:
  [ ] audit manifest rebuilt and valid
  [ ] structural audit rerun
  [ ] semantic evidence rerun
  [ ] acceptability/risk rerun
  [ ] atom is not left broken or misleading

Review:
  [ ] high-risk atoms are queued for review when required
  [ ] no atom is treated as trusted without review-backed provenance
```

## Do Not Rely On

- `COMPLETED.json` as proof of correctness
- matcher CLI summary text as the final judgment
- `trace.jsonl` as the only provenance record
- old flat-layout assumptions for new generated atoms

The deterministic audit toolchain in this repo is now the authoritative local
acceptance layer for ingested atoms.
