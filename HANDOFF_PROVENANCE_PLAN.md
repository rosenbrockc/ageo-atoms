# Handoff Provenance Plan

This document is the implementation plan for improving artifact provenance at
the boundary between `ageo-atoms` and `../ageo-matcher`.

The immediate goal is to make the ingest handoff durable enough that this
repository can keep only the atom artifacts that matter while dropping
operational byproducts such as `.ingest_status.json`, `COMPLETED.json`,
`FAILED.json`, `trace.jsonl`, and `shared_context_metrics.json`.

The current system already preserves the core products:

- `atoms.py`
- `witnesses.py`
- `state_models.py` where needed
- `cdg.json`
- `matches.json`
- `references.json`
- `uncertainty.json` where measured

What it does not preserve well is *how* those products were produced. That
provenance is currently fragmented across:

- transient monitor files in `ageoa/**`
- CLI stdout
- optional `trace.jsonl`
- matcher-side telemetry such as `planner_artifacts.json`
- local agent memory / docs

## Scope

This plan covers:

- ingest-time provenance for generated atom artifacts
- cross-repo reproducibility between `ageo-atoms` and `../ageo-matcher`
- stable artifact manifests suitable for version control
- audit integration in `ageo-atoms`

This plan does not cover:

- full runtime telemetry retention
- deep benchmark storage
- replacement of `cdg.json`, `matches.json`, or `uncertainty.json`

## Problem Statement

Today, a committed atom directory can tell us *what* artifacts exist, but it
often cannot tell us enough about *where they came from*:

- which `../ageo-matcher` commit produced them
- which ingest mode generated them
- which source file and class were ingested
- whether the artifact came from procedural ingest, structured ingest, or a
  retry/refinement variant
- whether the `matches.json` file reflects a specific matcher index snapshot
- what verification outcomes were observed at generation time
- which optional sidecars were present at generation time, even if they are no
  longer committed

That makes it harder to:

- rerun or compare ingest output deterministically
- interpret `matches.json` or `uncertainty.json` years later
- distinguish canonical outputs from ephemeral run clutter
- connect audit findings in this repo back to the exact matcher/ingester state

## Target State

Each atom directory that contains generated artifacts should also contain one
small committed provenance manifest:

- `artifact_provenance.json`

This file becomes the committed summary of the handoff from
`../ageo-matcher` into `ageo-atoms`.

It should preserve enough information to answer:

- what command mode created the artifact set
- which source code was ingested
- which matcher commit/config/index snapshot was used
- which artifacts were produced
- which verification outcomes were observed
- whether `matches.json` and `uncertainty.json` are expected and current

The file should be stable, compact, and reviewable. It should not try to
replace `trace.jsonl`.

## Phase Breakdown

### Phase 1: Define the canonical provenance schema

Create a schema for `artifact_provenance.json` and document it in both repos.

Required top-level fields:

- `schema_version`
- `producer`
- `source`
- `generation`
- `artifacts`
- `verification`
- `hints`

Required nested fields:

- `producer.repo`
  Value: `"../ageo-matcher"` logical source
- `producer.git_commit`
  Commit SHA of the matcher repo that produced the artifacts
- `producer.git_dirty`
  Boolean flag so we know whether the run came from a clean matcher tree
- `producer.cli_command`
  Exact command argv used to generate the atom output
- `producer.cli_subcommand`
  Usually `ingest`
- `producer.version`
  Optional semver/build string if the matcher exposes one

- `source.path`
  Path of the ingested upstream source file
- `source.class_name`
  Class/module/function target passed to the ingester
- `source.language`
  Language detected by matcher
- `source.procedural`
  Whether `--procedural` was used
- `source.max_depth`
  Effective decomposition depth

- `generation.generated_at`
  UTC ISO timestamp
- `generation.run_id`
  Stable ingest run id from the matcher monitor
- `generation.output_dir`
  Output directory used by matcher
- `generation.execution_mode`
  One of: `stateful`, `procedural`, `ffi`, `retry`, `refinement`
- `generation.retry_of`
  Optional prior run id or artifact hash if this was a retry

- `artifacts.atoms_py`
- `artifacts.witnesses_py`
- `artifacts.state_models_py`
- `artifacts.cdg_json`
- `artifacts.matches_json`
- `artifacts.references_json`
- `artifacts.uncertainty_json`
  Each entry should include:
  - `path`
  - `exists`
  - `sha256`
  - `size_bytes`

- `verification.mypy_passed`
- `verification.ghost_sim_passed`
- `verification.cdg_nodes`
- `verification.cdg_edges`
- `verification.match_count`
- `verification.failure_reason`
  Optional when generation failed or only partial artifacts were published

- `hints.expected_ephemeral_artifacts`
  Example: `.ingest_status.json`, `trace.jsonl`, `shared_context_metrics.json`
- `hints.notes`
  Freeform compact notes for unusual generation cases

Exit criteria:

- schema written down in both repos
- representative example committed in docs/tests

### Phase 2: Emit provenance from `../ageo-matcher`

Update `sciona ingest` to emit `artifact_provenance.json` atomically alongside
the published artifacts.

Implementation tasks in `../ageo-matcher`:

- extend `sciona/ingester/monitor.py` or a new helper module to stage the
  provenance manifest
- collect git metadata from the matcher repo at command start
- include run id and summary information already present in monitor state
- compute SHA-256 for published stable artifacts after publish
- write `artifact_provenance.json` on both success and partial-failure paths

Recommended files to change:

- `sciona/ingester/monitor.py`
- `sciona/commands/ingest_cmds.py`
- possibly `sciona/commands/shared_context_helpers.py` if metrics provenance is
  folded in

Recommended tests in `../ageo-matcher`:

- success path emits manifest
- partial publish emits manifest with `failure_reason`
- hashes are present for `cdg.json` and `matches.json`
- procedural ingest populates `source.procedural`
- dirty matcher worktree is reflected in `producer.git_dirty`

Exit criteria:

- new manifest emitted for every ingest run
- no dependence on `.ingest_status.json` or `COMPLETED.json` for long-term
  provenance

### Phase 3: Ingest the provenance in `ageo-atoms`

Teach local audit and inventory tooling to read `artifact_provenance.json`.

Implementation tasks in `ageo-atoms`:

- extend `scripts/auditlib/inventory.py`
- add explicit `has_artifact_provenance` field
- load summary fields into the audit manifest:
  - `ingest_run_id`
  - `matcher_commit`
  - `matcher_dirty`
  - `ingest_mode`
  - `source_path`
  - `source_language`
  - `artifact_hashes`
  - `generation_timestamp`
- treat `artifact_provenance.json` as the durable source for
  `generated_ingest` classification instead of inferring that from
  `.ingest_status.json` / `COMPLETED.json` / `trace.jsonl`

Recommended files to change:

- `scripts/auditlib/inventory.py`
- `scripts/auditlib/models.py`
- `scripts/auditlib/manifest_validation.py`
- `scripts/build_audit_manifest.py`

Recommended tests:

- inventory loads manifest correctly for generated atom directories
- absence of `.ingest_status.json` no longer affects generated/hand-written
  classification when provenance exists
- audit manifest captures matcher commit and source path

Exit criteria:

- audit manifest no longer needs ephemeral status files to classify generated
  atoms

### Phase 4: Bridge to audit/review provenance

Use the ingest provenance to strengthen the Phase 4/7 review artifacts.

Implementation tasks:

- seed review drafts with matcher commit, source path, and generation timestamp
- add `artifact_provenance_path` and `artifact_hashes` to review inputs
- include provenance mismatch warnings when:
  - `matches.json` hash changes without provenance refresh
  - `cdg.json` hash changes without provenance refresh
  - `references.json` or `uncertainty.json` exist but are absent from provenance

Recommended files:

- `scripts/auditlib/reviews.py`
- `scripts/auditlib/review_pass.py`
- `scripts/validate_audit_reviews.py`

Exit criteria:

- review records can cite the exact ingest artifact set that was reviewed

### Phase 5: Remove remaining dependence on ephemeral sidecars

After Phases 2-4 land, remove all audit or policy dependence on:

- `.ingest_status.json`
- `COMPLETED.json`
- `FAILED.json`
- `trace.jsonl`
- `shared_context_metrics.json`

These may still be produced locally during runs, but they should no longer be
required for repository state or audit interpretation.

Exit criteria:

- a clean checkout with only committed stable artifacts is fully auditable

## Detailed Schema Proposal

Example:

```json
{
  "schema_version": "1.0",
  "producer": {
    "repo": "../ageo-matcher",
    "git_commit": "abc123...",
    "git_dirty": false,
    "cli_subcommand": "ingest",
    "cli_command": [
      "sciona",
      "ingest",
      "third_party/astroflow/dedisp_ingest.py",
      "--class",
      "AstroflowDedispIngest",
      "--output",
      "ageoa/astroflow"
    ],
    "version": null
  },
  "source": {
    "path": "third_party/astroflow/dedisp_ingest.py",
    "class_name": "AstroflowDedispIngest",
    "language": "python",
    "procedural": false,
    "max_depth": 12
  },
  "generation": {
    "generated_at": "2026-03-26T18:12:10Z",
    "run_id": "9ceea52ad0284e529e0506e56534f9a8",
    "output_dir": "ageoa/astroflow",
    "execution_mode": "stateful",
    "retry_of": null
  },
  "artifacts": {
    "atoms_py": {"path": "ageoa/astroflow/atoms.py", "exists": true, "sha256": "...", "size_bytes": 1234},
    "witnesses_py": {"path": "ageoa/astroflow/witnesses.py", "exists": true, "sha256": "...", "size_bytes": 456},
    "state_models_py": {"path": "ageoa/astroflow/state_models.py", "exists": false, "sha256": null, "size_bytes": null},
    "cdg_json": {"path": "ageoa/astroflow/cdg.json", "exists": true, "sha256": "...", "size_bytes": 789},
    "matches_json": {"path": "ageoa/astroflow/matches.json", "exists": true, "sha256": "...", "size_bytes": 999},
    "references_json": {"path": "ageoa/astroflow/references.json", "exists": false, "sha256": null, "size_bytes": null},
    "uncertainty_json": {"path": "ageoa/astroflow/uncertainty.json", "exists": false, "sha256": null, "size_bytes": null}
  },
  "verification": {
    "mypy_passed": false,
    "ghost_sim_passed": false,
    "cdg_nodes": 5,
    "cdg_edges": 2,
    "match_count": 3,
    "failure_reason": null
  },
  "hints": {
    "expected_ephemeral_artifacts": [
      ".ingest_status.json",
      "COMPLETED.json",
      "trace.jsonl",
      "shared_context_metrics.json"
    ],
    "notes": ""
  }
}
```

## Migration Plan

### Step 1: Land matcher-side emitter

- implement schema and writer in `../ageo-matcher`
- do not remove any old files yet

### Step 2: Backfill manifests in this repo

- create a one-shot migration script in `ageo-atoms`, for example:
  `scripts/backfill_artifact_provenance.py`
- infer what we can from existing:
  - `.ingest_status.json`
  - `COMPLETED.json`
  - `FAILED.json`
  - `trace.jsonl`
  - current artifact hashes
- mark inferred fields explicitly with `"inferred": true` if needed

### Step 3: Update audit inventory and validation

- switch `generated_ingest` detection to prefer provenance manifests
- emit warnings when generated artifacts lack provenance

### Step 4: Drop ephemeral tracking from the repo

- once coverage is high enough, keep the provenance manifests and stable
  products only

## Open Questions

- Should `matches.json` remain committed long-term, or should only its hash and
  top-line counts remain after synthesis succeeds?
- Should `artifact_provenance.json` live per atom directory or in a central
  registry under `data/`?
- Should `uncertainty.json` hashes be included in provenance even when produced
  long after ingest?
- Do we want to capture matcher index identity, such as catalog hash or FAISS
  snapshot id, in addition to matcher git commit?

My recommendation:

- keep `artifact_provenance.json` per atom directory
- include artifact hashes for `cdg.json`, `matches.json`, `references.json`,
  and `uncertainty.json`
- add optional `matcher_index_id` once the matcher can expose it

## Restart Checklist

When resuming this work after a process restart:

1. Confirm the current cleanup policy in `.gitignore`.
2. Confirm whether `artifact_provenance.json` already exists anywhere.
3. In `../ageo-matcher`, inspect:
   - `sciona/ingester/monitor.py`
   - `sciona/commands/ingest_cmds.py`
   - `tests/test_cli_command_telemetry.py`
4. In `ageo-atoms`, inspect:
   - `scripts/auditlib/inventory.py`
   - `scripts/auditlib/models.py`
   - `scripts/build_audit_manifest.py`
5. Implement matcher-side emission first.
6. Only then change local audit classification to rely on the new manifest.

## Definition of Done

This plan is complete when:

- generated atom directories contain a stable committed provenance manifest
- `ageo-atoms` audit tooling can explain which matcher commit produced a given
  atom artifact set
- repo policy no longer needs to keep operational status/trace sidecars
- review artifacts can cite the exact generated artifact hashes they refer to
