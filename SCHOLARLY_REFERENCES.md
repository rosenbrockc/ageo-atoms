# Scholarly References

This document describes how to find and persist scholarly references for atoms. It is written as a procedure that an agent (human or automated) follows when attributing an atom to its underlying publications, algorithms, or prior art.

## Overview

Every atom in `ageoa` wraps a concrete algorithm. Most of those algorithms were first described in a paper, book, or public repository. The reference system tracks these origins so that:

- Contributors to the original work are properly attributed.
- Users of the atom library can cite the correct source.
- Overlapping prior art is surfaced automatically when new atoms are added.

References live in three places, all kept in sync by tooling:

| Location | Role |
|---|---|
| `data/references/registry.json` | Single source of truth. Every reference object lives here, keyed by `ref_id`. |
| `ageoa/<module>/references.json` | Per-atom file listing which `ref_id`s apply to that atom. |
| `data/references/ageo_atoms.bib` | Generated BibTeX aggregating all registry entries. |

The hyperparams manifest (`data/hyperparams/manifest.json`, schema v0.4) and SQLite index carry a `scholarly_references` array per atom that mirrors the per-atom file. These are populated by `scripts/build_references.py`, not by hand.

## Reference Object Schema

Each entry in the registry conforms to `data/references/schema.json`. The key fields are:

```
ref_id          Short author-year slug (e.g. "hamilton1986"). Stable across all files.
type            paper | repository | web | book | thesis | standard
doi             Digital Object Identifier (without https://doi.org/ prefix). At least one of doi/url required.
url             Canonical URL. At least one of doi/url required.
title           Full title of the work.
authors         List of author name strings.
year            Publication year.
venue           Journal, conference, or publisher.
bibtex_raw      Complete BibTeX entry for lossless round-tripping.
match_metadata  How the reference was linked to the atom (see below).
```

The `match_metadata` object records provenance of the attribution itself:

```
similarity_score   0-1 float from the similarity engine, or null for manual attributions.
match_type         manual | ast_subgraph | name_heuristic
matched_nodes      List of CDG node_ids that triggered an automated match.
confidence         high | medium | low
notes              Free-text explanation.
```

## Procedure: Finding References for an Atom

### Step 1: Understand what the atom does

Read the atom's source and docstring. The wrapper is in `ageoa/<module>/atoms.py` and the upstream implementation is identified in `scripts/atom_manifest.yml`. The CDG file (`ageoa/<module>/cdg.json`) describes the algorithm's computational structure — node names and conceptual summaries are especially useful for identifying the method.

For example, `ageoa/biosppy/ecg_hamilton/atoms.py` wraps `hamilton_segmenter` from BioSPPy, which implements the Hamilton-Tompkins QRS detection algorithm. The CDG node names ("hamilton_segmenter_root") and the docstring ("Hamilton algorithm") directly name the method.

### Step 2: Search for the original publication

Use the algorithm name, author names, and domain to search for the source publication. Useful strategies:

1. **Name in the code.** Algorithm names in function/class identifiers or docstrings (e.g. "Hamilton", "Engelse-Zeelenberg", "Almgren-Chriss") often directly name the original authors. Search for `"<name> algorithm"` or `"<name> method"` plus the domain keywords.

2. **Upstream repository.** The repo listed in `REFERENCES.md` or `scripts/atom_manifest.yml` may cite papers in its own README, docs, or code comments. Check these first.

3. **Domain literature.** For well-known methods, textbooks and survey papers reliably identify the canonical citation. For example, Glasserman (2003) is the standard reference for Monte Carlo methods in finance.

4. **CrossRef / Google Scholar.** When you have a partial citation or title, resolve it to a DOI. The `add_reference.py` script can do this automatically given a DOI.

The goal is to find at least a DOI or a stable URL for each source. A DOI is strongly preferred because it enables automatic metadata resolution.

### Step 3: Check whether the reference already exists

Before creating a new entry, search the registry:

```bash
grep -i "hamilton" data/references/registry.json
```

If the `ref_id` already exists, skip to Step 5 (linking it to the atom). If a similar reference exists under a different key, use the existing one rather than creating a duplicate.

### Step 4: Add the reference to the registry

Use the CLI tool:

```bash
# With a DOI (preferred) — metadata and BibTeX are fetched automatically:
python scripts/add_reference.py \
    --atom ageoa/biosppy/ecg_hamilton \
    --doi "10.1109/TBME.1986.325695" \
    --match-type manual \
    --confidence high \
    --notes "Foundational paper for the Hamilton QRS detector implemented in BioSPPy."

# Without a DOI — provide fields manually:
python scripts/add_reference.py \
    --atom ageoa/biosppy/ecg_hamilton \
    --ref-id hamilton1986 \
    --type paper \
    --url "https://ieeexplore.ieee.org/document/4122029" \
    --title "Quantitative Investigation of QRS Detection Rules Using the MIT/BIH Arrhythmia Database" \
    --authors "P. S. Hamilton" "W. J. Tompkins" \
    --year 1986 \
    --venue "IEEE Transactions on Biomedical Engineering" \
    --match-type manual \
    --confidence high \
    --notes "Foundational paper for the Hamilton QRS detector."
```

This does two things:
1. Upserts the reference into `data/references/registry.json`.
2. Creates or updates `ageoa/biosppy/ecg_hamilton/references.json` with the `ref_id`.

Use `--dry-run` to preview the reference object without writing anything.

### Step 5: Link an existing reference to an atom

If the reference is already in the registry but not yet linked to a particular atom, run `add_reference.py` with the same `--doi` or `--ref-id`. The script is idempotent — it will not create a duplicate registry entry, and will append the `ref_id` to the atom's `references.json` only if it is not already present.

### Step 6: Sync manifests and generate BibTeX

After adding or linking references, run the build script to propagate changes:

```bash
python scripts/build_references.py
```

This:
1. Walks all `ageoa/**/references.json` files.
2. Updates `scholarly_references` arrays in `data/hyperparams/manifest.json`.
3. Regenerates `data/references/ageo_atoms.bib`.

Then rebuild the SQLite index so it includes the new scholarly_references table rows:

```bash
python scripts/build_hyperparams_manifest.py
```

### Step 7: Validate

```bash
python scripts/build_references.py --check-only
```

This verifies:
- Every `ref_id` referenced by a per-atom file exists in the registry.
- Every registry entry has at least a `doi` or `url`.

## Automated Attribution Detection

For bulk or exploratory attribution, the similarity detection stub can scan all atoms:

```bash
# Dry run — see what would be matched without writing:
python scripts/detect_attribution.py --all --threshold 0.7 --dry-run

# Single atom:
python scripts/detect_attribution.py --atom ageoa/biosppy/ecg_hamilton --threshold 0.5

# Write results (updates references.json with matches and logs the run):
python scripts/detect_attribution.py --all --threshold 0.7
```

The current engine uses name-heuristic matching: it compares CDG node names and descriptions against known algorithm name patterns in a lookup table (`NAME_PATTERNS` in `detect_attribution.py`). Matches above the threshold are added to the atom's `references.json` with `match_type: "name_heuristic"` and the run is recorded in `auto_attribution_runs`.

To extend the pattern table, add entries to `NAME_PATTERNS` mapping a regex to a `(ref_id, base_score)` tuple. The `ref_id` must exist in the registry.

Future versions will add AST sub-graph isomorphism and embedding-based similarity. The `match_metadata` schema already accommodates these — `match_type: "ast_subgraph"` with `matched_nodes` listing the specific CDG nodes, and a computed `similarity_score`.

## Adding References for a New Atom (Checklist)

When a new atom is ingested into the repository:

1. Read the atom wrapper and upstream implementation to identify the algorithm.
2. Search for the original publication(s). Prefer DOIs.
3. Check `data/references/registry.json` for existing entries.
4. Run `scripts/add_reference.py` for each new reference.
5. Run `scripts/detect_attribution.py --atom <path> --dry-run` to check for automated matches.
6. Run `scripts/build_references.py` and `scripts/build_hyperparams_manifest.py` to sync.
7. Run `scripts/build_references.py --check-only` to validate.

## File Layout

```
data/references/
    registry.json       Global reference registry (source of truth)
    schema.json         JSON Schema for reference objects
    ageo_atoms.bib      Generated BibTeX (do not hand-edit)

ageoa/<module>/
    references.json     Per-atom reference list + auto-attribution run log

scripts/
    add_reference.py        Add a reference (DOI resolution + registry upsert)
    build_references.py     Sync per-atom refs to manifests + generate .bib
    detect_attribution.py   Automated name-heuristic attribution detection
```
