#!/usr/bin/env python3
"""Collect per-atom references, sync to manifests, and generate BibTeX.

Usage:
    python scripts/build_references.py             # full sync + generate .bib
    python scripts/build_references.py --check-only # validate consistency (CI)
    python scripts/build_references.py --atom ageoa/biosppy/ecg_hamilton  # single atom
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGEOA = ROOT / 'ageoa'
REGISTRY_PATH = ROOT / 'data' / 'references' / 'registry.json'
BIB_PATH = ROOT / 'data' / 'references' / 'ageo_atoms.bib'
MANIFEST_PATH = ROOT / 'data' / 'hyperparams' / 'manifest.json'
ATOM_MANIFEST_PATH = ROOT / 'scripts' / 'atom_manifest.yml'


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def collect_atom_refs(atom_filter: str | None = None) -> list[dict]:
    """Walk ageoa/**/references.json and return parsed contents."""
    results = []
    pattern = '*/references.json' if atom_filter else '**/references.json'
    search_root = ROOT / atom_filter if atom_filter else AGEOA
    for path in sorted(search_root.rglob('references.json')):
        if '__pycache__' in path.parts:
            continue
        data = json.loads(path.read_text())
        data['_path'] = str(path.relative_to(ROOT))
        results.append(data)
    return results


def validate(registry: dict, atom_refs_list: list[dict]) -> list[str]:
    """Return a list of validation errors."""
    errors = []
    known_ids = set(registry.get('references', {}).keys())

    # Every registry entry must have at least doi or url
    for ref_id, ref in registry.get('references', {}).items():
        if not ref.get('doi') and not ref.get('url'):
            errors.append(f'Registry: {ref_id} has neither doi nor url')

    # Every per-atom ref_id must exist in registry
    for atom_data in atom_refs_list:
        src = atom_data.get('_path', '?')
        for ref_id in atom_data.get('references', []):
            if ref_id not in known_ids:
                errors.append(f'{src}: ref_id "{ref_id}" not found in registry')

    return errors


def sync_hyperparams_manifest(registry: dict, atom_refs_list: list[dict]) -> int:
    """Add scholarly_references arrays to manifest.json entries. Returns count of atoms updated."""
    manifest = json.loads(MANIFEST_PATH.read_text())

    # Build atom_id -> [ref_ids] mapping from per-atom files
    ref_map: dict[str, list[str]] = {}
    for atom_data in atom_refs_list:
        atom_id = atom_data.get('atom_id', '')
        if atom_id:
            ref_map[atom_id] = atom_data.get('references', [])

    updated = 0
    for entry in manifest.get('reviewed_atoms', []):
        aid = entry.get('atom_id', '')
        if aid in ref_map:
            entry['scholarly_references'] = ref_map[aid]
            updated += 1

    # Bump schema version
    if manifest.get('schema_version') == '0.3':
        manifest['schema_version'] = '0.4'

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + '\n')
    return updated


def generate_bibtex(registry: dict) -> str:
    """Generate a .bib file from the global registry."""
    entries = []
    for ref_id, ref in sorted(registry.get('references', {}).items()):
        raw = ref.get('bibtex_raw')
        if raw:
            entries.append(raw)
        else:
            # Synthesize a minimal entry from structured fields
            entry_type = 'article'
            if ref.get('type') == 'book':
                entry_type = 'book'
            elif ref.get('type') == 'thesis':
                entry_type = 'phdthesis'
            elif ref.get('type') == 'repository':
                entry_type = 'misc'
            elif ref.get('type') == 'web':
                entry_type = 'misc'

            key = ref.get('bibtex_key', ref_id)
            lines = [f'@{entry_type}{{{key},']
            if ref.get('authors'):
                lines.append(f'  author = {{{" and ".join(ref["authors"])}}},')
            if ref.get('title'):
                lines.append(f'  title = {{{ref["title"]}}},')
            if ref.get('venue'):
                field = 'publisher' if entry_type == 'book' else 'journal'
                lines.append(f'  {field} = {{{ref["venue"]}}},')
            if ref.get('year'):
                lines.append(f'  year = {{{ref["year"]}}},')
            if ref.get('doi'):
                lines.append(f'  doi = {{{ref["doi"]}}},')
            if ref.get('url'):
                lines.append(f'  url = {{{ref["url"]}}},')
            lines.append('}')
            entries.append('\n'.join(lines))

    return '\n\n'.join(entries) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect references, sync manifests, generate BibTeX.')
    parser.add_argument('--check-only', action='store_true', help='Validate without writing')
    parser.add_argument('--atom', help='Restrict to a single atom directory (relative to repo root)')
    args = parser.parse_args()

    registry = load_registry()
    atom_refs_list = collect_atom_refs(args.atom)

    errors = validate(registry, atom_refs_list)
    if errors:
        for err in errors:
            print(f'ERROR: {err}', file=sys.stderr)
        if args.check_only:
            sys.exit(1)
        print(f'\n{len(errors)} validation error(s) found.', file=sys.stderr)

    if args.check_only:
        print(f'Validated {len(atom_refs_list)} atom reference files against {len(registry.get("references", {}))} registry entries.')
        if not errors:
            print('All checks passed.')
        return

    # Sync to hyperparams manifest
    updated = sync_hyperparams_manifest(registry, atom_refs_list)
    print(f'Manifest: updated {updated} atom entries with scholarly_references')

    # Generate BibTeX
    bib_content = generate_bibtex(registry)
    BIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    BIB_PATH.write_text(bib_content)
    ref_count = len(registry.get('references', {}))
    print(f'BibTeX: wrote {ref_count} entries to {BIB_PATH.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
