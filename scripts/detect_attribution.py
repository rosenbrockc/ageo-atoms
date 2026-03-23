#!/usr/bin/env python3
"""Stub for automated attribution detection via AST sub-graph similarity.

Currently implements only name-heuristic matching: compares CDG node names
against known algorithm/method names in the reference registry.

Future versions will add AST sub-graph isomorphism and embedding-based
similarity scoring.

Usage:
    python scripts/detect_attribution.py --atom ageoa/biosppy/ecg_hamilton --threshold 0.5
    python scripts/detect_attribution.py --all --threshold 0.7
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGEOA = ROOT / 'ageoa'
REGISTRY_PATH = ROOT / 'data' / 'references' / 'registry.json'
MANIFEST_PATH = ROOT / 'data' / 'hyperparams' / 'manifest.json'

# Known algorithm name patterns mapped to ref_ids.
# This table grows as references are added to the registry.
# Keys are lowercased regex patterns; values are (ref_id, base_score) tuples.
NAME_PATTERNS: dict[str, tuple[str, float]] = {
    r'hamilton(?!ian)': ('hamilton1986', 0.8),
    r'engzee|engelse.*zeelenberg': ('engzee1979', 0.8),
    r'monte.?carlo': ('glasserman2003', 0.6),
}


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def is_registered(node: ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name) and target.id == 'register_atom':
            return True
        if isinstance(target, ast.Attribute) and target.attr == 'register_atom':
            return True
    return False


def discover_atom_ids(atom_dir: Path) -> list[str]:
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        rel_dir = atom_dir.relative_to(ROOT)
        matches = [
            entry.get('atom_id', '')
            for entry in manifest.get('reviewed_atoms', [])
            if Path(entry.get('path', '')).parent == rel_dir and entry.get('atom_id')
        ]
        matches = sorted(dict.fromkeys(matches))
        if matches:
            return matches

    registered: list[str] = []
    for py_path in sorted(atom_dir.glob('*.py')):
        tree = ast.parse(py_path.read_text())
        rel = py_path.relative_to(ROOT).with_suffix('')
        parts = list(rel.parts)
        if parts and parts[-1] == 'atoms':
            parts = parts[:-1]
        registered.extend(
            f'{".".join(parts)}.{node.name}@{py_path.relative_to(ROOT)}:{node.lineno}'
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and is_registered(node)
        )
    return sorted(dict.fromkeys(registered))


def load_cdg(atom_dir: Path) -> dict | None:
    cdg_path = atom_dir / 'cdg.json'
    if not cdg_path.exists():
        return None
    return json.loads(cdg_path.read_text())


def load_atom_refs_document(atom_dir: Path) -> dict:
    refs_path = atom_dir / 'references.json'
    if refs_path.exists():
        return json.loads(refs_path.read_text())
    return {'schema_version': '1.1', 'atoms': {}}


def resolve_target_atom_id(atom_dir: Path, requested_atom_id: str | None, data: dict) -> str:
    discovered = discover_atom_ids(atom_dir)
    if requested_atom_id:
        if discovered and requested_atom_id not in discovered:
            raise ValueError(f'{requested_atom_id} is not registered under {atom_dir.relative_to(ROOT)}')
        return requested_atom_id

    atoms = data.get('atoms')
    if isinstance(atoms, dict) and len(atoms) == 1:
        return next(iter(atoms))

    legacy_atom_id = data.get('atom_id')
    if legacy_atom_id:
        return legacy_atom_id

    if len(discovered) == 1:
        return discovered[0]

    rel = atom_dir.relative_to(ROOT)
    raise ValueError(f'{rel} contains multiple atoms; pass --atom-id to disambiguate')


def load_atom_refs(atom_dir: Path, requested_atom_id: str | None) -> tuple[dict, str, dict]:
    data = load_atom_refs_document(atom_dir)
    atom_id = resolve_target_atom_id(atom_dir, requested_atom_id, data)

    atoms = data.get('atoms')
    if isinstance(atoms, dict):
        payload = atoms.setdefault(atom_id, {})
        payload.setdefault('references', [])
        payload.setdefault('auto_attribution_runs', [])
        data['schema_version'] = '1.1'
        return data, atom_id, payload

    upgraded = {
        'schema_version': '1.1',
        'atoms': {
            atom_id: {
                'references': data.get('references', []),
                'auto_attribution_runs': data.get('auto_attribution_runs', []),
            }
        },
    }
    return upgraded, atom_id, upgraded['atoms'][atom_id]


def save_atom_refs(atom_dir: Path, data: dict) -> None:
    refs_path = atom_dir / 'references.json'
    refs_path.write_text(json.dumps(data, indent=2) + '\n')


def entry_ref_id(entry: str | dict) -> str:
    if isinstance(entry, str):
        return entry
    return entry.get('ref_id', '')


def score_confidence(score: float) -> str:
    if score >= 0.8:
        return 'high'
    if score >= 0.6:
        return 'medium'
    return 'low'


def upsert_match_entry(entries: list, match: dict) -> None:
    ref_id = match['ref_id']
    match_metadata = {
        'similarity_score': match['similarity_score'],
        'match_type': 'name_heuristic',
        'matched_nodes': match['matched_nodes'],
        'confidence': score_confidence(match['similarity_score']),
        'notes': 'Auto-detected from CDG node names/descriptions.',
    }
    ref_entry = {'ref_id': ref_id, 'match_metadata': match_metadata}

    for idx, entry in enumerate(entries):
        if entry_ref_id(entry) != ref_id:
            continue
        if isinstance(entry, dict):
            existing_meta = entry.get('match_metadata', {})
            if existing_meta.get('match_type') == 'manual':
                return
            merged_nodes = list(dict.fromkeys(existing_meta.get('matched_nodes', []) + match['matched_nodes']))
            similarity = max(existing_meta.get('similarity_score') or 0.0, match['similarity_score'])
            entries[idx] = {
                'ref_id': ref_id,
                'match_metadata': {
                    'similarity_score': similarity,
                    'match_type': 'name_heuristic',
                    'matched_nodes': merged_nodes,
                    'confidence': score_confidence(similarity),
                    'notes': existing_meta.get('notes') or match_metadata['notes'],
                },
            }
        else:
            entries[idx] = ref_entry
        return

    entries.append(ref_entry)


def name_heuristic_match(cdg: dict, registry: dict, threshold: float) -> list[dict]:
    """Match CDG node names against known algorithm name patterns."""
    best: dict[str, dict] = {}
    known_ref_ids = set(registry.get('references', {}).keys())

    for node in cdg.get('nodes', []):
        node_name = node.get('name', '').lower()
        node_desc = node.get('description', '').lower()
        text = f'{node_name} {node_desc}'

        for pattern, (ref_id, base_score) in NAME_PATTERNS.items():
            if ref_id not in known_ref_ids:
                continue
            if re.search(pattern, text):
                if base_score >= threshold:
                    node_id = node.get('node_id', '')
                    match = best.get(ref_id)
                    if match is None:
                        best[ref_id] = {
                            'ref_id': ref_id,
                            'node_id': node_id,
                            'matched_nodes': [node_id] if node_id else [],
                            'similarity_score': base_score,
                            'match_type': 'name_heuristic',
                        }
                        continue
                    if node_id and node_id not in match['matched_nodes']:
                        match['matched_nodes'].append(node_id)
                    if base_score > match['similarity_score']:
                        match['similarity_score'] = base_score
                        match['node_id'] = node_id
    return list(best.values())


def process_atom(atom_dir: Path, registry: dict, threshold: float, dry_run: bool, atom_id: str | None = None) -> list[dict]:
    """Run attribution detection for a single atom. Returns matches found."""
    cdg = load_cdg(atom_dir)
    if cdg is None:
        return []

    matches = name_heuristic_match(cdg, registry, threshold)

    if dry_run:
        return matches

    if matches:
        atom_doc, resolved_atom_id, atom_refs = load_atom_refs(atom_dir, atom_id)

        for match in matches:
            upsert_match_entry(atom_refs['references'], match)

        atom_refs['auto_attribution_runs'].append({
            'run_id': uuid.uuid4().hex[:12],
            'engine_version': '0.1',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'candidates_evaluated': len(registry.get('references', {})),
            'threshold': threshold,
            'matches_above_threshold': len(matches),
            'matches': [
                {
                    'ref_id': match['ref_id'],
                    'matched_nodes': match['matched_nodes'],
                    'similarity_score': match['similarity_score'],
                    'match_type': match['match_type'],
                }
                for match in matches
            ],
        })

        save_atom_refs(atom_dir, atom_doc)

    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description='Detect attribution via AST/name similarity.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--atom', help='Single atom directory (relative to repo root)')
    group.add_argument('--all', action='store_true', help='Process all atoms with cdg.json')
    parser.add_argument('--atom-id', help='Exact manifest atom_id when the target directory contains multiple atoms')
    parser.add_argument('--threshold', type=float, default=0.7, help='Minimum similarity score (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true', help='Print matches without saving')
    args = parser.parse_args()

    registry = load_registry()

    if args.atom:
        atom_dir = ROOT / args.atom
        if not atom_dir.is_dir():
            print(f'Error: {atom_dir} is not a directory', file=sys.stderr)
            sys.exit(1)
        matches = process_atom(atom_dir, registry, args.threshold, args.dry_run, args.atom_id)
        if matches:
            print(f'{args.atom}: {len(matches)} match(es)')
            for m in matches:
                print(f'  {m["ref_id"]} (score={m["similarity_score"]:.2f}, node={m["node_id"]})')
        else:
            print(f'{args.atom}: no matches above threshold {args.threshold}')
    else:
        total_matches = 0
        total_atoms = 0
        for cdg_path in sorted(AGEOA.rglob('cdg.json')):
            if '__pycache__' in cdg_path.parts:
                continue
            atom_dir = cdg_path.parent
            try:
                matches = process_atom(atom_dir, registry, args.threshold, args.dry_run)
            except ValueError as exc:
                rel = atom_dir.relative_to(ROOT)
                print(f'{rel}: skipped ({exc})')
                continue
            total_atoms += 1
            if matches:
                total_matches += len(matches)
                rel = atom_dir.relative_to(ROOT)
                print(f'{rel}: {len(matches)} match(es)')
                for m in matches:
                    print(f'  {m["ref_id"]} (score={m["similarity_score"]:.2f}, node={m["node_id"]})')
        print(f'\nProcessed {total_atoms} atoms, found {total_matches} total matches.')


if __name__ == '__main__':
    main()
