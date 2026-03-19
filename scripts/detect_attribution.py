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
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGEOA = ROOT / 'ageoa'
REGISTRY_PATH = ROOT / 'data' / 'references' / 'registry.json'

# Known algorithm name patterns mapped to ref_ids.
# This table grows as references are added to the registry.
# Keys are lowercased regex patterns; values are (ref_id, base_score) tuples.
NAME_PATTERNS: dict[str, tuple[str, float]] = {
    r'hamilton': ('hamilton1986', 0.8),
    r'engzee|engelse.*zeelenberg': ('engzee1979', 0.8),
    r'monte.?carlo': ('glasserman2003', 0.6),
}


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def load_cdg(atom_dir: Path) -> dict | None:
    cdg_path = atom_dir / 'cdg.json'
    if not cdg_path.exists():
        return None
    return json.loads(cdg_path.read_text())


def load_atom_refs(atom_dir: Path) -> dict:
    refs_path = atom_dir / 'references.json'
    if refs_path.exists():
        return json.loads(refs_path.read_text())
    return {
        'schema_version': '1.0',
        'atom_id': '',
        'references': [],
        'auto_attribution_runs': [],
    }


def save_atom_refs(atom_dir: Path, data: dict) -> None:
    refs_path = atom_dir / 'references.json'
    refs_path.write_text(json.dumps(data, indent=2) + '\n')


def name_heuristic_match(cdg: dict, registry: dict, threshold: float) -> list[dict]:
    """Match CDG node names against known algorithm name patterns."""
    matches = []
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
                    matches.append({
                        'ref_id': ref_id,
                        'node_id': node.get('node_id', ''),
                        'similarity_score': base_score,
                        'match_type': 'name_heuristic',
                    })

    # Deduplicate by ref_id, keeping highest score
    best: dict[str, dict] = {}
    for m in matches:
        rid = m['ref_id']
        if rid not in best or m['similarity_score'] > best[rid]['similarity_score']:
            best[rid] = m
    return list(best.values())


def process_atom(atom_dir: Path, registry: dict, threshold: float, dry_run: bool) -> list[dict]:
    """Run attribution detection for a single atom. Returns matches found."""
    cdg = load_cdg(atom_dir)
    if cdg is None:
        return []

    matches = name_heuristic_match(cdg, registry, threshold)

    if dry_run:
        return matches

    if matches:
        atom_refs = load_atom_refs(atom_dir)

        for match in matches:
            if match['ref_id'] not in atom_refs['references']:
                atom_refs['references'].append(match['ref_id'])

        atom_refs['auto_attribution_runs'].append({
            'run_id': uuid.uuid4().hex[:12],
            'engine_version': '0.1',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'candidates_evaluated': len(registry.get('references', {})),
            'threshold': threshold,
            'matches_above_threshold': len(matches),
        })

        save_atom_refs(atom_dir, atom_refs)

    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description='Detect attribution via AST/name similarity.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--atom', help='Single atom directory (relative to repo root)')
    group.add_argument('--all', action='store_true', help='Process all atoms with cdg.json')
    parser.add_argument('--threshold', type=float, default=0.7, help='Minimum similarity score (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true', help='Print matches without saving')
    args = parser.parse_args()

    registry = load_registry()

    if args.atom:
        atom_dir = ROOT / args.atom
        if not atom_dir.is_dir():
            print(f'Error: {atom_dir} is not a directory', file=sys.stderr)
            sys.exit(1)
        matches = process_atom(atom_dir, registry, args.threshold, args.dry_run)
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
            matches = process_atom(atom_dir, registry, args.threshold, args.dry_run)
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
