#!/usr/bin/env python3
"""CLI to add scholarly references to atoms.

Resolves DOIs via the CrossRef API, upserts into the global registry,
and creates/updates per-atom references.json files.

Usage:
    python scripts/add_reference.py \
        --atom ageoa/biosppy/ecg_hamilton \
        --doi "10.1109/TBME.1986.325695" \
        --match-type manual \
        --notes "Original Hamilton-Tompkins QRS detection algorithm"

    # Manual entry without DOI resolution:
    python scripts/add_reference.py \
        --atom ageoa/biosppy/ecg_hamilton \
        --ref-id hamilton1986 \
        --type paper \
        --url "https://example.com/paper.pdf" \
        --title "Some Paper Title" \
        --authors "A. Author" "B. Author" \
        --year 1986 \
        --match-type manual
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / 'data' / 'references' / 'registry.json'


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {'schema_version': '1.0', 'references': {}}


def save_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + '\n')


def resolve_doi(doi: str) -> dict | None:
    """Fetch metadata from CrossRef for a DOI."""
    url = f'https://api.crossref.org/works/{doi}'
    req = urllib.request.Request(url, headers={'Accept': 'application/json', 'User-Agent': 'ageo-atoms/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f'Warning: CrossRef lookup failed for {doi}: {exc}', file=sys.stderr)
        return None
    msg = data.get('message', {})
    authors = []
    for a in msg.get('author', []):
        given = a.get('given', '')
        family = a.get('family', '')
        authors.append(f'{given} {family}'.strip())
    titles = msg.get('title', [])
    containers = msg.get('container-title', [])
    issued = msg.get('issued', {}).get('date-parts', [[None]])[0]
    return {
        'title': titles[0] if titles else '',
        'authors': authors,
        'year': issued[0] if issued else None,
        'venue': containers[0] if containers else '',
    }


def fetch_bibtex(doi: str) -> str | None:
    """Fetch BibTeX from doi.org content negotiation."""
    url = f'https://doi.org/{doi}'
    req = urllib.request.Request(url, headers={'Accept': 'application/x-bibtex', 'User-Agent': 'ageo-atoms/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode('utf-8').strip()
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f'Warning: BibTeX fetch failed for {doi}: {exc}', file=sys.stderr)
        return None


def make_ref_id(authors: list[str], year: int | None) -> str:
    """Generate a ref_id slug from first author surname and year."""
    if authors:
        surname = authors[0].split()[-1].lower()
        surname = re.sub(r'[^a-z]', '', surname)
    else:
        surname = 'unknown'
    year_str = str(year) if year else '0000'
    return f'{surname}{year_str}'


def ensure_unique_ref_id(ref_id: str, registry: dict) -> str:
    """Append a-z suffix if ref_id already exists with different content."""
    if ref_id not in registry['references']:
        return ref_id
    for suffix in 'abcdefghijklmnopqrstuvwxyz':
        candidate = f'{ref_id}{suffix}'
        if candidate not in registry['references']:
            return candidate
    raise ValueError(f'Too many references with base id {ref_id}')


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


def main() -> None:
    parser = argparse.ArgumentParser(description='Add a scholarly reference to an atom.')
    parser.add_argument('--atom', required=True, help='Atom directory path relative to repo root (e.g. ageoa/biosppy/ecg_hamilton)')
    parser.add_argument('--doi', help='DOI to resolve via CrossRef')
    parser.add_argument('--url', help='URL (required if no DOI)')
    parser.add_argument('--ref-id', help='Override ref_id slug (auto-generated from authors+year if omitted)')
    parser.add_argument('--type', default='paper', choices=['paper', 'repository', 'web', 'book', 'thesis', 'standard'])
    parser.add_argument('--title', help='Manual title (overrides CrossRef)')
    parser.add_argument('--authors', nargs='+', help='Manual author list (overrides CrossRef)')
    parser.add_argument('--year', type=int, help='Manual year (overrides CrossRef)')
    parser.add_argument('--venue', help='Journal or conference name')
    parser.add_argument('--match-type', default='manual', choices=['manual', 'ast_subgraph', 'name_heuristic'])
    parser.add_argument('--confidence', default='high', choices=['high', 'medium', 'low'])
    parser.add_argument('--notes', default='', help='Attribution notes')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be written without saving')
    args = parser.parse_args()

    if not args.doi and not args.url:
        parser.error('At least one of --doi or --url is required.')

    atom_dir = ROOT / args.atom
    if not atom_dir.is_dir():
        print(f'Error: atom directory not found: {atom_dir}', file=sys.stderr)
        sys.exit(1)

    # Resolve metadata
    crossref = {}
    bibtex_raw = None
    if args.doi:
        crossref = resolve_doi(args.doi) or {}
        bibtex_raw = fetch_bibtex(args.doi)

    title = args.title or crossref.get('title', '')
    authors = args.authors or crossref.get('authors', [])
    year = args.year or crossref.get('year')
    venue = args.venue or crossref.get('venue', '')

    ref_id = args.ref_id or make_ref_id(authors, year)

    registry = load_registry()
    ref_id = ensure_unique_ref_id(ref_id, registry)

    reference = {
        'ref_id': ref_id,
        'type': args.type,
    }
    if args.doi:
        reference['doi'] = args.doi
    if args.url:
        reference['url'] = args.url
    if title:
        reference['title'] = title
    if authors:
        reference['authors'] = authors
    if year:
        reference['year'] = year
    if venue:
        reference['venue'] = venue
    if bibtex_raw:
        reference['bibtex_raw'] = bibtex_raw
    reference['match_metadata'] = {
        'similarity_score': None,
        'match_type': args.match_type,
        'matched_nodes': [],
        'confidence': args.confidence,
        'notes': args.notes,
    }

    if args.dry_run:
        print(json.dumps(reference, indent=2))
        return

    # Upsert into registry
    registry['references'][ref_id] = reference
    save_registry(registry)
    print(f'Registry: upserted {ref_id}')

    # Update per-atom references.json
    atom_refs = load_atom_refs(atom_dir)
    if ref_id not in atom_refs['references']:
        atom_refs['references'].append(ref_id)
    save_atom_refs(atom_dir, atom_refs)
    print(f'Atom: updated {atom_dir / "references.json"}')


if __name__ == '__main__':
    main()
