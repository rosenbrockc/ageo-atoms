#!/usr/bin/env python3
"""Harvest I/O fixtures from Rust upstream repos.

Since Rust functions can't be monkeypatched, this driver:
1. Builds the Rust crate's test binary with JSON-logging output
2. Parses captured stdout for structured I/O records
3. Falls back to running reference Python scripts that invoke the Rust
   binary with known inputs and capture outputs

Usage::

    python scripts/harvest_io_rust.py                  # all Rust entries
    python scripts/harvest_io_rust.py --repo mini-mcmc
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from io_codec import serialize_value, save_fixture

logger = logging.getLogger(__name__)

MANIFEST_PATH = ROOT / "scripts" / "atom_manifest.yml"
THIRD_PARTY = ROOT / "third_party"
FIXTURES_DIR = ROOT / "tests" / "fixtures"


def load_rust_entries(repo_filter: str | None = None) -> list[dict]:
    with open(MANIFEST_PATH) as f:
        entries = yaml.safe_load(f)
    return [
        e
        for e in entries
        if e["upstream"].get("language") == "rust"
        and (repo_filter is None or e["upstream"]["repo"] == repo_filter)
    ]


def _fixture_path(atom_key: str) -> Path:
    module_part, func_name = atom_key.split(":")
    return FIXTURES_DIR / module_part / f"{func_name}.json"


def harvest_cargo_test(repo: str, repo_path: Path) -> list[dict]:
    """Run ``cargo test`` and parse any JSON-formatted I/O lines from stdout.

    The convention is that test functions print lines starting with
    ``IO_RECORD:`` followed by a JSON object.
    """
    records: list[dict] = []
    try:
        result = subprocess.run(
            ["cargo", "test", "--", "--nocapture"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("IO_RECORD:"):
                try:
                    record = json.loads(line[len("IO_RECORD:"):])
                    records.append(record)
                except json.JSONDecodeError:
                    pass
        logger.info("cargo test for %s: %d I/O records captured", repo, len(records))
    except FileNotFoundError:
        logger.warning("cargo not found, skipping Rust test run for %s", repo)
    except subprocess.TimeoutExpired:
        logger.warning("cargo test timed out for %s", repo)
    except Exception as exc:
        logger.warning("cargo test failed for %s: %s", repo, exc)

    return records


def harvest_python_bindings(entry: dict) -> list[dict]:
    """If the Rust crate exposes Python bindings (pyo3), import and probe."""
    up = entry["upstream"]
    atom_key = entry["atom"]

    # Check for a Python module matching the crate name
    crate = up.get("crate", up["repo"]).replace("-", "_")
    try:
        mod = __import__(crate)
        func = getattr(mod, up["function"], None)
        if func is None:
            return []
        # Attempt a simple call — this is crate-specific
        logger.info("Found Python bindings for %s.%s", crate, up["function"])
        return []  # placeholder — specific probes would go here
    except ImportError:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest I/O fixtures from Rust repos")
    parser.add_argument("--repo", help="Filter to a single upstream repo")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    entries = load_rust_entries(repo_filter=args.repo)
    if not entries:
        logger.error("No matching Rust manifest entries found")
        sys.exit(1)

    # Group by repo
    by_repo: dict[str, list[dict]] = {}
    for entry in entries:
        repo = entry["upstream"]["repo"]
        by_repo.setdefault(repo, []).append(entry)

    saved = 0
    for repo, repo_entries in by_repo.items():
        repo_path = THIRD_PARTY / repo
        if not repo_path.is_dir():
            logger.warning("Repo %s not found at %s", repo, repo_path)
            continue

        # Try cargo test harvest
        cargo_records = harvest_cargo_test(repo, repo_path)

        # Match records to atoms by function name
        for entry in repo_entries:
            atom_key = entry["atom"]
            func_name = entry["upstream"]["function"]
            matched = [r for r in cargo_records if r.get("function") == func_name]

            # Fallback to Python bindings
            if not matched:
                matched = harvest_python_bindings(entry)

            if matched:
                path = _fixture_path(atom_key)
                save_fixture(matched, path)
                logger.info("Saved %d records → %s", len(matched), path)
                saved += 1
            else:
                logger.info("No records captured for %s", atom_key)

    logger.info("Done: %d fixture files written", saved)


if __name__ == "__main__":
    main()
