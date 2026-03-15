#!/usr/bin/env python3
"""Harvest I/O fixtures from upstream Python repos.

Instruments upstream functions listed in ``atom_manifest.yml``, runs each
repo's test suite (or a synthetic probe), and serializes captured call
records to ``tests/fixtures/``.

Usage::

    python scripts/harvest_io.py                  # all Python entries
    python scripts/harvest_io.py --repo BioSPPy   # single repo
    python scripts/harvest_io.py --atom biosppy/ecg_hamilton:hamilton_segmentation
"""
from __future__ import annotations

import argparse
import functools
import importlib
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from io_codec import serialize_args, serialize_value, save_fixture

logger = logging.getLogger(__name__)

MANIFEST_PATH = ROOT / "scripts" / "atom_manifest.yml"
THIRD_PARTY = ROOT / "third_party"
FIXTURES_DIR = ROOT / "tests" / "fixtures"

MAX_RECORDS_PER_FUNCTION = 10  # keep fixtures small


# ---------------------------------------------------------------------------
# Recording wrapper
# ---------------------------------------------------------------------------


def make_recorder(
    module: Any,
    func_name: str,
    atom_key: str,
) -> list[dict]:
    """Replace *module.func_name* with a recording wrapper.

    Returns the mutable list that accumulates call records.
    """
    original = getattr(module, func_name)
    records: list[dict] = []

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if len(records) < MAX_RECORDS_PER_FUNCTION:
            try:
                record = {
                    "function": func_name,
                    "atom": atom_key,
                    "inputs": serialize_args(args, kwargs, original),
                    "output": serialize_value(result),
                }
                records.append(record)
            except Exception as exc:
                logger.debug("Serialization failed for %s: %s", func_name, exc)
        return result

    setattr(module, func_name, wrapper)
    return records


def restore_original(module: Any, func_name: str, original: Any) -> None:
    setattr(module, func_name, original)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_manifest(
    repo_filter: str | None = None,
    atom_filter: str | None = None,
) -> list[dict]:
    with open(MANIFEST_PATH) as f:
        entries = yaml.safe_load(f)

    result = []
    for entry in entries:
        up = entry["upstream"]
        if up.get("language") != "python":
            continue
        if repo_filter and up["repo"] != repo_filter:
            continue
        if atom_filter and entry["atom"] != atom_filter:
            continue
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Harvesting
# ---------------------------------------------------------------------------


def _fixture_path(atom_key: str) -> Path:
    """Map ``biosppy/ecg_hamilton:hamilton_segmentation`` → fixture path."""
    module_part, func_name = atom_key.split(":")
    return FIXTURES_DIR / module_part / f"{func_name}.json"


def harvest_with_tests(entries: list[dict]) -> dict[str, list[dict]]:
    """Instrument upstream functions and run repo tests.

    Returns ``{atom_key: [records]}`` for all atoms that captured data.
    """
    # Group entries by repo
    by_repo: dict[str, list[dict]] = {}
    for entry in entries:
        repo = entry["upstream"]["repo"]
        by_repo.setdefault(repo, []).append(entry)

    all_records: dict[str, list[dict]] = {}

    for repo, repo_entries in by_repo.items():
        repo_path = THIRD_PARTY / repo
        if not repo_path.is_dir():
            logger.warning("Repo %s not found at %s, skipping", repo, repo_path)
            continue

        # Add repo to sys.path
        sys.path.insert(0, str(repo_path))

        # Instrument all target functions
        originals: list[tuple[Any, str, Any]] = []
        entry_records: dict[str, list[dict]] = {}

        for entry in repo_entries:
            up = entry["upstream"]
            atom_key = entry["atom"]
            try:
                mod = importlib.import_module(up["module"])
                original = getattr(mod, up["function"])
                records = make_recorder(mod, up["function"], atom_key)
                originals.append((mod, up["function"], original))
                entry_records[atom_key] = records
                logger.info("Instrumented %s.%s → %s", up["module"], up["function"], atom_key)
            except Exception as exc:
                logger.warning("Cannot instrument %s: %s", atom_key, exc)

        # Try running repo tests
        _run_repo_tests(repo, repo_path)

        # If no test data captured, try synthetic probes
        for entry in repo_entries:
            atom_key = entry["atom"]
            records = entry_records.get(atom_key, [])
            if not records:
                logger.info("No test captures for %s, trying synthetic probe", atom_key)
                _synthetic_probe(entry, entry_records)

        # Restore originals
        for mod, func_name, original in originals:
            restore_original(mod, func_name, original)

        # Remove repo from sys.path
        sys.path.remove(str(repo_path))

        # Collect results
        for atom_key, records in entry_records.items():
            if records:
                all_records[atom_key] = records

    return all_records


def _run_repo_tests(repo: str, repo_path: Path) -> None:
    """Attempt to run the repo's test suite."""
    test_dirs = [
        repo_path / "tests",
        repo_path / "test",
    ]
    test_dir = next((d for d in test_dirs if d.is_dir()), None)

    if test_dir is None:
        logger.info("No test directory found for %s, skipping test run", repo)
        return

    try:
        import pytest  # noqa: F811

        logger.info("Running tests for %s from %s", repo, test_dir)
        pytest.main(
            [
                str(test_dir),
                "-x",            # stop on first failure
                "--no-header",
                "-q",
                "--tb=no",
                "-p", "no:warnings",
            ]
        )
    except Exception as exc:
        logger.warning("Test run failed for %s: %s", repo, exc)


def _synthetic_probe(entry: dict, records: dict[str, list[dict]]) -> None:
    """Generate synthetic inputs for an upstream function as a fallback."""
    import numpy as np

    up = entry["upstream"]
    atom_key = entry["atom"]

    try:
        mod = importlib.import_module(up["module"])
        func = getattr(mod, up["function"])
    except Exception:
        return

    # Inspect signature to build plausible inputs
    import inspect

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return

    params = list(sig.parameters.values())
    kwargs: dict[str, Any] = {}

    for p in params:
        name = p.name.lower()
        if name == "self":
            return  # Skip methods that need an instance
        if "signal" in name or name in ("a", "x", "data", "arr"):
            # Generate a plausible 1-D signal (ECG-like sine wave)
            t = np.linspace(0, 2, 2000)
            kwargs[p.name] = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.random.randn(len(t))
        elif "sampling_rate" in name or "fs" in name or name == "sr":
            kwargs[p.name] = 1000.0
        elif "rate" in name:
            kwargs[p.name] = 1000.0
        elif "threshold" in name or name == "pth":
            kwargs[p.name] = 0.5
        elif p.default is not inspect.Parameter.empty:
            kwargs[p.name] = p.default
        else:
            # Can't guess this parameter
            return

    try:
        result = func(**kwargs)
        record = {
            "function": up["function"],
            "atom": atom_key,
            "inputs": serialize_args((), kwargs, func),
            "output": serialize_value(result),
            "source": "synthetic_probe",
        }
        records.setdefault(atom_key, []).append(record)
        logger.info("Synthetic probe succeeded for %s", atom_key)
    except Exception as exc:
        logger.debug("Synthetic probe failed for %s: %s", atom_key, exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest I/O fixtures from upstream repos")
    parser.add_argument("--repo", help="Filter to a single upstream repo")
    parser.add_argument("--atom", help="Filter to a single atom key")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    entries = load_manifest(repo_filter=args.repo, atom_filter=args.atom)
    if not entries:
        logger.error("No matching manifest entries found")
        sys.exit(1)

    logger.info("Harvesting %d atom(s)", len(entries))
    all_records = harvest_with_tests(entries)

    saved = 0
    for atom_key, records in all_records.items():
        path = _fixture_path(atom_key)
        save_fixture(records, path)
        logger.info("Saved %d records → %s", len(records), path)
        saved += 1

    logger.info("Done: %d fixture files written", saved)


if __name__ == "__main__":
    main()
