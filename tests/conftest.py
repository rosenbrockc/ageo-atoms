"""pytest configuration for ageoa test suite."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure scripts/ is importable for io_codec
_scripts = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)


def pytest_collection_modifyitems(config, items):
    """Skip parity tests when no fixtures exist yet."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    has_fixtures = any(fixtures_dir.rglob("*.json"))

    if not has_fixtures:
        skip = __import__("pytest").mark.skip(reason="No fixture files — run harvest_io.py first")
        for item in items:
            if "parity" in item.nodeid:
                item.add_marker(skip)
