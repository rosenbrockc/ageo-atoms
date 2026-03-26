#!/usr/bin/env python3
"""Generate per-domain parity test files from harvested fixtures.

While ``tests/test_parity.py`` dynamically discovers all fixtures,
this script generates explicit per-domain test files for better CI
reporting and IDE integration.

Usage::

    python scripts/generate_parity_tests.py
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "tests" / "fixtures"
TESTS_DIR = ROOT / "tests"

HEADER = '''\
"""Auto-generated parity tests for {domain} atoms.

DO NOT EDIT — regenerate with: python scripts/generate_parity_tests.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

# Import helpers from test_parity without requiring tests to be a package
_spec = importlib.util.spec_from_file_location(
    "test_parity", Path(__file__).parent / "test_parity.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_assert_outputs_match = _mod._assert_outputs_match
_deserialize = _mod._deserialize
_import_atom = _mod._import_atom

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "{domain}"

'''


def _generate_domain_test(domain: str, fixture_files: list[Path]) -> str:
    """Generate a test module string for one domain."""
    lines = [HEADER.format(domain=domain)]

    for fixture_file in sorted(fixture_files):
        func_name = fixture_file.stem
        rel_path = fixture_file.relative_to(FIXTURES_DIR)

        # Read fixture to get atom_key
        with open(fixture_file) as f:
            cases = json.load(f)
        if not cases:
            continue

        atom_key = cases[0]["atom"]
        # Use subdirectory + filename relative to domain dir
        rel_to_domain = fixture_file.relative_to(FIXTURES_DIR / domain)
        class_name = "Test" + "".join(
            part.capitalize() for part in func_name.replace("-", "_").split("_")
        )

        lines.append(
            textwrap.dedent(f"""\
            class {class_name}:
                FIXTURE = FIXTURES_DIR / "{rel_to_domain}"

                @pytest.fixture(autouse=True)
                def _load(self):
                    with open(self.FIXTURE) as f:
                        self.cases = json.load(f)

                @pytest.mark.parametrize("idx", range({len(cases)}))
                def test_parity(self, idx):
                    case = self.cases[idx]
                    atom_fn = _import_atom("{atom_key}")
                    inputs = _deserialize(case["inputs"])
                    expected = _deserialize(case["output"])
                    try:
                        result = atom_fn(**inputs)
                    except NotImplementedError:
                        pytest.skip("stub")
                    _assert_outputs_match(result, expected)

            """)
        )

    return "".join(lines)


def main() -> None:
    if not FIXTURES_DIR.is_dir():
        print("No fixtures directory found. Run harvest_io.py first.")
        return

    # Group fixture files by top-level domain
    by_domain: dict[str, list[Path]] = {}
    for fixture_file in FIXTURES_DIR.rglob("*.json"):
        rel = fixture_file.relative_to(FIXTURES_DIR)
        domain = rel.parts[0]
        by_domain.setdefault(domain, []).append(fixture_file)

    generated = 0
    for domain, files in sorted(by_domain.items()):
        content = _generate_domain_test(domain, files)
        out_path = TESTS_DIR / f"test_parity_{domain}.py"
        out_path.write_text(content)
        print(f"Generated {out_path} ({len(files)} fixture(s))")
        generated += 1

    print(f"Done: {generated} test file(s) generated")


if __name__ == "__main__":
    main()
