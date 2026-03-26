"""Test-side fixture loader — thin re-export of scripts/io_codec.

Allows tests to ``from tests.io_codec import load_fixture`` without
reaching into ``scripts/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make scripts/ importable
_scripts = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

from io_codec import (  # noqa: F401, E402
    deserialize_inputs,
    deserialize_output,
    deserialize_value,
    load_fixture,
    save_fixture,
    serialize_args,
    serialize_value,
)
