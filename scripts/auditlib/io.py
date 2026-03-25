"""JSON and filesystem helpers for audit tooling."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    """Create a directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    """Read JSON from disk."""
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    """Write indented JSON to disk."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def safe_atom_stem(atom_id: str) -> str:
    """Make a stable filesystem-safe stem for an atom id."""
    compact = re.sub(r"[^A-Za-z0-9._-]+", "_", atom_id).strip("_")
    compact = compact[:96] if compact else "atom"
    digest = hashlib.sha1(atom_id.encode("utf-8")).hexdigest()[:12]
    return f"{compact}__{digest}"
