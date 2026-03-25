#!/usr/bin/env python3
"""Build the deterministic audit manifest for all registered atoms."""

from __future__ import annotations

from auditlib.inventory import write_manifest


def main() -> None:
    payload = write_manifest()
    atoms = payload.get("atoms", [])
    errors = payload.get("inventory_errors", [])
    print(f"Wrote data/audit_manifest.json with {len(atoms)} atoms")
    if errors:
        print(f"Inventory errors: {len(errors)}")


if __name__ == "__main__":
    main()
