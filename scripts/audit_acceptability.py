#!/usr/bin/env python3
"""Score deterministic acceptability for atoms in the audit manifest."""

from __future__ import annotations

import argparse

from auditlib.acceptability import score_acceptability, write_acceptability_result, write_scores_csv
from auditlib.io import read_json, safe_atom_stem, write_json
from auditlib.paths import AUDIT_EVIDENCE_DIR, AUDIT_MANIFEST_PATH


def _load_signature_evidence(atom_id: str) -> dict | None:
    path = AUDIT_EVIDENCE_DIR / f"{safe_atom_stem(atom_id)}.json"
    if not path.exists():
        return None
    return read_json(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atom-id", help="Only process one atom id")
    args = parser.parse_args()

    manifest = read_json(AUDIT_MANIFEST_PATH)
    enriched_rows = []
    csv_rows = []
    for record in manifest.get("atoms", []):
        if args.atom_id and record["atom_id"] != args.atom_id:
            enriched_rows.append(record)
            continue
        evidence = _load_signature_evidence(record["atom_id"])
        result = score_acceptability(record, evidence)
        record["acceptability_score"] = result["acceptability_score"]
        record["acceptability_band"] = result["acceptability_band"]
        record["max_reviewable_verdict"] = result["max_reviewable_verdict"]
        record["overall_verdict"] = result["overall_verdict"]
        record["blocking_findings"] = sorted(
            dict.fromkeys(list(record.get("blocking_findings", [])) + list(result["hard_blockers"]))
        )
        record["required_actions"] = list(
            dict.fromkeys(list(record.get("required_actions", [])) + list(result["required_actions"]))
        )
        record["references_status"] = result["dimension_evidence"]["references_status"]
        record["parity_test_status"] = result["dimension_evidence"]["parity_test_status"]
        write_acceptability_result(result)
        enriched_rows.append(record)
        csv_rows.append(result)

    manifest["atoms"] = enriched_rows
    write_json(AUDIT_MANIFEST_PATH, manifest)
    write_scores_csv(csv_rows)
    print(f"Scored {len(csv_rows)} atom(s)")


if __name__ == "__main__":
    main()
