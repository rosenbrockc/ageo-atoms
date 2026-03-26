#!/usr/bin/env python3
"""Validate semantic review records and build a review index."""

from __future__ import annotations

import sys

from auditlib.reviews import validate_review_directory


def main() -> None:
    payload = validate_review_directory()
    summary = payload["summary"]
    print(
        "Review validation: "
        f"ok={payload['ok']} errors={summary['error_count']} warnings={summary['warning_count']}"
    )
    if not payload["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
