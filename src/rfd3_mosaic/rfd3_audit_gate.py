"""Fail a pipeline only after all requested JSON audits have been written."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def failed_audit_paths(paths: list[Path]) -> list[Path]:
    """Return reports whose top-level ``passed`` value is not exactly true."""

    failed: list[Path] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("passed") is not True:
            failed.append(path)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        type=Path,
    )
    arguments = parser.parse_args()
    failed = failed_audit_paths(arguments.report)
    if failed:
        raise SystemExit(
            "Required result audits failed: "
            + ", ".join(path.name for path in failed)
        )
    print(
        "Required result audits: PASSED ("
        + ", ".join(path.name for path in arguments.report)
        + ")"
    )


if __name__ == "__main__":
    main()
