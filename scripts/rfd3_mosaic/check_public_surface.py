#!/usr/bin/env python3
"""Validate the maintained public documentation boundary."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
PUBLIC_DOCUMENTS = (
    REPOSITORY / "README.md",
    REPOSITORY / "CONTRIBUTING.md",
    REPOSITORY / "DEVELOPMENT_STATUS.md",
    REPOSITORY / "SECURITY.md",
    REPOSITORY / "PUBLIC_RELEASE_CHECKLIST.md",
    REPOSITORY / "examples" / "rfd3_mosaic" / "inputs" / "README.md",
    *(sorted((REPOSITORY / "docs" / "rfd3_mosaic").glob("*.md"))),
)
FORBIDDEN_TRACKED_FILES = {".env"}
FORBIDDEN_TRACKED_PREFIXES = (
    "docs/internal/",
    "reports/",
    "scripts/rfd3_mosaic/archive/legacy_direct/",
)
PRIVATE_PATTERNS = {
    "institution-specific deployment name": re.compile(r"\bLRZ\b", re.I),
    "private filesystem": re.compile(r"/dss/|dssfs|pn57ki", re.I),
    "private user path": re.compile(r"/home/haixi|\bre73rub2\b", re.I),
    "private login host": re.compile(r"login\.ai\.lrz", re.I),
    "historical numeric job id": re.compile(r"\b57[0-9]{5}\b"),
    "temporary outage language": re.compile(
        r"AI cluster is unavailable|three-day demo", re.I
    ),
}
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def main() -> None:
    failures: list[str] = []
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=REPOSITORY,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    for path in tracked:
        if path in FORBIDDEN_TRACKED_FILES:
            failures.append(f"{path}: local secret-bearing file is tracked")
        if path.startswith(FORBIDDEN_TRACKED_PREFIXES):
            failures.append(f"{path}: private development artifact is tracked")
        if "/" not in path and path.lower().endswith((".ppt", ".pptx")):
            failures.append(f"{path}: root-level personal presentation is tracked")

    for source in PUBLIC_DOCUMENTS:
        text = source.read_text(encoding="utf-8")
        for label, pattern in PRIVATE_PATTERNS.items():
            match = pattern.search(text)
            if match is not None:
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{source.relative_to(REPOSITORY)}:{line}: {label}")
        for target in MARKDOWN_LINK.findall(text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path_text = target.split("#", 1)[0]
            if not path_text:
                continue
            destination = (source.parent / path_text).resolve()
            if not destination.exists():
                failures.append(
                    f"{source.relative_to(REPOSITORY)}: missing link {target!r}"
                )

    if failures:
        raise SystemExit("Public-surface validation failed:\n- " + "\n- ".join(failures))
    print(f"RFD3-Mosaic public surface: PASSED ({len(PUBLIC_DOCUMENTS)} documents)")


if __name__ == "__main__":
    main()
