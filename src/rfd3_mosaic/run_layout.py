"""Canonical date-first physical layout for Mosaic executions."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def utc_run_day(value: object | None = None) -> str:
    """Return a normalized UTC execution day in ``YYYY-MM-DD`` form."""

    if value is None:
        parsed = datetime.now(timezone.utc)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("Run timestamp cannot be empty")
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        parsed = parsed.astimezone(timezone.utc)
    return parsed.date().isoformat()


def safe_layout_name(value: object, *, label: str) -> str:
    """Return one readable, path-safe physical directory name."""

    cleaned = _SAFE_NAME.sub("-", str(value)).strip("-.")
    if not cleaned:
        raise ValueError(f"{label} must contain a path-safe character")
    return cleaned[:120]


def dated_experiment_root(
    root: str | Path,
    *,
    run_day: str,
    experiment: str,
) -> Path:
    """Return ``ROOT/YYYY-MM-DD/EXPERIMENT`` without creating it."""

    normalized_day = utc_run_day(f"{run_day}T00:00:00+00:00")
    return (
        Path(root).expanduser().resolve()
        / normalized_day
        / safe_layout_name(experiment, label="experiment")
    )


def dated_run_directory(
    root: str | Path,
    *,
    run_day: str,
    experiment: str,
    job_id: str,
) -> Path:
    """Return the canonical physical directory for one execution."""

    return dated_experiment_root(
        root,
        run_day=run_day,
        experiment=experiment,
    ) / safe_layout_name(job_id, label="job_id")
