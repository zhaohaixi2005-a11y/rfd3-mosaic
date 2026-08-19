"""Safe migration of historical runs into the date-first physical layout."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rfd3_mosaic.run_index import list_run_records, relocate_run_record
from rfd3_mosaic.run_layout import dated_run_directory, safe_layout_name, utc_run_day


SCHEMA_VERSION = 1
_MOVABLE_STATES = frozenset({"completed", "failed"})


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _submission_target(
    root: Path,
    *,
    run_day: str,
    experiment: str,
    source: Path,
) -> Path:
    return (
        root
        / run_day
        / "_submissions"
        / safe_layout_name(experiment, label="experiment")
        / source.name
    )


def plan_date_reorganization(root: str | Path) -> dict[str, Any]:
    """Build a read-only move plan for indexed, terminal-state runs."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {root_path}")
    entries: list[dict[str, Any]] = []
    for record in list_run_records(root_path):
        job_id = str(record.get("job_id") or "unknown")
        experiment = str(record.get("experiment") or "unknown")
        state = str(record.get("state") or "unknown")
        source_value = record.get("run_directory")
        source = (
            Path(str(source_value)).expanduser().resolve()
            if source_value
            else None
        )
        timestamp = record.get("created_at") or record.get("updated_at")
        try:
            run_day = (
                utc_run_day(timestamp)
                if timestamp is not None
                else "unknown-date"
            )
        except ValueError:
            run_day = "unknown-date"
        target = (
            dated_run_directory(
                root_path,
                run_day=run_day,
                experiment=experiment,
                job_id=job_id,
            )
            if run_day != "unknown-date"
            else None
        )
        submission_value = record.get("submission_directory")
        submission = (
            Path(str(submission_value)).expanduser().resolve()
            if submission_value
            else None
        )
        submission_target = (
            _submission_target(
                root_path,
                run_day=run_day,
                experiment=experiment,
                source=submission,
            )
            if submission is not None
            and submission.is_dir()
            and _within(submission, root_path)
            and run_day != "unknown-date"
            else None
        )
        reason = None
        ready = True
        already_organized = bool(source and target and source == target)
        if state not in _MOVABLE_STATES:
            ready = False
            reason = f"state {state!r} is not terminal"
        elif source is None or not source.is_dir():
            ready = False
            reason = "indexed run directory is missing"
        elif source.is_symlink():
            ready = False
            reason = "indexed run directory is a symbolic link"
        elif not _within(source, root_path):
            ready = False
            reason = "indexed run directory is outside the run root"
        elif target is None:
            ready = False
            reason = "run date cannot be determined"
        elif already_organized:
            ready = False
            reason = "already organized"
        elif target.exists() or target.is_symlink():
            ready = False
            reason = "target already exists"
        elif submission_target is not None and (
            submission_target.exists() or submission_target.is_symlink()
        ):
            ready = False
            reason = "submission target already exists"
        entries.append(
            {
                "job_id": job_id,
                "experiment": experiment,
                "state": state,
                "run_day": run_day,
                "source": str(source) if source is not None else None,
                "target": str(target) if target is not None else None,
                "submission_source": (
                    str(submission) if submission is not None else None
                ),
                "submission_target": (
                    str(submission_target)
                    if submission_target is not None
                    else None
                ),
                "ready": ready,
                "reason": reason,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root_path),
        "ready_count": sum(bool(entry["ready"]) for entry in entries),
        "skipped_count": sum(not bool(entry["ready"]) for entry in entries),
        "entries": entries,
    }


def _prune_empty_parents(start: Path, root: Path) -> None:
    current = start
    while current != root and _within(current, root):
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _update_submission_receipt(submission: Path, target: Path) -> bytes | None:
    receipt_path = submission / "submission.json"
    if not receipt_path.is_file():
        return None
    original = receipt_path.read_bytes()
    payload = json.loads(original)
    if not isinstance(payload, dict):
        raise ValueError(f"Submission receipt is not a mapping: {receipt_path}")
    payload["run_root"] = str(target.parent)
    payload["expected_run_directory"] = str(target)
    _atomic_json(receipt_path, payload)
    return original


def apply_date_reorganization(root: str | Path) -> dict[str, Any]:
    """Apply one preflighted date-first migration and record every move."""

    plan = plan_date_reorganization(root)
    root_path = Path(plan["root"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    manifest_path = root_path / "_migrations" / stamp / "reorganization.json"
    completed: list[dict[str, Any]] = []
    plan["mode"] = "apply"
    plan["manifest"] = str(manifest_path)
    plan["completed"] = completed
    _atomic_json(manifest_path, plan)
    for entry in plan["entries"]:
        if not entry["ready"]:
            continue
        source = Path(entry["source"])
        target = Path(entry["target"])
        submission_source = (
            Path(entry["submission_source"])
            if entry["submission_source"]
            else None
        )
        submission_target = (
            Path(entry["submission_target"])
            if entry["submission_target"]
            else None
        )
        moved_run = False
        moved_submission = False
        receipt_original: bytes | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            moved_run = True
            active_submission = (
                submission_source
                if submission_source is not None and submission_source.is_dir()
                else None
            )
            if submission_source is not None and submission_target is not None:
                submission_target.parent.mkdir(parents=True, exist_ok=True)
                submission_source.rename(submission_target)
                moved_submission = True
                active_submission = submission_target
            if active_submission is not None and active_submission.is_dir():
                receipt_original = _update_submission_receipt(
                    active_submission,
                    target,
                )
            relocate_run_record(
                root_path,
                entry["job_id"],
                run_directory=target,
                submission_directory=active_submission,
            )
        except Exception:
            if receipt_original is not None:
                receipt_path = (
                    submission_target if moved_submission else submission_source
                ) / "submission.json"
                receipt_path.write_bytes(receipt_original)
            if moved_submission and submission_target is not None:
                submission_target.rename(submission_source)
            if moved_run:
                target.rename(source)
            raise
        completed.append(
            {
                "job_id": entry["job_id"],
                "source": str(source),
                "target": str(target),
                "submission_target": (
                    str(submission_target) if moved_submission else None
                ),
            }
        )
        _atomic_json(manifest_path, plan)
        _prune_empty_parents(source.parent, root_path)
        if moved_submission and submission_source is not None:
            _prune_empty_parents(submission_source.parent, root_path)
    plan["completed_count"] = len(completed)
    plan["completed_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest_path, plan)
    return plan
