"""Persistent, append-safe job index for RFD3-Mosaic run roots."""

from __future__ import annotations

import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1
INDEX_DIRECTORY = ".rfd3-mosaic"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_job_id(job_id: str) -> str:
    value = str(job_id)
    if not RUN_ID_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Invalid execution run ID: {value!r}")
    return value


def valid_run_id(value: str) -> bool:
    """Return whether an executor identity is safe for paths and indexing."""

    return bool(RUN_ID_PATTERN.fullmatch(str(value))) and value not in {".", ".."}


def _index_path(root: Path, job_id: str) -> Path:
    return root / INDEX_DIRECTORY / "jobs" / f"{_validate_job_id(job_id)}.json"


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
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


def read_run_record(root: str | Path, job_id: str) -> dict[str, Any] | None:
    path = _index_path(Path(root).expanduser().resolve(), job_id)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a run-index mapping in {path}")
    if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported run-index schema in {path}")
    return payload


def record_submission(
    *,
    root: str | Path,
    job_id: str,
    experiment: str,
    campaign: str,
    run_directory: str | Path,
    submission_directory: str | Path,
    executor: str,
) -> Path:
    """Create the durable identity record immediately after submission."""

    root_path = Path(root).expanduser().resolve()
    path = _index_path(root_path, job_id)
    if path.exists():
        existing = read_run_record(root_path, job_id)
        if existing is None or existing.get("executor") != "unknown":
            raise FileExistsError(f"Run index already contains run ID {job_id}: {path}")
        # A synchronous local worker finishes before submit() returns to the
        # CLI. It therefore creates the lifecycle record first. Attach the
        # submission identity without erasing its completed/failed state.
        existing.update(
            {
                "executor": executor,
                "submission_directory": str(
                    Path(submission_directory).expanduser().resolve()
                ),
                "updated_at": _now(),
            }
        )
        _atomic_write(path, existing)
        return path
    payload = {
        "schema_version": SCHEMA_VERSION,
        "job_id": _validate_job_id(job_id),
        "experiment": experiment,
        "campaign": campaign,
        "executor": executor,
        "state": "submitted",
        "created_at": _now(),
        "updated_at": _now(),
        "run_directory": str(Path(run_directory).expanduser().resolve()),
        "submission_directory": str(Path(submission_directory).expanduser().resolve()),
    }
    _atomic_write(path, payload)
    return path


def update_run_state(
    *,
    root: str | Path,
    job_id: str,
    state: str,
    experiment: str,
    campaign: str,
    run_directory: str | Path,
    error: str | None = None,
    observed_at: str | None = None,
) -> Path:
    """Upsert lifecycle state from the allocated worker."""

    if state not in {"running", "completed", "failed"}:
        raise ValueError(f"Invalid indexed run state: {state!r}")
    root_path = Path(root).expanduser().resolve()
    path = _index_path(root_path, job_id)
    existing = read_run_record(root_path, job_id) or {
        "schema_version": SCHEMA_VERSION,
        "job_id": _validate_job_id(job_id),
        "experiment": experiment,
        "campaign": campaign,
        "executor": "unknown",
        "created_at": observed_at or _now(),
        "submission_directory": None,
    }
    existing.update(
        {
            "state": state,
            "updated_at": observed_at or _now(),
            "run_directory": str(Path(run_directory).resolve()),
            "error": error,
        }
    )
    _atomic_write(path, existing)
    return path


def list_run_records(root: str | Path) -> list[dict[str, Any]]:
    """Return newest indexed runs first, skipping no malformed records."""

    root_path = Path(root).expanduser().resolve()
    directory = root_path / INDEX_DIRECTORY / "jobs"
    if not directory.is_dir():
        return []
    records = []
    for path in directory.glob("*.json"):
        payload = read_run_record(root_path, path.stem)
        if payload is not None:
            records.append(payload)
    return sorted(
        records,
        key=lambda item: str(item.get("created_at") or ""),
        reverse=True,
    )


def relocate_run_record(
    root: str | Path,
    job_id: str,
    *,
    run_directory: str | Path,
    submission_directory: str | Path | None = None,
) -> Path:
    """Atomically record a verified physical run-directory relocation."""

    root_path = Path(root).expanduser().resolve()
    path = _index_path(root_path, job_id)
    payload = read_run_record(root_path, job_id)
    if payload is None:
        raise FileNotFoundError(f"Run index does not contain run ID {job_id}")
    old_run = payload.get("run_directory")
    old_submission = payload.get("submission_directory")
    new_run = Path(run_directory).expanduser().resolve()
    if not new_run.is_dir():
        raise FileNotFoundError(f"Relocated run directory does not exist: {new_run}")
    new_submission = (
        Path(submission_directory).expanduser().resolve()
        if submission_directory is not None
        else None
    )
    history = payload.get("relocation_history")
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "moved_at": _now(),
            "old_run_directory": old_run,
            "new_run_directory": str(new_run),
            "old_submission_directory": old_submission,
            "new_submission_directory": (
                str(new_submission) if new_submission is not None else None
            ),
        }
    )
    payload.update(
        {
            "run_directory": str(new_run),
            "submission_directory": (
                str(new_submission) if new_submission is not None else None
            ),
            "relocation_history": history,
            "updated_at": _now(),
        }
    )
    _atomic_write(path, payload)
    return path


def rebuild_run_index(root: str | Path) -> dict[str, Any]:
    """Import historical worker summaries into the persistent job index."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {root_path}")
    indexed = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    for summary_path in root_path.rglob("experiment_summary.json"):
        run_directory = summary_path.parent.resolve()
        if INDEX_DIRECTORY in run_directory.parts or "software" in run_directory.parts:
            skipped += 1
            continue
        job_id = run_directory.name
        if not valid_run_id(job_id):
            skipped += 1
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                raise ValueError("worker summary is not a JSON mapping")
            state = str(summary.get("status") or "")
            if state not in {"running", "completed", "failed"}:
                raise ValueError(f"unsupported worker status {state!r}")
            resolved_path = run_directory / "resolved_config.yaml"
            resolved: dict[str, Any] = {}
            if resolved_path.is_file():
                try:
                    loaded = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        resolved = loaded
                except (OSError, ValueError, yaml.YAMLError):
                    resolved = {}
            relative_parts = run_directory.relative_to(root_path).parts
            inferred_campaign = (
                relative_parts[-3] if len(relative_parts) >= 3 else "legacy"
            )
            output = resolved.get("output") or {}
            campaign = str(output.get("campaign") or inferred_campaign)
            experiment = str(
                summary.get("experiment")
                or resolved.get("name")
                or run_directory.parent.name
            )
            update_run_state(
                root=root_path,
                job_id=job_id,
                state=state,
                experiment=experiment,
                campaign=campaign,
                run_directory=run_directory,
                error=(
                    str(summary["error"]) if summary.get("error") is not None else None
                ),
                observed_at=datetime.fromtimestamp(
                    summary_path.stat().st_mtime,
                    timezone.utc,
                ).isoformat(),
            )
            indexed += 1
        except (
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            yaml.YAMLError,
        ) as error:
            failures.append({"path": str(summary_path.resolve()), "error": str(error)})
    return {
        "schema_version": SCHEMA_VERSION,
        "root": str(root_path),
        "indexed": indexed,
        "skipped": skipped,
        "failed": len(failures),
        "failures": failures,
    }
