"""Non-destructive, version-aware views over a Mosaic run root."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rfd3_mosaic.run_index import list_run_records


SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _safe_name(value: object) -> str:
    cleaned = _SAFE_NAME.sub("-", str(value)).strip("-.")
    return cleaned[:120] or "unknown"


def _run_label(
    job_id: str,
    experiment: str,
    state: str,
    revision_label: str,
) -> str:
    """Build an informative name that stays below filesystem limits."""

    return (
        f"{_safe_name(job_id)}__{_safe_name(experiment)[:80]}"
        f"__{_safe_name(state)[:20]}__{revision_label}"
    )


def _calendar_day(value: object) -> str:
    """Return one stable UTC calendar directory for an indexed timestamp."""

    text = str(value or "").strip()
    if not text:
        return "unknown-date"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return _safe_name(text[:10])
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date().isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _source_revision(run_directory: Path) -> str | None:
    for name in ("runtime_provenance.json", "provenance.json"):
        payload = _read_json(run_directory / name)
        repository = payload.get("repository")
        if isinstance(repository, dict):
            value = repository.get("commit") or repository.get("revision")
            if value:
                return str(value)
        for key in ("source_commit", "git_commit", "revision"):
            value = payload.get(key)
            if value:
                return str(value)
    return None


def _structure_paths(run_directory: Path) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for pattern in (
        "*model_*.cif",
        "*model_*.cif.gz",
        "*model_*.pdb",
        "*model_*.pdb.gz",
    ):
        paths.update(path.resolve() for path in run_directory.glob(pattern))
    return tuple(sorted(path for path in paths if path.is_file()))


def _symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def _markdown(entries: list[dict[str, Any]], retained: set[str]) -> str:
    lines = [
        "# RFD3-Mosaic run catalog",
        "",
        "This is a generated, non-destructive view. Its links point to the "
        "original indexed run directories; no scientific output was moved.",
        "",
        "| Newest | Job | State | Experiment | Source | Structures | Retained |",
        "| ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    for index, entry in enumerate(entries, start=1):
        revision = str(entry.get("source_revision") or "unknown")[:12]
        job_id = str(entry["job_id"])
        lines.append(
            "| "
            f"{index} | `{job_id}` | {entry['state']} | "
            f"{entry['experiment']} | `{revision}` | "
            f"{len(entry['structures'])} | "
            f"{'yes' if job_id in retained else ''} |"
        )
    lines.extend(
        [
            "",
            "Views:",
            "",
            "- `by-date/YYYY-MM-DD/`: primary daily view; one parent directory "
            "per UTC calendar day;",
            "- `latest/`: newest indexed runs in chronological order;",
            "- `by-state/`: completed, failed, running and submitted runs;",
            "- `by-version/`: runs grouped by frozen source commit;",
            "- `structures/`: direct links to every generated structure;",
            "- `retained/`: explicitly protected results supplied to `--retain`.",
            "",
        ]
    )
    return "\n".join(lines)


def _daily_markdown(day: str, entries: list[dict[str, Any]]) -> str:
    lines = [
        f"# RFD3-Mosaic runs for {day}",
        "",
        "This directory contains symbolic links to the original run directories.",
        "No result or per-run `software/` source snapshot was copied or moved.",
        "",
        "| Job | State | Experiment | Source | Structures |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for entry in entries:
        revision = str(entry.get("source_revision") or "unknown")[:12]
        lines.append(
            f"| `{entry['job_id']}` | {entry['state']} | "
            f"{entry['experiment']} | `{revision}` | "
            f"{len(entry['structures'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_run_catalog(
    root: str | Path,
    *,
    output_directory: str | Path | None = None,
    retained_job_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Write one immutable catalog snapshot and update the ``CURRENT`` link."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {root_path}")

    catalog_root = root_path / "_catalog"
    automatic_output = output_directory is None
    stamp = _stamp()
    output = (
        catalog_root / "snapshots" / stamp[:8] / stamp
        if automatic_output
        else Path(output_directory).expanduser().resolve()
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Catalog output already exists: {output}")
    output.mkdir(parents=True)

    previous = _read_json(catalog_root / "CURRENT" / "catalog.json")
    previous_retained = previous.get("retained_job_ids") or []
    if not isinstance(previous_retained, list):
        previous_retained = []
    retained = {
        *(str(value) for value in previous_retained),
        *(str(value) for value in retained_job_ids),
    }
    entries: list[dict[str, Any]] = []
    entries_by_day: dict[str, list[dict[str, Any]]] = {}
    for position, record in enumerate(list_run_records(root_path), start=1):
        job_id = str(record.get("job_id") or "unknown")
        state = str(record.get("state") or "unknown")
        experiment = str(record.get("experiment") or "unknown")
        run_value = record.get("run_directory")
        run_directory = (
            Path(str(run_value)).expanduser().resolve() if run_value else None
        )
        present = bool(run_directory and run_directory.is_dir())
        revision = _source_revision(run_directory) if present else None
        structures = _structure_paths(run_directory) if present else ()
        entry = {
            "job_id": job_id,
            "state": state,
            "experiment": experiment,
            "campaign": str(record.get("campaign") or "unknown"),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "source_revision": revision,
            "run_directory": str(run_directory) if run_directory else None,
            "run_directory_present": present,
            "structures": [str(path) for path in structures],
        }
        entries.append(entry)
        day = _calendar_day(record.get("created_at") or record.get("updated_at"))
        entries_by_day.setdefault(day, []).append(entry)
        if not present or run_directory is None:
            continue

        revision_label = _safe_name((revision or "unknown")[:12])
        label = _run_label(job_id, experiment, state, revision_label)
        _symlink(run_directory, output / "by-date" / day / label)
        _symlink(run_directory, output / "by-state" / _safe_name(state) / label)
        _symlink(
            run_directory,
            output / "by-version" / revision_label / label,
        )
        _symlink(run_directory, output / "latest" / f"{position:04d}__{label}")
        if job_id in retained:
            _symlink(run_directory, output / "retained" / label)
        for structure_index, structure in enumerate(structures, start=1):
            structure_label = (
                f"{job_id}__{structure_index:03d}__{_safe_name(structure.name)}"
            )
            _symlink(structure, output / "structures" / structure_label)

    for day, daily_entries in entries_by_day.items():
        day_directory = output / "by-date" / day
        day_directory.mkdir(parents=True, exist_ok=True)
        (day_directory / "RUNS.md").write_text(
            _daily_markdown(day, daily_entries),
            encoding="utf-8",
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(root_path),
        "catalog_directory": str(output),
        "run_count": len(entries),
        "structure_count": sum(len(entry["structures"]) for entry in entries),
        "retained_job_ids": sorted(retained),
        "runs": entries,
    }
    (output / "catalog.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "CATALOG.md").write_text(
        _markdown(entries, retained),
        encoding="utf-8",
    )

    if automatic_output:
        catalog_root.mkdir(parents=True, exist_ok=True)
        temporary = catalog_root / f".CURRENT-{os.getpid()}"
        temporary.symlink_to(
            os.path.relpath(output, catalog_root),
            target_is_directory=True,
        )
        temporary.replace(catalog_root / "CURRENT")
    return payload
