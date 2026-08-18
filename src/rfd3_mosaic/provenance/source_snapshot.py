"""Immutable source snapshots for queued Mosaic executions."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Iterable

from rfd3_mosaic.provenance.software import file_identity, sha256_file

SOURCE_SNAPSHOT_SCHEMA_VERSION = 1
SOURCE_SNAPSHOT_MANIFEST = "source_snapshot_manifest.json"
DEFAULT_SOURCE_ROOTS = (
    "pyproject.toml",
    "src/foundry",
    "src/foundry_cli",
    "src/rfd3_mosaic",
    "models/rfd3/src/rfd3",
    "models/rfd3/configs",
    "configs/rfd3_mosaic",
)


def _git_files(repository: Path, roots: tuple[str, ...]) -> list[str] | None:
    command = [
        "git",
        "-C",
        str(repository),
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        *roots,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return sorted(set(completed.stdout.splitlines()))


def _filesystem_files(
    repository: Path,
    roots: tuple[str, ...],
) -> list[str]:
    files: set[str] = set()
    for root_text in roots:
        root = repository / root_text
        if root.is_file():
            files.add(root.relative_to(repository).as_posix())
        elif root.is_dir():
            files.update(
                path.relative_to(repository).as_posix()
                for path in root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    return sorted(files)


def source_snapshot_files(
    repository: Path,
    *,
    roots: Iterable[str] = DEFAULT_SOURCE_ROOTS,
) -> tuple[str, ...]:
    """Return the versioned and untracked source files used at runtime."""

    resolved = repository.resolve()
    root_tuple = tuple(roots)
    filesystem_files = _filesystem_files(resolved, root_tuple)
    git_files = _git_files(resolved, root_tuple)
    files = sorted(
        set(filesystem_files)
        | (set(git_files) if git_files is not None else set())
    )
    existing = [
        relative
        for relative in files
        if (resolved / relative).is_file()
        and not relative.endswith((".pyc", ".pyo"))
    ]
    if not existing:
        raise RuntimeError("Source snapshot contains no runtime files")
    return tuple(existing)


def create_source_snapshot(
    repository: Path,
    archive: Path,
    *,
    roots: Iterable[str] = DEFAULT_SOURCE_ROOTS,
) -> dict[str, Any]:
    """Create a compressed runtime source tree and return its identity."""

    resolved_repository = repository.resolve()
    resolved_archive = archive.resolve()
    root_tuple = tuple(roots)
    files = source_snapshot_files(
        resolved_repository,
        roots=root_tuple,
    )
    resolved_archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(resolved_archive, mode="w:gz") as handle:
        records = []
        for relative in files:
            path = resolved_repository / relative
            before = path.stat()
            payload = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise RuntimeError(
                    f"Source file changed while snapshotting: {relative}"
                )
            records.append(
                {
                    "path": relative,
                    "sha256": sha256_bytes(payload),
                    "size_bytes": len(payload),
                }
            )
            info = tarfile.TarInfo(relative)
            info.size = len(payload)
            info.mode = after.st_mode & 0o777
            handle.addfile(info, io.BytesIO(payload))
        manifest = {
            "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
            "source_roots": list(root_tuple),
            "files": records,
        }
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        info = tarfile.TarInfo(SOURCE_SNAPSHOT_MANIFEST)
        info.size = len(manifest_bytes)
        info.mode = 0o644
        handle.addfile(info, io.BytesIO(manifest_bytes))
    archive_identity = file_identity(
        resolved_archive,
        role="Mosaic source snapshot",
    )
    return {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "archive": archive_identity,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "file_count": len(files),
        "source_roots": list(root_tuple),
    }


def sha256_bytes(payload: bytes) -> str:
    """Hash an in-memory manifest without creating a second sidecar file."""

    return hashlib.sha256(payload).hexdigest()


def verify_source_snapshot_tree(
    source_root: Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify every extracted source file before the experiment runs."""

    root = source_root.resolve()
    manifest_path = root / SOURCE_SNAPSHOT_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Source snapshot manifest is missing: {manifest_path}"
        )
    manifest_bytes = manifest_path.read_bytes()
    observed_manifest_sha256 = sha256_bytes(manifest_bytes)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            "Source snapshot manifest SHA256 changed before runtime"
        )
    manifest = json.loads(manifest_bytes)
    if int(manifest.get("schema_version", 0)) != 1:
        raise RuntimeError("Unsupported source snapshot manifest schema")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Source snapshot manifest contains no files")
    for record in records:
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(
                f"Unsafe source snapshot manifest path: {relative}"
            )
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"Snapshot source file is missing: {relative}")
        if path.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError(
                f"Snapshot source file size changed: {relative}"
            )
        if sha256_file(path) != str(record["sha256"]):
            raise RuntimeError(
                f"Snapshot source file SHA256 changed: {relative}"
            )
    return manifest


__all__ = [
    "DEFAULT_SOURCE_ROOTS",
    "SOURCE_SNAPSHOT_MANIFEST",
    "SOURCE_SNAPSHOT_SCHEMA_VERSION",
    "create_source_snapshot",
    "source_snapshot_files",
    "verify_source_snapshot_tree",
]
