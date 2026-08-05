"""Reproducible software, repository, and runtime provenance."""

from __future__ import annotations

import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import yaml


PROVENANCE_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(
    repository: Path,
    *arguments: str,
    binary: bool = False,
) -> str | bytes | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=not binary,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if binary:
        return completed.stdout
    return completed.stdout.strip()


def collect_repository_provenance(repository: Path) -> dict[str, Any]:
    """Describe the exact Git source tree used to render or run a design."""

    root = repository.resolve()
    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    origin = _git(root, "remote", "get-url", "origin")
    tracked_status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    )
    untracked = _git(root, "ls-files", "--others", "--exclude-standard")
    diff = _git(root, "diff", "--binary", "HEAD", "--", binary=True)
    diff_sha256 = (
        hashlib.sha256(diff).hexdigest()
        if isinstance(diff, bytes) and diff
        else None
    )
    return {
        "repository_root": str(root),
        "origin": origin,
        "commit": commit,
        "branch": branch,
        "tracked_dirty": bool(tracked_status),
        "tracked_status": tracked_status.splitlines() if tracked_status else [],
        "untracked_files": untracked.splitlines() if untracked else [],
        "working_tree_diff_sha256": diff_sha256,
    }


def load_compatibility_manifest(path: Path) -> dict[str, Any]:
    """Load and fingerprint the declared Foundry compatibility contract."""

    resolved = path.resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {resolved}")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(
            f"Unsupported compatibility schema in {resolved}: "
            f"{payload.get('schema_version')!r}"
        )
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "manifest": payload,
    }


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def collect_runtime_provenance(
    repository: Path,
    *,
    checkpoint: Path | None = None,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Capture runtime identity without hashing a large checkpoint implicitly."""

    checkpoint_record: dict[str, Any] | None = None
    if checkpoint is not None:
        resolved = checkpoint.expanduser().resolve()
        checkpoint_record = {
            "path": str(resolved),
            "exists": resolved.is_file(),
            "declared_sha256": checkpoint_sha256,
        }
        if resolved.is_file():
            stat = resolved.stat()
            checkpoint_record.update(
                {
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "repository": collect_repository_provenance(repository),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            "rc-foundry": _package_version("rc-foundry"),
            "torch": _package_version("torch"),
            "pydantic": _package_version("pydantic"),
            "hydra-core": _package_version("hydra-core"),
        },
        "scheduler": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_cluster_name": os.environ.get("SLURM_CLUSTER_NAME"),
        },
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "checkpoint": checkpoint_record,
    }
