"""Read-only installation diagnostics for RFD3-Mosaic."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from rfd3_mosaic.installation import (
    bundled_resource_path,
    distribution_version,
    source_repository_root,
)


def installation_diagnostics(
    *,
    profile: str = "local",
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Return a side-effect-free installation and runtime readiness report."""

    checks: list[dict[str, Any]] = []

    def record(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"id": check_id, "passed": passed, "detail": detail})

    record(
        "python",
        sys.version_info >= (3, 12),
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    try:
        import torch

        cuda = torch.cuda.is_available()
        detail = f"torch={torch.__version__}; cuda={cuda}"
        if cuda:
            detail += f"; gpu={torch.cuda.get_device_name(0)}"
        record("torch", True, detail)
    except Exception as error:  # pragma: no cover - environment dependent
        record("torch", False, repr(error))
    try:
        import rfd3

        record("rfd3", True, str(Path(rfd3.__file__).resolve()))
    except Exception as error:
        record("rfd3", False, repr(error))

    try:
        compatibility = bundled_resource_path(
            "configs/rfd3_mosaic/compatibility/foundry.yaml"
        )
        record("compatibility_manifest", True, str(compatibility))
    except Exception as error:
        record("compatibility_manifest", False, str(error))

    requested = Path(profile).expanduser()
    try:
        if requested.suffix in {".yaml", ".yml"} or requested.parent != Path("."):
            profile_path = requested.resolve()
        else:
            profile_path = bundled_resource_path(
                Path("configs/rfd3_mosaic/execution") / f"{profile}.yaml"
            )
        payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        executor = str(payload.get("executor", "slurm"))
        record("execution_profile", True, f"{profile_path}; executor={executor}")
        if executor == "slurm":
            record(
                "executor",
                shutil.which("sbatch") is not None,
                shutil.which("sbatch") or "sbatch not found",
            )
        else:
            record("executor", executor == "local", executor)
        declared_checkpoint = Path(
            checkpoint or payload.get("checkpoint", "")
        ).expanduser()
    except Exception as error:
        record("execution_profile", False, str(error))
        declared_checkpoint = Path(checkpoint).expanduser() if checkpoint else None

    if declared_checkpoint is not None:
        record(
            "checkpoint",
            declared_checkpoint.is_file() and declared_checkpoint.stat().st_size > 0,
            str(declared_checkpoint.resolve()),
        )

    repository = source_repository_root()
    return {
        "product": "RFD3-Mosaic",
        "version": distribution_version(),
        "installation_mode": (
            "source_checkout" if repository is not None else "installed_distribution"
        ),
        "source_repository": str(repository) if repository is not None else None,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
