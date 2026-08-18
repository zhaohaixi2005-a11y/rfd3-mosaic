"""Execution backends for frozen RFD3-Mosaic submission plans."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SubmissionResult:
    """Backend-neutral identity returned after accepting one frozen plan."""

    executor: str
    job_id: str
    message: str


class Executor(Protocol):
    """Small boundary used by the CLI instead of scheduler-specific code."""

    id: str

    def submit(self, script: Path) -> SubmissionResult: ...


class SlurmExecutor:
    """Submit a rendered script through Slurm's stable parsable interface."""

    id = "slurm"

    def submit(self, script: Path) -> SubmissionResult:
        try:
            completed = subprocess.run(
                ["sbatch", "--parsable", str(script)],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "Slurm executor cannot find the sbatch executable"
            ) from error
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or str(error)).strip()
            raise RuntimeError(f"Slurm submission failed: {detail}") from error
        output = completed.stdout.strip()
        # --parsable may return "jobid" or "jobid;cluster".
        job_id = output.split(";", maxsplit=1)[0].strip()
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
            raise RuntimeError(
                "sbatch --parsable returned an invalid JobID: " f"{output!r}"
            )
        return SubmissionResult(
            executor=self.id,
            job_id=job_id,
            message=f"Submitted batch job {job_id}",
        )


class LocalExecutor:
    """Run one frozen plan synchronously on the current workstation."""

    id = "local"

    def submit(self, script: Path) -> SubmissionResult:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        job_id = f"local-{timestamp}-{os.getpid()}"
        environment = dict(os.environ)
        environment["RFD3_MOSAIC_JOB_ID"] = job_id
        try:
            completed = subprocess.run(
                ["bash", str(script)],
                check=True,
                env=environment,
            )
        except FileNotFoundError as error:
            raise RuntimeError("Local executor cannot find bash") from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Local execution failed with exit code {error.returncode}"
            ) from error
        if completed.returncode != 0:
            raise RuntimeError(
                f"Local execution failed with exit code {completed.returncode}"
            )
        return SubmissionResult(
            executor=self.id,
            job_id=job_id,
            message=f"Completed local run {job_id}",
        )


def executor_for_id(executor_id: str) -> Executor:
    """Resolve an executor explicitly and fail closed on unknown values."""

    if executor_id == "slurm":
        return SlurmExecutor()
    if executor_id == "local":
        return LocalExecutor()
    raise ValueError(f"Unknown executor: {executor_id!r}")
