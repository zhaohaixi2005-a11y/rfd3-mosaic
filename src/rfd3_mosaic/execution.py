"""Execution backends for frozen RFD3-Mosaic submission plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
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
                "sbatch --parsable returned an invalid JobID: "
                f"{output!r}"
            )
        return SubmissionResult(
            executor=self.id,
            job_id=job_id,
            message=f"Submitted batch job {job_id}",
        )


def executor_for_id(executor_id: str) -> Executor:
    """Resolve an executor explicitly and fail closed on unknown values."""

    if executor_id == "slurm":
        return SlurmExecutor()
    raise ValueError(f"Unknown executor: {executor_id!r}")
