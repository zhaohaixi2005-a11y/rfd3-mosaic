#!/usr/bin/env python3
"""Freeze and optionally submit the Mosaic LHD101 C3 design campaign."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml


DEFAULT_RUN_ROOT = Path(
    "/dss/dssfs02/lwp-dss-0001/pn57ki/"
    "pn57ki-dss-0000/haixi/runs/rfd3-mosaic"
)
DEFAULT_TEMPLATE = Path(
    "experiments/lrz_mosaic_lhd101_c3_guided_50step_template.yaml"
)
DEFAULT_PROFILE = Path("configs/rfd3_mosaic/sites/lrz/any_gpu.yaml")
JOB_ID_PATTERN = re.compile(r"Submitted batch job ([0-9]+)")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one YAML mapping in {path}")
    return payload


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="", flush=True)
    return completed


def _revision(project: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create recoverable Mosaic/RFD3 shards for the LHD101 C3 "
            "interface-seed experiment."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("pilot", "full"),
        default="pilot",
        help="pilot emits one design; full defaults to 1000 designs.",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--total-designs", type=int)
    parser.add_argument(
        "--designs-per-job",
        type=int,
        default=1,
        help=(
            "Independent outputs packed into one GPU job. Current Mosaic "
            "materializes a distinct feasible initial pose per output when "
            "the pose declaration is stochastic, while loading RFD3 once."
        ),
    )
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument(
        "--pose-seeds",
        type=int,
        nargs="+",
        help=(
            "Explicit initial-pose seeds for a multi-pose pilot. Each seed "
            "is frozen into one independent one-design GPU job; diffusion "
            "seeds still start at --seed-start."
        ),
    )
    parser.add_argument(
        "--diffusion-seeds-per-pose",
        type=int,
        default=1,
        help=(
            "Number of independent one-design diffusion jobs emitted for "
            "each explicit pilot pose seed."
        ),
    )
    arguments = parser.parse_args()

    project = Path.cwd().resolve()
    template_path = (project / arguments.template).resolve()
    profile_path = (project / arguments.profile).resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"Template does not exist: {template_path}")
    if not profile_path.is_file():
        raise FileNotFoundError(f"Profile does not exist: {profile_path}")
    if arguments.seed_start < 0:
        raise ValueError("--seed-start cannot be negative")
    pose_seeds = list(arguments.pose_seeds or [])
    if any(seed < 0 for seed in pose_seeds):
        raise ValueError("--pose-seeds cannot contain negative values")
    if len(set(pose_seeds)) != len(pose_seeds):
        raise ValueError("--pose-seeds must be unique")
    if arguments.diffusion_seeds_per_pose < 1:
        raise ValueError("--diffusion-seeds-per-pose must be positive")
    if pose_seeds and arguments.mode != "pilot":
        raise ValueError("--pose-seeds is reserved for independent pilot jobs")
    if (
        arguments.diffusion_seeds_per_pose != 1
        and not pose_seeds
    ):
        raise ValueError(
            "--diffusion-seeds-per-pose requires explicit --pose-seeds"
        )
    pilot_matrix_size = (
        len(pose_seeds) * arguments.diffusion_seeds_per_pose
        if pose_seeds
        else 0
    )
    if (
        pose_seeds
        and arguments.total_designs is not None
        and arguments.total_designs != pilot_matrix_size
    ):
        raise ValueError(
            "--total-designs must equal pose seed count times "
            "--diffusion-seeds-per-pose"
        )

    default_total = (
        pilot_matrix_size
        if pose_seeds
        else (1 if arguments.mode == "pilot" else 1000)
    )
    total = arguments.total_designs or default_total
    per_job = 1 if arguments.mode == "pilot" else arguments.designs_per_job
    if total < 1 or total > 10000:
        raise ValueError("total designs must be between 1 and 10000")
    if per_job < 1 or per_job > 10000:
        raise ValueError("designs per job must be between 1 and 10000")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign_tag = f"mosaic-lhd101-c3-{arguments.mode}-{stamp}"
    output = (
        arguments.output_dir.expanduser().resolve()
        if arguments.output_dir is not None
        else arguments.run_root.expanduser().resolve()
        / "_campaigns"
        / "mosaic-lhd101-c3-guided"
        / stamp
    )
    output.mkdir(parents=True, exist_ok=False)
    designs_dir = output / "designs"
    designs_dir.mkdir()

    source = _load_yaml(template_path)
    source_input = Path(str(source["input"])).expanduser()
    if not source_input.is_absolute():
        source_input = (template_path.parent / source_input).resolve()
    if not source_input.is_file():
        raise FileNotFoundError(f"Input structure does not exist: {source_input}")

    shard_count = math.ceil(total / per_job)
    records: list[dict[str, Any]] = []
    pilot_pose_schedule = [
        pose_seed
        for pose_seed in pose_seeds
        for _ in range(arguments.diffusion_seeds_per_pose)
    ]
    remaining = total
    for shard_index in range(shard_count):
        shard_designs = min(per_job, remaining)
        remaining -= shard_designs
        diffusion_seed = arguments.seed_start + shard_index
        pose_seed = (
            pilot_pose_schedule[shard_index]
            if pilot_pose_schedule
            else diffusion_seed
        )
        payload = deepcopy(source)
        payload["name"] = (
            f"lhd101-c3-guided-{shard_index:04d}"
            f"-p{pose_seed}-s{diffusion_seed}"
        )
        payload["input"] = str(source_input)
        payload["sampling"] = dict(payload["sampling"])
        payload["sampling"]["designs"] = shard_designs
        payload["sampling"]["seed"] = diffusion_seed
        payload["sampling"]["initial_pose"] = dict(
            payload["sampling"]["initial_pose"]
        )
        payload["sampling"]["initial_pose"]["seed"] = pose_seed
        payload["output"] = dict(payload["output"])
        payload["output"]["root"] = str(
            arguments.run_root.expanduser().resolve()
        )
        payload["output"]["campaign"] = campaign_tag

        frozen = designs_dir / f"shard_{shard_index:04d}.yaml"
        frozen.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        record: dict[str, Any] = {
            "shard_index": shard_index,
            "design": str(frozen),
            # Keep seed for manifest compatibility; it is the RFD3 diffusion
            # seed. Pose and diffusion randomness are also recorded
            # separately so a multi-pose pilot is unambiguous.
            "seed": diffusion_seed,
            "diffusion_seed": diffusion_seed,
            "pose_seed": pose_seed,
            "requested_designs": shard_designs,
            "submitted": False,
            "job_id": None,
            "returncode": None,
        }
        if arguments.submit:
            result = _run(
                [
                    sys.executable,
                    "-m",
                    "rfd3_mosaic.cli",
                    "submit",
                    str(frozen),
                    "--profile",
                    str(profile_path),
                ],
                cwd=project,
            )
            match = JOB_ID_PATTERN.search(result.stdout)
            record["returncode"] = result.returncode
            record["job_id"] = match.group(1) if match else None
            record["submitted"] = result.returncode == 0 and match is not None
        records.append(record)

    compiled_pose_count = len(pose_seeds) if pose_seeds else total
    pose_semantics = (
        "explicit_pose_matrix; repeated diffusion streams intentionally "
        "share each listed pose"
        if pose_seeds
        else (
            "one_independently_seeded_feasible_pose_per_design; one RFD3 "
            "model load per shard"
        )
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": arguments.mode,
        "git_revision": _revision(project),
        "template": str(template_path),
        "profile": str(profile_path),
        "input": str(source_input),
        "total_designs": total,
        "designs_per_job": per_job,
        "shard_count": shard_count,
        "compiled_pose_count": compiled_pose_count,
        "pose_semantics": pose_semantics,
        "seed_start": arguments.seed_start,
        "pose_seeds": pose_seeds,
        "diffusion_seeds_per_pose": arguments.diffusion_seeds_per_pose,
        "submitted": arguments.submit,
        "records": records,
    }
    manifest_path = output / "campaign_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"campaign: {output}")
    print(f"manifest: {manifest_path}")
    print(
        f"designs: {total} across {shard_count} shard(s), "
        f"up to {per_job} per GPU job"
    )
    print(
        f"poses: {compiled_pose_count} compiled pose(s); "
        f"{shard_count} GPU model load(s)"
    )

    failed = [
        record["shard_index"]
        for record in records
        if arguments.submit and not record["submitted"]
    ]
    if failed:
        raise SystemExit(f"Campaign submission failed for shards: {failed}")


if __name__ == "__main__":
    main()
