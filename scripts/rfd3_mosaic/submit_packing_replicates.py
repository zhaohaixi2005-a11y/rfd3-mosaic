#!/usr/bin/env python3
"""Materialize and optionally submit reproducible flagship packing runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml


BASE_DESIGNS = {
    "locked": (
        "experiments/"
        "lrz_public_c3_locked_packing_patch_capture_v100_50step.yaml"
    ),
    "guided": (
        "experiments/"
        "lrz_public_c3_joint_packing_patch_capture_v100_50step.yaml"
    ),
}
DEFAULT_PROFILE = "configs/rfd3_mosaic/sites/lrz/any_gpu.yaml"
JOB_ID_PATTERN = re.compile(r"Submitted batch job ([0-9]+)")
RUN_INDEX_PATTERN = re.compile(r"^run index:\s+(.+)$", re.MULTILINE)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="", flush=True)
    return completed


def _revision(project: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return completed.stdout.strip() or None


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create independent locked/guided packing replicates and run "
            "the normal Mosaic validation/submission path."
        )
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=tuple(BASE_DESIGNS),
        dest="modes",
    )
    parser.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        help=(
            "Execution profile. Repeated profiles are assigned round-robin "
            "to distinct jobs; Mosaic never duplicates one sample merely "
            "to fill several GPU types."
        ),
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help="Independent diffusion seed; repeat for replicates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for frozen generated YAMLs and campaign manifest.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help=(
            "Persistent run root written into generated YAMLs. This is "
            "required for portable local-GPU campaigns whose filesystem "
            "differs from LRZ."
        ),
    )
    parser.add_argument(
        "--designs-per-job",
        type=int,
        default=2,
        help=(
            "Independent outputs per GPU job. Variable-pose designs receive "
            "one independently seeded pose per output; locked designs keep "
            "one fixed arrangement and vary diffusion only (default: 2)."
        ),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit after validation; without this flag only validate.",
    )
    arguments = parser.parse_args()

    project = Path.cwd().resolve()
    modes = tuple(arguments.modes or ("locked", "guided"))
    profiles = tuple(arguments.profiles or (DEFAULT_PROFILE,))
    seeds = tuple(arguments.seeds or (946, 947, 948))
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be unique non-negative integers")
    if arguments.designs_per_job < 1 or arguments.designs_per_job > 10000:
        raise ValueError("--designs-per-job must be between 1 and 10000")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    first_design = _load(project / BASE_DESIGNS[modes[0]])
    run_root = (
        arguments.run_root.expanduser().resolve()
        if arguments.run_root is not None
        else Path(first_design["output"]["root"])
    )
    output = (
        arguments.output_dir.expanduser().resolve()
        if arguments.output_dir is not None
        else run_root / "_campaigns" / "packing-replicates" / stamp
    )
    output.mkdir(parents=True, exist_ok=False)

    records: list[dict[str, Any]] = []
    job_index = 0
    submission_failures: list[str] = []
    for mode in modes:
        source = project / BASE_DESIGNS[mode]
        base = _load(source)
        source_input = Path(str(base["input"])).expanduser()
        if not source_input.is_absolute():
            source_input = (source.parent / source_input).resolve()
        if not source_input.is_file():
            raise FileNotFoundError(
                f"Input structure does not exist: {source_input}"
            )
        for seed in seeds:
            profile = profiles[job_index % len(profiles)]
            job_index += 1
            payload = dict(base)
            payload["input"] = str(source_input)
            payload["sampling"] = dict(base["sampling"])
            payload["sampling"]["seed"] = seed
            payload["sampling"]["designs"] = arguments.designs_per_job
            payload["name"] = f"c3-{mode}-packing-t50-s{seed}"
            payload["output"] = dict(base["output"])
            payload["output"]["root"] = str(run_root)
            payload["output"]["campaign"] = (
                f"public-c3-packing-evidence-{stamp}"
            )
            generated = output / f"{mode}_s{seed}.yaml"
            generated.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            validate_command = [
                sys.executable,
                "-m",
                "rfd3_mosaic.cli",
                "validate",
                str(generated),
                "--profile",
                profile,
            ]
            validation = _run(validate_command)
            if validation.returncode != 0:
                raise SystemExit(
                    f"Packing validation failed for {mode} seed {seed}"
                )
            record: dict[str, Any] = {
                "mode": mode,
                "seed": seed,
                "requested_designs": arguments.designs_per_job,
                "profile": profile,
                "design": str(generated),
                "validated": True,
                "submitted": False,
                "job_id": None,
                "run_index": None,
                "returncode": None,
            }
            if arguments.submit:
                submission = _run(
                    [
                        sys.executable,
                        "-m",
                        "rfd3_mosaic.cli",
                        "run",
                        str(generated),
                        "--profile",
                        profile,
                    ]
                )
                job_match = JOB_ID_PATTERN.search(submission.stdout)
                index_match = RUN_INDEX_PATTERN.search(submission.stdout)
                record["returncode"] = submission.returncode
                record["job_id"] = (
                    job_match.group(1) if job_match is not None else None
                )
                record["run_index"] = (
                    index_match.group(1).strip()
                    if index_match is not None
                    else None
                )
                record["submitted"] = submission.returncode == 0
                if submission.returncode != 0:
                    submission_failures.append(f"{mode}:s{seed}")
            records.append(record)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _revision(project),
        "purpose": (
            "independent current-revision GPU evidence for locked and guided "
            "interface packing"
        ),
        "run_root": str(run_root),
        "profile_assignment": "round_robin_without_duplicate_samples",
        "designs_per_job": arguments.designs_per_job,
        "requested_output_count": len(modes)
        * len(seeds)
        * arguments.designs_per_job,
        "records": records,
    }
    manifest_path = output / "campaign_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"campaign manifest: {manifest_path}")
    if submission_failures:
        raise SystemExit(
            "Packing campaign submission failed for: "
            + ", ".join(submission_failures)
        )


if __name__ == "__main__":
    main()
