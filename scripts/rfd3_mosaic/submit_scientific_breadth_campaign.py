#!/usr/bin/env python3
"""Freeze and optionally submit the non-redundant Mosaic breadth campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml


JOB_ID_PATTERN = re.compile(r"Submitted batch job ([0-9]+)")


@dataclass(frozen=True)
class Case:
    name: str
    source: str
    resource_class: str
    length_summary: str
    mutate: Callable[[dict[str, Any]], None]


def _set_generation_lengths(*lengths: int | dict[str, int]) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        regions = payload.get("generation", [])
        if len(regions) != len(lengths):
            raise ValueError(
                f"Expected {len(lengths)} generation regions, found {len(regions)}"
            )
        for region, length in zip(regions, lengths, strict=True):
            region["length"] = deepcopy(length)

    return mutate


def _set_component_motion(mode: str) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        payload.setdefault("preferences", {})["component_motion"] = mode
        _set_generation_lengths({"minimum": 85, "maximum": 115})(payload)

    return mutate


def _mutate_supplied_complex(payload: dict[str, Any]) -> None:
    _set_generation_lengths(50, 50, 50, 50)(payload)
    payload.setdefault("preferences", {})["component_motion"] = "guided"


def _mutate_multi_component(payload: dict[str, Any]) -> None:
    connections = payload.get("connections", [])
    lengths: tuple[int | dict[str, int], ...] = (
        {"minimum": 25, "maximum": 45},
        {"minimum": 25, "maximum": 45},
        {"minimum": 12, "maximum": 20},
    )
    if len(connections) != len(lengths):
        raise ValueError("Unexpected multi-component connection count")
    for connection, length in zip(connections, lengths, strict=True):
        connection["length"] = deepcopy(length)


CASES: tuple[Case, ...] = (
    Case(
        "c3-fixed-motif-locked",
        "experiments/lrz_public_c3_locked_packing_patch_capture_v100_50step.yaml",
        "small",
        "two 55-residue terminal regions",
        _set_generation_lengths(55, 55),
    ),
    Case(
        "c3-fixed-motif-guided",
        "experiments/lrz_public_c3_joint_packing_patch_capture_v100_50step.yaml",
        "small",
        "two 55-residue terminal regions",
        _set_generation_lengths(55, 55),
    ),
    Case(
        "c3-supplied-interface-locked",
        "experiments/lrz_mosaic_lhd101_c3_guided_50step_template.yaml",
        "small",
        "85--115-residue adjacent-copy linker",
        _set_component_motion("locked"),
    ),
    Case(
        "c3-supplied-interface-guided",
        "experiments/lrz_mosaic_lhd101_c3_guided_50step_template.yaml",
        "small",
        "85--115-residue adjacent-copy linker",
        _set_component_motion("guided"),
    ),
    Case(
        "c3-supplied-complex-terminal",
        "examples/rfd3_mosaic/supplied_interface_higher_oligomer.yaml",
        "small",
        "four independent 50-residue terminal regions",
        _mutate_supplied_complex,
    ),
    Case(
        "c3-multi-component-graph",
        "examples/rfd3_mosaic/public_three_component_graph.yaml",
        "small",
        "25--45, 25--45 and 12--20-residue graph connections",
        _mutate_multi_component,
    ),
    Case(
        "d3-bounded-multi-orbit",
        "experiments/lrz_public_d3_two_orbit_mobility_v100_canary.yaml",
        "small",
        "100-residue between-linker",
        _set_generation_lengths(100),
    ),
    Case(
        "c4-c2-quotient-orbit",
        "experiments/lrz_public_c4_c2_quotient_orbit_v100_canary_s943.yaml",
        "small",
        "60-residue quotient-orbit linker",
        _set_generation_lengths(60),
    ),
    Case(
        "t-bounded-multi-orbit",
        "experiments/lrz_public_t_two_orbit_mobility_t50_large_gpu_canary.yaml",
        "large",
        "60-residue between-linker",
        _set_generation_lengths(60),
    ),
    Case(
        "o-bounded-multi-orbit",
        "experiments/lrz_public_o_orbit_mobility_t50_large_gpu_canary.yaml",
        "large",
        "60-residue between-linker",
        _set_generation_lengths(60),
    ),
    Case(
        "i-fixed-continuous-extension",
        "experiments/lrz_public_i_alternative_motif_continuity_t50_large_gpu.yaml",
        "large",
        "45-residue terminal extension on a 30-residue motif",
        _set_generation_lengths(45),
    ),
)


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _resolve_input(payload: dict[str, Any], source: Path) -> None:
    input_path = Path(str(payload["input"])).expanduser()
    if not input_path.is_absolute():
        input_path = (source.parent / input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input structure does not exist: {input_path}")
    payload["input"] = str(input_path)


def _advance_pose_seeds(sampling: dict[str, Any], seed: int) -> None:
    if isinstance(sampling.get("initial_pose"), dict):
        sampling["initial_pose"]["seed"] = seed
    initial_poses = sampling.get("initial_poses")
    if isinstance(initial_poses, dict):
        for index, pose in enumerate(initial_poses.values()):
            if isinstance(pose, dict):
                pose["seed"] = seed + index


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    environment = os.environ.copy()
    environment.update(
        {"DEBUG": "false", "TYPE_CHECK": "false", "NAN_CHECK": "true"}
    )
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--profile-small", type=Path, required=True)
    parser.add_argument("--profile-large", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--designs", type=int, default=30)
    parser.add_argument(
        "--designs-per-job",
        type=int,
        help=(
            "Optional recovery sharding. By default every scientific task is "
            "one YAML and one GPU job containing all requested designs."
        ),
    )
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=200000)
    parser.add_argument("--site-label", required=True)
    parser.add_argument("--case", action="append", choices=[case.name for case in CASES])
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    designs_per_job = args.designs_per_job or args.designs
    if args.designs < 1 or designs_per_job < 1 or args.timesteps < 1:
        raise ValueError("design counts and timesteps must be positive")
    project = Path.cwd().resolve()
    run_root = args.run_root.expanduser().resolve()
    profiles = {
        "small": args.profile_small.expanduser().resolve(),
        "large": args.profile_large.expanduser().resolve(),
    }
    for path in profiles.values():
        if not path.is_file():
            raise FileNotFoundError(f"Profile does not exist: {path}")

    selected = [case for case in CASES if not args.case or case.name in args.case]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign_name = f"scientific-breadth-v1-{args.site_label}-{stamp}"
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_root / "_campaigns" / "scientific-breadth-v1" / stamp
    )
    output.mkdir(parents=True, exist_ok=False)
    config_dir = output / "designs"
    config_dir.mkdir()

    records: list[dict[str, Any]] = []
    for case_index, case in enumerate(selected):
        source = (project / case.source).resolve()
        base = _load(source)
        _resolve_input(base, source)
        case.mutate(base)
        shard_count = math.ceil(args.designs / designs_per_job)
        for shard_index in range(shard_count):
            design_start = shard_index * designs_per_job
            shard_designs = min(designs_per_job, args.designs - design_start)
            seed = args.seed_start + case_index * 1000 + design_start
            payload = deepcopy(base)
            payload["name"] = f"{case.name}-{shard_index:02d}-s{seed}"
            sampling = payload.setdefault("sampling", {})
            sampling["timesteps"] = args.timesteps
            sampling["designs"] = shard_designs
            sampling["replicates_per_pose"] = 1
            sampling["seed"] = seed
            sampling.setdefault("preset", "exact_mosaic")
            sampling.setdefault("low_memory_mode", True)
            sampling.setdefault("execution_backend", "explicit_all_copy")
            _advance_pose_seeds(sampling, seed + 500000)
            payload["output"] = {
                "root": str(run_root),
                "campaign": campaign_name,
            }
            frozen_name = (
                f"{case.name}.yaml"
                if shard_count == 1
                else f"{case.name}__shard_{shard_index:02d}.yaml"
            )
            frozen = config_dir / frozen_name
            frozen.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

            profile = profiles[case.resource_class]
            record: dict[str, Any] = {
                "case": case.name,
                "resource_class": case.resource_class,
                "length_summary": case.length_summary,
                "shard_index": shard_index,
                "design_start": design_start,
                "requested_designs": shard_designs,
                "seed": seed,
                "config": str(frozen),
                "profile": str(profile),
                "validated": False,
                "submitted": False,
                "job_id": None,
            }
            if args.validate:
                result = _run(
                    [
                        sys.executable,
                        "-m",
                        "rfd3_mosaic.cli",
                        "validate",
                        str(frozen),
                        "--profile",
                        str(profile),
                    ],
                    project,
                )
                record["validation_returncode"] = result.returncode
                record["validated"] = result.returncode == 0
                if result.returncode != 0:
                    records.append(record)
                    continue
            if args.submit:
                result = _run(
                    [
                        sys.executable,
                        "-m",
                        "rfd3_mosaic.cli",
                        "submit",
                        str(frozen),
                        "--profile",
                        str(profile),
                        "--defer-runtime-preflight",
                    ],
                    project,
                )
                match = JOB_ID_PATTERN.search(result.stdout)
                record["submission_returncode"] = result.returncode
                record["job_id"] = match.group(1) if match else None
                record["submitted"] = result.returncode == 0 and match is not None
            records.append(record)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign": campaign_name,
        "site_label": args.site_label,
        "designs_per_case": args.designs,
        "designs_per_job": designs_per_job,
        "timesteps": args.timesteps,
        "case_count": len(selected),
        "requested_designs": len(selected) * args.designs,
        "records": records,
    }
    manifest_path = output / "campaign_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"campaign: {output}")
    print(f"manifest: {manifest_path}")
    print(f"cases: {len(selected)}")
    print(f"requested designs: {len(selected) * args.designs}")
    print(f"jobs: {len(records)}")


if __name__ == "__main__":
    main()
