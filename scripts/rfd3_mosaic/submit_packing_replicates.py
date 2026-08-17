#!/usr/bin/env python3
"""Materialize and optionally submit reproducible flagship packing runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


BASE_DESIGNS = {
    "locked": "experiments/lrz_public_c3_locked_packing_v100_50step.yaml",
    "guided": (
        "experiments/"
        "lrz_public_c3_joint_packing_mobility_v100_50step.yaml"
    ),
}


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


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
        help="Execution profile; repeat for several GPU types.",
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
        "--submit",
        action="store_true",
        help="Submit after validation; without this flag only validate.",
    )
    arguments = parser.parse_args()

    project = Path.cwd().resolve()
    modes = tuple(arguments.modes or ("locked", "guided"))
    profiles = tuple(arguments.profiles or ("v100", "p100"))
    seeds = tuple(arguments.seeds or (946, 947, 948))
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be unique non-negative integers")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    first_design = _load(project / BASE_DESIGNS[modes[0]])
    default_run_root = Path(first_design["output"]["root"])
    output = (
        arguments.output_dir.expanduser().resolve()
        if arguments.output_dir is not None
        else default_run_root / "_campaigns" / "packing-replicates" / stamp
    )
    output.mkdir(parents=True, exist_ok=False)

    records: list[dict[str, Any]] = []
    for mode in modes:
        source = project / BASE_DESIGNS[mode]
        base = _load(source)
        source_input = Path(str(base["input"])).expanduser()
        if not source_input.is_absolute():
            source_input = (source.parent / source_input).resolve()
        for seed in seeds:
            payload = dict(base)
            payload["input"] = str(source_input)
            payload["sampling"] = dict(base["sampling"])
            payload["sampling"]["seed"] = seed
            payload["name"] = f"c3-{mode}-packing-t50-s{seed}"
            payload["output"] = dict(base["output"])
            payload["output"]["campaign"] = (
                f"public-c3-{mode}-packing-replicates"
            )
            generated = output / f"{mode}_s{seed}.yaml"
            generated.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            for profile in profiles:
                validate_command = [
                    sys.executable,
                    "-m",
                    "rfd3_mosaic.cli",
                    "validate",
                    str(generated),
                    "--profile",
                    profile,
                ]
                _run(validate_command)
                if arguments.submit:
                    _run(
                        [
                            sys.executable,
                            "-m",
                            "rfd3_mosaic.cli",
                            "submit",
                            str(generated),
                            "--profile",
                            profile,
                        ]
                    )
                records.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "profile": profile,
                        "design": str(generated),
                        "submitted": arguments.submit,
                    }
                )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "independent GPU evidence for locked and guided interface packing"
        ),
        "records": records,
    }
    manifest_path = output / "campaign_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"campaign manifest: {manifest_path}")


if __name__ == "__main__":
    main()
