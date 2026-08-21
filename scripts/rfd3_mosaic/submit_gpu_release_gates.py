#!/usr/bin/env python3
"""Validate and submit the non-redundant LRZ GPU release-gate matrix."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


SMALL_PROFILE = "configs/rfd3_mosaic/sites/lrz/any_gpu.yaml"
LARGE_PROFILE = "configs/rfd3_mosaic/sites/lrz/large_gpu.yaml"

# One job supplies one distinct piece of evidence. Hardware portability is
# expressed by a comma-separated Slurm partition list, not by duplicating the
# same trajectory on four accelerator models.
GATES: dict[str, dict[str, Any]] = {
    "fixed-components": {
        "tier": "core",
        "design": "experiments/lrz_public_fixed_components_v100_canary.yaml",
        "profile": SMALL_PROFILE,
        "designs": 1,
        "claim": "exact fixed component and output scaffold",
    },
    "locked-packing": {
        "tier": "core",
        "design": (
            "experiments/"
            "lrz_public_c3_locked_packing_patch_capture_v100_50step.yaml"
        ),
        "profile": SMALL_PROFILE,
        "designs": 2,
        "claim": (
            "generated-only interface packing from independently instantiated "
            "poses that remain locked during diffusion"
        ),
    },
    "guided-packing": {
        "tier": "core",
        "design": (
            "experiments/"
            "lrz_public_c3_joint_packing_patch_capture_v100_50step.yaml"
        ),
        "profile": SMALL_PROFILE,
        "designs": 2,
        "claim": (
            "matched initial poses plus joint radial/axial/rotation packing "
            "transaction"
        ),
    },
    "d3-dynamic": {
        "tier": "core",
        "design": (
            "experiments/"
            "lrz_public_d3_two_orbit_mobility_v100_canary.yaml"
        ),
        "profile": SMALL_PROFILE,
        "designs": 1,
        "claim": "six-action D3 dynamic multi-orbit control",
    },
    "c4-c2-quotient": {
        "tier": "core",
        "design": (
            "experiments/"
            "lrz_public_c4_c2_quotient_orbit_v100_canary_s943.yaml"
        ),
        "profile": SMALL_PROFILE,
        "designs": 1,
        "claim": "physical C4/C2 quotient orbit execution",
    },
    "t-static": {
        "tier": "core",
        "design": (
            "experiments/"
            "lrz_public_t_two_orbit_initialized_short_v100_smoke.yaml"
        ),
        "profile": LARGE_PROFILE,
        "designs": 1,
        "claim": "complete twelve-action tetrahedral execution",
    },
    "t-packing": {
        "tier": "core",
        "design": (
            "experiments/"
            "lrz_public_t_designed_interface_packing_v4_v100_canary.yaml"
        ),
        "profile": LARGE_PROFILE,
        "designs": 1,
        "claim": "tetrahedral graph-guidance and final interface audit",
    },
    "o-static": {
        "tier": "extended",
        "design": (
            "experiments/"
            "lrz_public_o_static_runtime_t50_large_gpu_canary.yaml"
        ),
        "profile": LARGE_PROFILE,
        "designs": 1,
        "claim": "50-step twenty-four-action octahedral runtime closure",
    },
    "i-static": {
        "tier": "extended",
        "design": (
            "experiments/"
            "lrz_public_i_static_runtime_t50_large_gpu_canary.yaml"
        ),
        "profile": LARGE_PROFILE,
        "designs": 1,
        "claim": "50-step sixty-action icosahedral runtime closure",
    },
}


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


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


def _git_revision(project: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return completed.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze, CPU-validate and optionally submit the minimal LRZ GPU "
            "evidence matrix."
        )
    )
    parser.add_argument(
        "--tier",
        choices=("core", "extended", "all"),
        default="core",
    )
    parser.add_argument(
        "--gate",
        action="append",
        choices=tuple(GATES),
        help="Run only this named gate; repeat to select several.",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--list", action="store_true")
    arguments = parser.parse_args()

    if arguments.list:
        for name, gate in GATES.items():
            print(f"{name:20s} {gate['tier']:8s} {gate['claim']}")
        return

    project = Path.cwd().resolve()
    if arguments.gate:
        selected_names = tuple(arguments.gate)
    elif arguments.tier == "all":
        selected_names = tuple(GATES)
    else:
        selected_names = tuple(
            name
            for name, gate in GATES.items()
            if gate["tier"] == arguments.tier
        )
    if not selected_names:
        raise ValueError("No GPU gates selected")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        arguments.output_dir.expanduser().resolve()
        if arguments.output_dir is not None
        else Path(
            "/dss/dssfs02/lwp-dss-0001/pn57ki/"
            "pn57ki-dss-0000/haixi/runs/rfd3-mosaic/"
            f"_campaigns/gpu-release-gates/{stamp}"
        )
    )
    output.mkdir(parents=True, exist_ok=False)

    records: list[dict[str, Any]] = []
    for gate_name in selected_names:
        gate = GATES[gate_name]
        source = (project / str(gate["design"])).resolve()
        payload = deepcopy(_load(source))
        source_input = Path(str(payload["input"])).expanduser()
        if not source_input.is_absolute():
            source_input = (source.parent / source_input).resolve()
        payload["input"] = str(source_input)
        # Experiment submission names are intentionally capped at 64
        # characters.  Several descriptive source-design names already use
        # that budget, so appending a suffix makes an otherwise valid frozen
        # design impossible to submit.  Gate names are unique within this
        # matrix and remain short, stable identifiers for the run index.
        payload["name"] = f"{gate_name}-release-gate"
        payload["sampling"] = dict(payload["sampling"])
        payload["sampling"]["designs"] = int(gate["designs"])
        payload["output"] = dict(payload["output"])
        payload["output"]["campaign"] = "gpu-release-gates"
        frozen = output / f"{gate_name}.yaml"
        frozen.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        profile = str((project / str(gate["profile"])).resolve())
        validate = _run(
            [
                sys.executable,
                "-m",
                "rfd3_mosaic.cli",
                "validate",
                str(frozen),
                "--profile",
                profile,
            ]
        )
        record: dict[str, Any] = {
            "gate": gate_name,
            "tier": gate["tier"],
            "claim": gate["claim"],
            "design": str(frozen),
            "profile": profile,
            "requested_designs": gate["designs"],
            "validation_returncode": validate.returncode,
            "submitted": False,
            "submission_output": None,
        }
        if validate.returncode == 0 and arguments.submit:
            submitted = _run(
                [
                    sys.executable,
                    "-m",
                    "rfd3_mosaic.cli",
                    "submit",
                    str(frozen),
                    "--profile",
                    profile,
                ]
            )
            record["submission_returncode"] = submitted.returncode
            record["submission_output"] = submitted.stdout.strip()
            record["submitted"] = submitted.returncode == 0
        records.append(record)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(project),
        "submitted": arguments.submit,
        "records": records,
    }
    manifest_path = output / "gpu_validation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"campaign manifest: {manifest_path}")

    failed_validations = [
        record["gate"]
        for record in records
        if record["validation_returncode"] != 0
    ]
    failed_submissions = [
        record["gate"]
        for record in records
        if arguments.submit and not record["submitted"]
    ]
    if failed_validations or failed_submissions:
        raise SystemExit(
            "GPU campaign incomplete: "
            f"validation_failures={failed_validations}, "
            f"submission_failures={failed_submissions}"
        )


if __name__ == "__main__":
    main()
