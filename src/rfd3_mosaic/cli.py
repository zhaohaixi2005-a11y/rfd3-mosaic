"""Unified command-line interface for RFD3-Mosaic experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Sequence

import yaml

from rfd3_mosaic.experiment import (
    build_execution_plan,
    render_submission,
    resolve_experiment,
)


def _add_quick_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", default="p100")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preset",
        choices=("exact_mosaic", "official_rfd3"),
        default="exact_mosaic",
    )
    parser.add_argument("--name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and render, but do not call sbatch.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rfd3-mosaic")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate",
        help="Validate and resolve an experiment without writing files.",
    )
    validate.add_argument("config", type=Path)
    validate.add_argument("--profile")

    plan = commands.add_parser(
        "plan",
        help="Resolve and display the execution plan without writing files.",
    )
    plan.add_argument("config", type=Path)
    plan.add_argument("--profile")
    plan.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
    )

    render = commands.add_parser(
        "render",
        help="Freeze the config and render a short Slurm job.",
    )
    render.add_argument("config", type=Path)
    render.add_argument("--profile")
    render.add_argument("--output-dir", type=Path)

    submit = commands.add_parser(
        "submit",
        help="Validate, render and submit one experiment with sbatch.",
    )
    submit.add_argument("config", type=Path)
    submit.add_argument("--profile")
    submit.add_argument("--output-dir", type=Path)
    submit.add_argument(
        "--dry-run",
        action="store_true",
        help="Render but do not call sbatch.",
    )

    central = commands.add_parser(
        "central",
        help="Generate around one fixed central motif and submit.",
    )
    central.add_argument("--input", required=True, type=Path)
    central.add_argument("--motif", required=True)
    central.add_argument("--n-length", type=int, default=35)
    central.add_argument("--c-length", type=int, default=35)
    central.add_argument("--campaign", default="central-motif")
    _add_quick_runtime_arguments(central)

    interface = commands.add_parser(
        "interface",
        help="Generate between a fixed interface seed and submit.",
    )
    interface.add_argument("--config", required=True, type=Path)
    pose = interface.add_mutually_exclusive_group(required=True)
    pose.add_argument("--manifest", type=Path)
    pose.add_argument("--pose-seed", type=int)
    interface.add_argument("--length", type=int)
    interface.add_argument("--campaign", default="interface-seed")
    _add_quick_runtime_arguments(interface)
    return parser


def _safe_default_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (cleaned or "rfd3-mosaic")[:64]


def _write_quick_experiment(arguments: argparse.Namespace) -> Path:
    output_root = arguments.output.expanduser().resolve()
    campaign = _safe_default_name(arguments.campaign)
    if campaign != arguments.campaign:
        raise ValueError(
            "--campaign must contain only letters, numbers, '.', '_' or '-'"
        )
    if arguments.command == "central":
        default_name = (
            f"central-n{arguments.n_length}-c{arguments.c_length}"
            f"-s{arguments.seed}"
        )
        topology = {
            "kind": "central_motif",
            "template_input": str(arguments.input.expanduser().resolve()),
            "fixed_selector": arguments.motif,
            "n_terminal_length": arguments.n_length,
            "c_terminal_length": arguments.c_length,
        }
    else:
        default_name = f"interface-t{arguments.steps}-s{arguments.seed}"
        topology = {
            "kind": "interface_seed",
            "config": str(arguments.config.expanduser().resolve()),
            "pose_candidate_manifest": (
                str(arguments.manifest.expanduser().resolve())
                if arguments.manifest is not None
                else None
            ),
            "pose_seed": arguments.pose_seed,
            "linker_length": arguments.length,
        }
    name = _safe_default_name(arguments.name or default_name)
    payload = {
        "schema_version": 1,
        "name": name,
        "topology": topology,
        "sampling": {
            "preset": arguments.preset,
            "timesteps": arguments.steps,
            "seed": arguments.seed,
        },
        "resources": {"profile": arguments.profile},
        "output": {
            "root": str(output_root),
            "campaign": campaign,
        },
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    request_directory = (
        output_root
        / campaign
        / "_requests"
        / name
        / timestamp
    )
    request_directory.mkdir(parents=True, exist_ok=False)
    path = request_directory / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _print_execution_plan(plan: dict, *, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    design = plan["design"]
    sampling = plan["sampling"]
    execution = plan["execution"]
    software = plan["software"]
    print("RFD3-Mosaic execution plan")
    print(f"name:       {plan['name']}")
    print(f"topology:   {design['topology']}")
    print(f"timesteps:  {sampling['timesteps']}")
    print(f"seed:       {sampling['seed']}")
    print(f"preset:     {sampling['preset']}")
    print(f"backend:    {sampling['execution_backend']}")
    print(f"profile:    {execution['profile']}")
    print(f"partitions: {execution['slurm']['partition']}")
    print(f"run root:   {plan['output']['run_root']}")
    print("effective constraints:")
    for constraint in design["effective_constraints"]:
        print(
            "  - "
            + constraint["operator"]
            + f" [{constraint['orbit_scope']}]"
            + f" selector={constraint['selector']}"
        )
    print(f"Mosaic commit: {software['commit']}")
    print(f"Foundry base:  {software['foundry_base_commit']}")
    print(f"tracked dirty: {software['tracked_dirty']}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    arguments = parser.parse_args(argv)
    quick_command = arguments.command in {"central", "interface"}
    try:
        config_path = (
            _write_quick_experiment(arguments)
            if quick_command
            else arguments.config
        )
        experiment = resolve_experiment(
            config_path,
            profile_override=arguments.profile,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    if arguments.command == "validate":
        print("Experiment validation: PASSED")
        print(f"name:     {experiment.name}")
        print(f"topology: {experiment.payload['topology']['kind']}")
        print(f"profile:  {experiment.payload['resources']['profile_name']}")
        print(f"run root: {experiment.run_root}")
        return

    if arguments.command == "plan":
        _print_execution_plan(
            build_execution_plan(experiment),
            output_format=arguments.format,
        )
        return

    if quick_command:
        print(f"generated experiment: {config_path}")

    try:
        script = render_submission(
            experiment,
            output_directory=getattr(arguments, "output_dir", None),
        )
    except (FileExistsError, OSError, ValueError) as error:
        parser.error(str(error))
    print(f"resolved config: {script.parent / 'resolved_config.yaml'}")
    print(f"Slurm script:    {script}")
    if arguments.command == "render" or getattr(arguments, "dry_run", False):
        print("Submission: skipped")
        return

    completed = subprocess.run(
        ["sbatch", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout.strip()
    job_id = output.rsplit(maxsplit=1)[-1] if output else "unknown"
    receipt = {
        "job_id": job_id,
        "sbatch_output": output,
        "script": str(script),
    }
    receipt_path = script.parent / "submission.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(f"submission receipt: {receipt_path}")


if __name__ == "__main__":
    main()
