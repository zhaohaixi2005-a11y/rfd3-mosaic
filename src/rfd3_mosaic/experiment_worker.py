"""Execute one frozen RFD3-Mosaic experiment inside an allocated job."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

import yaml

from rfd3_mosaic.assembly_compiler import compile_experiment_assembly


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _run(command: list[str]) -> None:
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_json(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("*model_0.json"))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one model_0 metadata JSON, observed "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0]


def _symmetry_multiplicity(rfd3_input: Path) -> int:
    payload = json.loads(rfd3_input.read_text(encoding="utf-8"))
    example = next(iter(payload.values()))
    return int(example["extra"]["symmetry_multiplicity"])


def execute(resolved_config: Path, run_dir: Path) -> None:
    config = _load(resolved_config)
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen = run_dir / "resolved_config.yaml"
    shutil.copy2(resolved_config, frozen)
    source_provenance = resolved_config.with_name("provenance.json")
    if source_provenance.is_file():
        shutil.copy2(source_provenance, run_dir / "provenance.json")

    started = {
        "status": "running",
        "experiment": config["name"],
        "topology": config["topology"]["kind"],
        "resolved_config_sha256": _sha256(frozen),
    }
    summary_path = run_dir / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(started, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    topology = config["topology"]
    sampling = config["sampling"]
    resources = config["resources"]
    project = Path(config["project_directory"])
    input_directory = run_dir / "input"
    input_directory.mkdir()

    kind = topology["kind"]
    assembly = compile_experiment_assembly(
        topology,
        input_directory,
        project_directory=project,
        experiment_name=config["name"],
    )
    example_id = assembly.example_id
    rfd3_input = assembly.input_path

    _run(
        [
            sys.executable,
            "-m",
            "rfd3_mosaic.rfd3_prevalidate",
            "--input",
            str(rfd3_input),
            "--report",
            str(run_dir / "rfd3_prevalidation.json"),
        ]
    )

    sampler = sampling["sampler"]
    inference_command = [
        sys.executable,
        "-m",
        "rfd3.run_inference",
        f"inference_sampler.kind={sampler['kind']}",
        f"out_dir={run_dir}",
        f"inputs={rfd3_input}",
        f"ckpt_path={resources['checkpoint']}",
        f"json_keys_subset=[{example_id}]",
        f"seed={sampling['seed']}",
        "diffusion_batch_size=1",
        "n_batches=1",
        f"inference_sampler.num_timesteps={sampling['timesteps']}",
        "inference_sampler.allow_realignment="
        + str(sampler["allow_realignment"]),
        "++inference_sampler.fixed_motif_finalization_mode="
        + str(sampler["fixed_motif_finalization_mode"]),
        "++inference_sampler.preserve_fixed_motif_during_symmetry="
        + str(sampler["preserve_fixed_motif_during_symmetry"]),
        "++inference_sampler.require_motif_constraint_groups="
        + str(sampler["require_motif_constraint_groups"]),
        "++inference_sampler.symmetry_state_mode="
        + str(sampler["symmetry_state_mode"]),
        "++inference_sampler.symmetry_noise_mode="
        + str(sampler["symmetry_noise_mode"]),
        "++inference_sampler.symmetry_execution_backend="
        + str(sampling["execution_backend"]),
        "++inference_sampler.symmetry_neighbour_radius="
        + str(sampling["neighbour_radius"]),
        f"low_memory_mode={sampling['low_memory_mode']}",
        "skip_existing=False",
        "dump_trajectories=False",
        "prevalidate_inputs=True",
    ]
    _run(inference_command)

    result_json = _result_json(run_dir)
    reports = []
    for audit in assembly.semantic_audits:
        report = run_dir / audit.report_name
        _run(
            audit.command(
                python=sys.executable,
                result_json=result_json,
                output_report=report,
            )
        )
        reports.append(report)

    scaffold_report = run_dir / "scaffold_validity_audit.json"
    _run(
        [
            sys.executable,
            "-m",
            "rfd3_mosaic.rfd3_scaffold_audit",
            "--result-json",
            str(result_json),
            "--rfd3-input",
            str(rfd3_input),
            "--output",
            str(scaffold_report),
            "--expected-symmetry-multiplicity",
            str(_symmetry_multiplicity(rfd3_input)),
            "--report-only",
        ]
    )
    reports.append(scaffold_report)

    gate_command = [sys.executable, "-m", "rfd3_mosaic.rfd3_audit_gate"]
    for report in reports:
        gate_command.extend(["--report", str(report)])
    _run(gate_command)

    completion = {
        "status": "completed",
        "experiment": config["name"],
        "topology": kind,
        "resolved_config_sha256": _sha256(frozen),
        "result_json": str(result_json),
        "reports": [str(path) for path in reports],
    }
    summary_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("RFD3-Mosaic experiment completed and passed all required audits")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    arguments = parser.parse_args()
    run_dir = arguments.run_dir.resolve()
    try:
        execute(arguments.resolved_config.resolve(), run_dir)
    except Exception as error:
        summary_path = run_dir / "experiment_summary.json"
        failed: dict[str, Any] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if summary_path.is_file():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                failed = {**existing, **failed}
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(failed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
