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
from rfd3_mosaic.design_preferences import ResolvedDesignPreferences
from rfd3_mosaic.provenance.software import (
    collect_runtime_provenance,
    verify_file_identities,
    verify_repository_identity,
)
from rfd3_mosaic.provenance.source_snapshot import (
    verify_source_snapshot_tree,
)
from rfd3_mosaic.result_auditing import (
    find_result_json,
    gate_result_audits,
    run_result_audits,
)
from rfd3_mosaic.run_index import update_run_state


_AUTHORING_SOURCE_ROLES = frozenset(
    {
        "experiment source",
        "execution profile",
        "Foundry compatibility manifest",
    }
)


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


def _motif_mobility_runtime(rfd3_input: Path) -> tuple[bool, str]:
    """Resolve sampler switches from compiler-emitted orbit mobility."""

    payload = json.loads(rfd3_input.read_text(encoding="utf-8"))
    example = next(iter(payload.values()))
    orbits = (example.get("extra") or {}).get(
        "motif_constraint_orbits", []
    )
    mobile = [
        orbit
        for orbit in orbits
        if orbit.get("mobility_mode") == "orbit_rigid"
    ]
    if not mobile:
        return False, "denoiser"
    proposals = {orbit.get("mobility_proposal") for orbit in mobile}
    if proposals == {"denoiser_fit"}:
        return True, "denoiser"
    if proposals == {"scaffold_objectives"}:
        return True, "scaffold_boundary"
    raise ValueError(
        "One RFD3 run cannot mix or omit motif mobility proposal sources: "
        f"{sorted(str(value) for value in proposals)}"
    )


def _graph_interface_guidance_runtime(rfd3_input: Path) -> bool:
    """Enable the shared sampler field for output-stage contact edges."""

    payload = json.loads(rfd3_input.read_text(encoding="utf-8"))
    example = next(iter(payload.values()))
    relations = (example.get("extra") or {}).get(
        "assembly_interface_relations", []
    )
    return any(
        bool(relation.get("required", True))
        and relation.get("satisfaction_stage") == "output"
        and (relation.get("target_geometry") or {}).get("mode")
        == "geometric_constraints"
        for relation in relations
    )


def _graph_interface_guidance_overrides(
    rfd3_input: Path,
) -> tuple[str, ...]:
    """Read compiler-resolved safe preset values from the frozen input."""

    payload = json.loads(rfd3_input.read_text(encoding="utf-8"))
    example = next(iter(payload.values()))
    preferences = (example.get("extra") or {}).get(
        "resolved_design_preferences"
    )
    if not preferences:
        return ()
    return ResolvedDesignPreferences.model_validate(
        preferences
    ).hydra_overrides()


def _record_worker_state(
    config: dict[str, Any],
    run_dir: Path,
    state: str,
    *,
    error: str | None = None,
) -> None:
    """Update the optional operational index without risking science output."""

    try:
        output = config["output"]
        update_run_state(
            root=output["root"],
            job_id=run_dir.name,
            state=state,
            experiment=str(config["name"]),
            campaign=str(output["campaign"]),
            run_directory=run_dir,
            error=error,
        )
    except (KeyError, OSError, TypeError, ValueError) as index_error:
        print(
            "WARNING: could not update the RFD3-Mosaic run index: "
            f"{index_error}",
            flush=True,
        )


def _verify_render_identity(
    config: dict[str, Any],
    *,
    source_root: Path | None = None,
) -> None:
    """Reject queued work whose source, inputs, or checkpoint changed."""

    provenance = config.get("provenance") or {}
    identity = provenance.get("render_identity")
    if not isinstance(identity, dict):
        raise RuntimeError(
            "Resolved configuration lacks the required render_identity contract"
        )
    if int(identity.get("schema_version", 0)) != 1:
        raise RuntimeError("Unsupported render_identity schema version")
    expected_repository = identity.get("repository")
    if not isinstance(expected_repository, dict):
        raise RuntimeError("render_identity lacks repository provenance")
    source_snapshot = identity.get("source_snapshot")
    if source_snapshot is None:
        verify_repository_identity(
            expected_repository,
            Path(config["project_directory"]),
        )
    else:
        if not isinstance(source_snapshot, dict):
            raise RuntimeError("Invalid source_snapshot identity")
        if int(source_snapshot.get("schema_version", 0)) != 1:
            raise RuntimeError("Unsupported source_snapshot schema version")
        archive = source_snapshot.get("archive")
        if not isinstance(archive, dict):
            raise RuntimeError("source_snapshot lacks archive identity")
        verify_file_identities([archive])
        if source_root is None:
            raise RuntimeError(
                "A source_root is required for snapshot-backed execution"
            )
        verify_source_snapshot_tree(
            source_root,
            expected_manifest_sha256=str(
                source_snapshot["manifest_sha256"]
            ),
        )
    records = identity.get("files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("render_identity lacks frozen runtime dependencies")
    # Authoring inputs are provenance once their values have been resolved;
    # they are not runtime dependencies.  In particular, editing an
    # experiment/profile while an older job is queued must not invalidate the
    # already frozen job.  Keep compatibility with schema-v1 render records
    # that still placed these roles in ``files``.
    runtime_records = [
        record
        for record in records
        if str(record.get("role")) not in _AUTHORING_SOURCE_ROLES
    ]
    if not runtime_records:
        raise RuntimeError("render_identity lacks executable runtime dependencies")
    verify_file_identities(runtime_records)
    checkpoint_record = identity.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise RuntimeError("render_identity lacks checkpoint identity")
    resources = config["resources"]
    if Path(str(checkpoint_record.get("path"))).resolve() != Path(
        resources["checkpoint"]
    ).resolve():
        raise RuntimeError("Resolved checkpoint path differs from render identity")
    if checkpoint_record.get("sha256") != resources.get("checkpoint_sha256"):
        raise RuntimeError("Resolved checkpoint digest differs from render identity")
    verify_file_identities([checkpoint_record])


def execute(
    resolved_config: Path,
    run_dir: Path,
    *,
    source_root: Path | None = None,
) -> None:
    config = _load(resolved_config)
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen = run_dir / "resolved_config.yaml"
    shutil.copy2(resolved_config, frozen)
    source_provenance = resolved_config.with_name("provenance.json")
    if not source_provenance.is_file():
        raise RuntimeError(
            f"Submission provenance is missing: {source_provenance}"
        )
    submission_provenance = json.loads(
        source_provenance.read_text(encoding="utf-8")
    )
    expected_resolved_sha = submission_provenance.get(
        "resolved_config_sha256"
    )
    observed_resolved_sha = _sha256(resolved_config)
    if expected_resolved_sha != observed_resolved_sha:
        raise RuntimeError(
            "Resolved configuration changed after render: "
            f"{expected_resolved_sha!r} != {observed_resolved_sha!r}"
        )
    _verify_render_identity(config, source_root=source_root)
    shutil.copy2(source_provenance, run_dir / "provenance.json")

    resources = config["resources"]
    runtime_provenance_path = run_dir / "runtime_provenance.json"
    runtime_provenance = collect_runtime_provenance(
        Path(config["project_directory"]),
        checkpoint=Path(resources["checkpoint"]),
        checkpoint_sha256=resources.get("checkpoint_sha256"),
    )
    runtime_provenance["execution_source_root"] = (
        str(source_root) if source_root is not None else None
    )
    runtime_provenance["source_snapshot"] = config["provenance"][
        "render_identity"
    ].get("source_snapshot")
    runtime_provenance_path.write_text(
        json.dumps(runtime_provenance, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    started = {
        "status": "running",
        "experiment": config["name"],
        "topology": config["topology"]["kind"],
        "resolved_config_sha256": _sha256(frozen),
        "runtime_provenance": str(runtime_provenance_path),
    }
    summary_path = run_dir / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(started, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _record_worker_state(config, run_dir, "running")

    topology = config["topology"]
    sampling = config["sampling"]
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
    mobility_enabled, mobility_proposal_source = (
        _motif_mobility_runtime(rfd3_input)
    )
    interface_guidance_enabled = _graph_interface_guidance_runtime(
        rfd3_input
    )
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
        "++inference_sampler.enable_orbit_rigid_motif_mobility="
        + str(mobility_enabled),
        "++inference_sampler.motif_mobility_proposal_source="
        + mobility_proposal_source,
        "++inference_sampler.motif_mobility_apply_updates=True",
        "++inference_sampler.enable_graph_interface_guidance="
        + str(interface_guidance_enabled),
        f"low_memory_mode={sampling['low_memory_mode']}",
        "skip_existing=False",
        "dump_trajectories=False",
        "prevalidate_inputs=True",
    ]
    if interface_guidance_enabled:
        inference_command.extend(
            _graph_interface_guidance_overrides(rfd3_input)
        )
    _run(inference_command)

    result_json = find_result_json(run_dir)
    audit_outcome = run_result_audits(
        run_directory=run_dir,
        rfd3_input=rfd3_input,
        result_json=result_json,
        semantic_audits=assembly.semantic_audits,
        python=sys.executable,
        command_runner=_run,
    )
    gate_result_audits(
        audit_outcome.reports,
        python=sys.executable,
        command_runner=_run,
    )

    completion = {
        "status": "completed",
        "experiment": config["name"],
        "topology": kind,
        "resolved_config_sha256": _sha256(frozen),
        "result_json": str(result_json),
        "reports": [str(path) for path in audit_outcome.reports],
        "runtime_provenance": str(runtime_provenance_path),
        "mobility_trajectory": (
            str(audit_outcome.mobility_trajectory)
            if audit_outcome.mobility_trajectory is not None
            else None
        ),
    }
    summary_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _record_worker_state(config, run_dir, "completed")
    print("RFD3-Mosaic experiment completed and passed all required audits")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    arguments = parser.parse_args()
    run_dir = arguments.run_dir.resolve()
    try:
        execute(
            arguments.resolved_config.resolve(),
            run_dir,
            source_root=(
                arguments.source_root.resolve()
                if arguments.source_root is not None
                else None
            ),
        )
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
        try:
            config = _load(arguments.resolved_config.resolve())
        except (OSError, TypeError, ValueError):
            config = None
        if config is not None:
            _record_worker_state(config, run_dir, "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
