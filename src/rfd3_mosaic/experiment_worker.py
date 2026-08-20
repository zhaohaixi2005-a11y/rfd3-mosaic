"""Execute one frozen RFD3-Mosaic experiment inside an allocated job."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
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
    find_result_jsons,
    gate_result_audits,
    run_result_audits,
)
from rfd3_mosaic.run_index import update_run_state
from rfd3_mosaic.sampling_plan import (
    DesignSamplingAssignment,
    compile_sampling_plan,
    design_sampling_assignments,
    pose_plan_is_stochastic,
)
from rfd3_mosaic.schema import load_user_design

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


def _sampling_assignments(
    config: dict[str, Any],
) -> tuple[DesignSamplingAssignment, ...]:
    """Resolve pose/diffusion semantics for current and legacy frontends."""

    topology = config["topology"]
    sampling = config["sampling"]
    requested = int(sampling.get("designs", 1))
    if topology["kind"] == "user_design":
        design = load_user_design(topology["config"])
        assignments = design_sampling_assignments(
            compile_sampling_plan(design)
        )
        if len(assignments) != requested:
            raise ValueError(
                "Frozen public design and execution envelope disagree on "
                "sampling.designs"
            )
        return assignments

    # Compatibility topologies did not declare a pose distribution.  Keep
    # their one-pose behavior but make every diffusion trajectory explicit.
    return tuple(
        DesignSamplingAssignment(
            design_index=index,
            pose_index=0,
            replicate_index=index,
            pose_seed=None,
            diffusion_seed=int(sampling["seed"]) + index,
        )
        for index in range(requested)
    )


def _uses_stochastic_pose_sampling(config: dict[str, Any]) -> bool:
    """Return whether this run intentionally varies assembly-level pose."""

    topology = config["topology"]
    if topology["kind"] != "user_design":
        return False
    design = load_user_design(topology["config"])
    return pose_plan_is_stochastic(compile_sampling_plan(design))


def _merged_rfd3_input(
    destination: Path,
    *,
    assemblies: dict[int, Any],
    assignments: tuple[DesignSamplingAssignment, ...],
) -> tuple[Path, dict[str, DesignSamplingAssignment]]:
    """Create one multi-example RFD3 input without duplicating user YAMLs."""

    merged: dict[str, Any] = {}
    by_example: dict[str, DesignSamplingAssignment] = {}
    for assignment in assignments:
        assembly = assemblies[assignment.pose_index]
        payload = json.loads(
            assembly.input_path.read_text(encoding="utf-8")
        )
        source = next(iter(payload.values()))
        source_input = Path(str(source["input"]))
        if not source_input.is_absolute():
            source_input = (
                assembly.input_path.parent / source_input
            ).resolve()
        source["input"] = str(source_input)
        source["extra"] = dict(source.get("extra") or {})
        source["extra"].update(
            {
                "mosaic_design_index": assignment.design_index,
                "mosaic_pose_index": assignment.pose_index,
                "mosaic_pose_seed": assignment.pose_seed,
                "mosaic_diffusion_seed": assignment.diffusion_seed,
                "mosaic_replicate_index": assignment.replicate_index,
            }
        )
        example_id = (
            f"design_{assignment.design_index:05d}"
            f"_pose_{assignment.pose_index:05d}"
            f"_rep_{assignment.replicate_index:03d}"
        )
        merged[example_id] = source
        by_example[example_id] = assignment

    destination.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination, by_example


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


def _symmetric_scaffold_packing_runtime(rfd3_input: Path) -> bool:
    """Enable explicit generated-scaffold packing for cyclic seed designs."""

    payload = json.loads(rfd3_input.read_text(encoding="utf-8"))
    example = next(iter(payload.values()))
    plan = (example.get("extra") or {}).get(
        "automatic_symmetric_scaffold_packing"
    )
    return isinstance(plan, dict) and plan.get("mode") == "symmetric_generated"


def _resolved_guidance_overrides(
    rfd3_input: Path,
) -> tuple[str, ...]:
    """Read all compiler-resolved guidance values from the frozen input."""

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
    assignments = _sampling_assignments(config)
    stochastic_pose_sampling = _uses_stochastic_pose_sampling(config)
    unique_pose_indices = sorted(
        {assignment.pose_index for assignment in assignments}
    )
    stochastic_pose_population = len(unique_pose_indices) > 1
    assemblies: dict[int, Any] = {}
    accepted_pose_seeds: dict[int, int | None] = {}
    prevalidation_reports: list[Path] = []
    for pose_index in unique_pose_indices:
        assignment = next(
            item for item in assignments if item.pose_index == pose_index
        )
        maximum_attempts = 64 if assignment.pose_seed is not None else 1
        pose_directory = (
            input_directory / f"pose_{pose_index:05d}"
            if stochastic_pose_population
            else input_directory
        )
        for attempt in range(maximum_attempts):
            # A rejected proposal must not leave compiler artifacts that can
            # leak into its replacement.  This also covers a one-design run
            # whose declared pose distribution is stochastic.
            if pose_directory.exists() and stochastic_pose_sampling:
                shutil.rmtree(pose_directory)
            pose_directory.mkdir(parents=True, exist_ok=True)
            candidate_seed = (
                assignment.pose_seed + attempt * 1_000_003
                if assignment.pose_seed is not None
                else None
            )
            try:
                assembly = compile_experiment_assembly(
                    topology,
                    pose_directory,
                    project_directory=project,
                    experiment_name=config["name"],
                    pose_seed=candidate_seed,
                    example_id=f"pose_{pose_index:05d}",
                )
                prevalidation_report = (
                    pose_directory / "rfd3_prevalidation.json"
                )
                _run(
                    [
                        sys.executable,
                        "-m",
                        "rfd3_mosaic.rfd3_prevalidate",
                        "--input",
                        str(assembly.input_path),
                        "--report",
                        str(prevalidation_report),
                    ]
                )
            except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError):
                if attempt + 1 >= maximum_attempts:
                    raise
                continue
            assemblies[pose_index] = assembly
            accepted_pose_seeds[pose_index] = candidate_seed
            prevalidation_reports.append(prevalidation_report)
            break

    assignments = tuple(
        replace(
            assignment,
            pose_seed=accepted_pose_seeds[assignment.pose_index],
        )
        for assignment in assignments
    )
    rfd3_input, example_assignments = _merged_rfd3_input(
        input_directory / "rfd3_input.json",
        assemblies=assemblies,
        assignments=assignments,
    )
    pose_manifest_path = run_dir / "pose_manifest.json"
    pose_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection_method": (
                    "seeded_hard_feasibility_rejection"
                    if stochastic_pose_sampling
                    else "fixed_pose_reuse"
                ),
                "distribution": (
                    "declared_radius_axial_times_haar_so3"
                    if stochastic_pose_sampling
                    else "fixed_input_pose"
                ),
                "requested_designs": len(assignments),
                "compiled_pose_count": len(assemblies),
                "model_load_count": 1,
                "assignments": [
                    {
                        **assignment.__dict__,
                        "compiled_input": str(
                            assemblies[assignment.pose_index].input_path
                        ),
                        "compiled_input_sha256": _sha256(
                            assemblies[assignment.pose_index].input_path
                        ),
                    }
                    for assignment in assignments
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    # Preserve the long-standing top-level report location while retaining
    # every pose-specific prevalidation beside its compiled input.
    if len(prevalidation_reports) == 1:
        shutil.copy2(
            prevalidation_reports[0],
            run_dir / "rfd3_prevalidation.json",
        )
    else:
        (run_dir / "rfd3_prevalidation.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "compiled_pose_count": len(prevalidation_reports),
                    "reports": [str(path) for path in prevalidation_reports],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    sampler = sampling["sampler"]
    mobility_enabled, mobility_proposal_source = (
        _motif_mobility_runtime(assemblies[0].input_path)
    )
    interface_guidance_enabled = _graph_interface_guidance_runtime(
        assemblies[0].input_path
    )
    scaffold_packing_enabled = _symmetric_scaffold_packing_runtime(
        assemblies[0].input_path
    )
    if interface_guidance_enabled and scaffold_packing_enabled:
        raise ValueError(
            "Compiled input cannot enable graph interfaces and automatic "
            "symmetric scaffold packing simultaneously"
        )
    requested_designs = int(sampling.get("designs", 1))
    inference_command = [
        sys.executable,
        "-m",
        "rfd3.run_inference",
        f"inference_sampler.kind={sampler['kind']}",
        f"out_dir={run_dir}",
        f"inputs={rfd3_input}",
        f"ckpt_path={resources['checkpoint']}",
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
        "++inference_sampler.enable_symmetric_scaffold_packing="
        + str(scaffold_packing_enabled),
        f"low_memory_mode={sampling['low_memory_mode']}",
        "skip_existing=False",
        "dump_trajectories=False",
        "prevalidate_inputs=True",
    ]
    # Resolved preferences also carry the independent intra/inter scaffold
    # field.  Do not couple those overrides to graph-interface activation:
    # supplied-interface jobs may legitimately request a compact monomer core
    # while explicitly declining creation of a second generated interface.
    inference_command.extend(
        _resolved_guidance_overrides(assemblies[0].input_path)
    )
    _run(inference_command)

    result_jsons = find_result_jsons(run_dir)
    expected_designs = requested_designs
    if len(result_jsons) != expected_designs:
        raise RuntimeError(
            "RFD3 output count does not match sampling.designs: "
            f"expected={expected_designs}, observed={len(result_jsons)}"
        )

    design_results: list[dict[str, Any]] = []
    all_reports: list[Path] = []
    mobility_trajectories: list[Path] = []
    observed_examples: set[str] = set()
    for result_json in result_jsons:
        matching_examples = [
            example_id
            for example_id in example_assignments
            if example_id in result_json.stem
        ]
        if len(matching_examples) != 1:
            raise RuntimeError(
                "Cannot map RFD3 result to exactly one design assignment: "
                f"result={result_json.name}, matches={matching_examples}"
            )
        example_id = matching_examples[0]
        observed_examples.add(example_id)
        assignment = example_assignments[example_id]
        assembly = assemblies[assignment.pose_index]
        design_id = result_json.stem.removesuffix("_model_0")
        audit_directory = run_dir / "audits" / design_id
        audit_outcome = run_result_audits(
            run_directory=run_dir,
            rfd3_input=assembly.input_path,
            result_json=result_json,
            semantic_audits=assembly.semantic_audits,
            output_directory=audit_directory,
            python=sys.executable,
            command_runner=_run,
        )
        accepted = True
        rejection_reason = None
        try:
            gate_result_audits(
                audit_outcome.reports,
                python=sys.executable,
                command_runner=_run,
            )
        except RuntimeError as error:
            accepted = False
            rejection_reason = str(error)
        all_reports.extend(audit_outcome.reports)
        if audit_outcome.mobility_trajectory is not None:
            mobility_trajectories.append(
                audit_outcome.mobility_trajectory
            )
        design_results.append(
            {
                "design_index": assignment.design_index,
                "design_id": design_id,
                "pose_index": assignment.pose_index,
                "replicate_index": assignment.replicate_index,
                "pose_seed": assignment.pose_seed,
                "diffusion_seed": assignment.diffusion_seed,
                "compiled_input": str(assembly.input_path),
                "result_json": str(result_json),
                "accepted": accepted,
                "rejection_reason": rejection_reason,
                "reports": [str(path) for path in audit_outcome.reports],
                "mobility_trajectory": (
                    str(audit_outcome.mobility_trajectory)
                    if audit_outcome.mobility_trajectory is not None
                    else None
                ),
            }
        )
    missing_examples = set(example_assignments) - observed_examples
    if missing_examples:
        raise RuntimeError(
            "RFD3 did not produce outputs for compiled design examples: "
            + ", ".join(sorted(missing_examples))
        )
    design_results.sort(key=lambda item: int(item["design_index"]))

    accepted_count = sum(
        bool(record["accepted"]) for record in design_results
    )
    rejected_count = len(design_results) - accepted_count

    completion = {
        "status": "completed",
        "experiment": config["name"],
        "topology": kind,
        "resolved_config_sha256": _sha256(frozen),
        "requested_designs": expected_designs,
        "compiled_pose_count": len(assemblies),
        "pose_manifest": str(pose_manifest_path),
        "model_load_count": 1,
        "produced_designs": len(design_results),
        "accepted_designs": accepted_count,
        "rejected_designs": rejected_count,
        "design_results": design_results,
        "result_json": (
            str(result_jsons[0]) if len(result_jsons) == 1 else None
        ),
        "result_jsons": [str(path) for path in result_jsons],
        "reports": [str(path) for path in all_reports],
        "runtime_provenance": str(runtime_provenance_path),
        "mobility_trajectory": (
            str(mobility_trajectories[0])
            if len(mobility_trajectories) == 1
            else None
        ),
        "mobility_trajectories": [
            str(path) for path in mobility_trajectories
        ],
    }
    summary_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _record_worker_state(config, run_dir, "completed")
    if expected_designs == 1 and rejected_count:
        raise RuntimeError(
            "The generated design failed required result audits: "
            + str(design_results[0]["rejection_reason"])
        )
    print(
        "RFD3-Mosaic experiment completed: "
        f"accepted={accepted_count}/{expected_designs}",
        flush=True,
    )


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
