"""Static inverse search over assembly-graph neighbour relations and poses.

This module is deliberately upstream of RFD3.  It expands one public assembly
graph into concrete finite-group neighbour assignments, realizes reproducible
component poses, and ranks the resulting complete assemblies with the same
static compiler reports used by ``validate``.  A ranked candidate is written
as an ordinary public design YAML, so search does not create a parallel cage
execution path.
"""

from __future__ import annotations

from itertools import product
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from rfd3_mosaic.design_compiler import (
    lower_user_design,
    transform_registry_for_design,
)
from rfd3_mosaic.output import compile_standalone
from rfd3_mosaic.schema import (
    CopyRelationSpec,
    UserDesignSpec,
    UserSymmetrySpec,
    load_user_design,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_neighbour_assignments(
    design: UserDesignSpec,
    *,
    interface_ids: Iterable[str] | None = None,
    include_identity: bool = False,
    max_combinations: int = 4096,
) -> tuple[dict[str, str], ...]:
    """Enumerate canonical group-element assignments for selected edges."""

    if not design.components or not design.interfaces:
        raise ValueError(
            "Graph neighbour search requires components and interfaces"
        )
    if max_combinations < 1:
        raise ValueError("max_combinations must be positive")

    requested = tuple(interface_ids or (item.id for item in design.interfaces))
    if not requested:
        raise ValueError("At least one interface must be searched")
    if len(requested) != len(set(requested)):
        raise ValueError("Searched interface IDs must be unique")
    available = {item.id for item in design.interfaces}
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(f"Unknown searched interface IDs: {unknown}")

    registry = transform_registry_for_design(design)
    transform_ids = tuple(
        transform_id
        for transform_id in registry.transform_ids
        if include_identity or transform_id != registry.identity_id
    )
    if not transform_ids:
        raise ValueError("Neighbour search produced no allowed transforms")
    interfaces_by_id = {item.id: item for item in design.interfaces}
    options_by_interface = tuple(
        tuple(
            transform_id
            for transform_id in transform_ids
            if not (
                transform_id == registry.identity_id
                and interfaces_by_id[interface_id].between[0]
                == interfaces_by_id[interface_id].between[1]
            )
        )
        for interface_id in requested
    )
    if any(not options for options in options_by_interface):
        raise ValueError(
            "Neighbour search produced no nonidentity transform for a "
            "self-interface"
        )
    combination_count = 1
    for options in options_by_interface:
        combination_count *= len(options)
    if combination_count > max_combinations:
        raise ValueError(
            "Graph neighbour search would create "
            f"{combination_count} combinations, exceeding "
            f"max_combinations={max_combinations}"
        )
    return tuple(
        dict(zip(requested, values, strict=True))
        for values in product(*options_by_interface)
    )


def _with_assignment(
    design: UserDesignSpec,
    assignment: dict[str, str],
) -> UserDesignSpec:
    interfaces = tuple(
        interface.model_copy(
            update={
                "copy_relation": CopyRelationSpec(
                    transform=assignment[interface.id]
                )
            }
        )
        if interface.id in assignment
        else interface
        for interface in design.interfaces
    )
    return design.model_copy(update={"interfaces": interfaces})


def _with_symmetry(
    design: UserDesignSpec,
    symmetry_id: str,
) -> UserDesignSpec:
    """Replace the candidate symmetry through normal public validation."""

    payload = design.model_dump(mode="python", by_alias=True)
    request = design.symmetry
    if isinstance(request, UserSymmetrySpec):
        payload["symmetry"] = request.model_copy(
            update={
                "id": symmetry_id,
                "secondary_axis": (
                    None
                    if symmetry_id.startswith("C")
                    else request.secondary_axis
                ),
            }
        ).model_dump(mode="python")
    else:
        payload["symmetry"] = symmetry_id
    return UserDesignSpec.model_validate(payload)


def _with_pose_sample(
    design: UserDesignSpec,
    *,
    sample_index: int,
    seed_start: int,
) -> UserDesignSpec:
    sampling = design.sampling
    if sampling.initial_pose is not None:
        pose = sampling.initial_pose.model_copy(
            update={"seed": seed_start + sample_index}
        )
        return design.model_copy(
            update={
                "sampling": sampling.model_copy(
                    update={"initial_pose": pose}
                )
            }
        )
    if sampling.initial_poses:
        count = len(sampling.initial_poses)
        poses = {
            component_id: pose.model_copy(
                update={
                    "seed": seed_start + sample_index * count + index
                }
            )
            for index, (component_id, pose) in enumerate(
                sampling.initial_poses.items()
            )
        }
        return design.model_copy(
            update={
                "sampling": sampling.model_copy(
                    update={"initial_poses": poses}
                )
            }
        )
    if sample_index:
        raise ValueError(
            "pose_samples greater than one requires sampling.initial_pose "
            "or sampling.initial_poses"
        )
    return design


def _summary(
    manifest: dict[str, Any],
    *,
    candidate_id: str,
    symmetry_id: str,
    assignment: dict[str, str],
    pose_sample_index: int,
    directory: Path,
) -> dict[str, Any]:
    validation = manifest["validation"]
    clashes = validation["inter_group_clashes"]
    interfaces = validation["interfaces"]
    linkers = validation["scaffold_link_geometry"]
    objectives = validation["objectives"]
    edge_reports = list(interfaces.get("edges", ()))
    link_reports = list(linkers.get("links", ()))
    contact_count = sum(
        int(edge.get("heavy_atom_contacts_below_4_5A", 0))
        for edge in edge_reports
    )
    centroid_distances = [
        float(edge["centroid_distance"])
        for edge in edge_reports
        if edge.get("centroid_distance") is not None
    ]
    endpoint_distances = [
        float(link["endpoint_distance"])
        for link in link_reports
        if link.get("endpoint_distance") is not None
    ]
    failed_interfaces = list(
        interfaces.get("failed_required_edge_instances", ())
    )
    unsatisfied_output_targets = list(
        interfaces.get("unsatisfied_output_target_instances", ())
    )
    infeasible_links = list(linkers.get("infeasible_link_instances", ()))
    required_objective_failures = int(
        objectives.get("required_failure_count", 0)
    )
    hard_clashes = int(clashes["total_hard_clashes"])
    accepted = bool(
        hard_clashes == 0
        and not failed_interfaces
        and not infeasible_links
        and required_objective_failures == 0
    )
    return {
        "candidate_id": candidate_id,
        "symmetry": symmetry_id,
        "accepted": accepted,
        "neighbour_transforms": dict(assignment),
        "pose_sample_index": pose_sample_index,
        "directory": str(directory.resolve()),
        "hard_clashes": hard_clashes,
        "minimum_inter_group_distance": float(
            clashes["minimum_inter_group_distance"]
        ),
        "failed_required_interfaces": failed_interfaces,
        # These are design objectives for the sampler, not static compiler
        # failures.  Keeping them explicit prevents callers from confusing a
        # diffusion-ready candidate with an already realized interface.
        "unsatisfied_output_targets": unsatisfied_output_targets,
        "requires_diffusion_interface_formation": bool(
            unsatisfied_output_targets
        ),
        "infeasible_links": infeasible_links,
        "required_objective_failures": required_objective_failures,
        "objective_penalty": float(
            objectives.get("total_weighted_penalty", 0.0)
        ),
        "interface_contact_count_below_4_5A": contact_count,
        "mean_interface_centroid_distance": (
            sum(centroid_distances) / len(centroid_distances)
            if centroid_distances
            else None
        ),
        "maximum_linker_endpoint_distance": (
            max(endpoint_distances) if endpoint_distances else None
        ),
        "initialization_samples": manifest.get(
            "initialization_samples", {}
        ),
    }


def _ranking_key(item: dict[str, Any]) -> tuple[Any, ...]:
    if item.get("error") is not None:
        return (
            True,
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            0,
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            item["candidate_id"],
        )
    maximum_span = item.get("maximum_linker_endpoint_distance")
    mean_interface_distance = item.get("mean_interface_centroid_distance")
    return (
        not bool(item["accepted"]),
        len(item["failed_required_interfaces"]),
        len(item["infeasible_links"]),
        int(item["required_objective_failures"]),
        int(item["hard_clashes"]),
        len(item.get("unsatisfied_output_targets", ())),
        -int(item["interface_contact_count_below_4_5A"]),
        float(item["objective_penalty"]),
        float("inf") if maximum_span is None else float(maximum_span),
        (
            float("inf")
            if mean_interface_distance is None
            else float(mean_interface_distance)
        ),
        -float(item["minimum_inter_group_distance"]),
        item["candidate_id"],
    )


def _public_payload(design: UserDesignSpec) -> dict[str, Any]:
    return design.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
    )


def _write_assembly_design(
    design: UserDesignSpec,
    path: Path,
) -> None:
    """Lower one public design into the canonical AssemblySpecification."""

    lowered = lower_user_design(design)
    path.write_text(
        yaml.safe_dump(
            {
                "assembly": lowered.specification.model_dump(mode="json")
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _strict_replay_candidate(
    design_path: Path,
    replay_directory: Path,
    *,
    expected_structure: Path,
) -> dict[str, Any]:
    """Reload and strictly recompile a frozen public candidate.

    Search is upstream of normal execution. A candidate is not publishable
    merely because its in-memory object compiled once: its written YAML must
    survive the ordinary loader/lowering path and reproduce the exact
    initialized assembly that was ranked.
    """

    replayed = load_user_design(design_path)
    replay_directory.mkdir(parents=True, exist_ok=False)
    assembly_path = replay_directory / "assembly.yaml"
    _write_assembly_design(replayed, assembly_path)
    artifacts = compile_standalone(
        assembly_path,
        replay_directory / "compiled",
        base_directory=replayed.input.parent,
        strict_validation=True,
    )
    expected_sha256 = _sha256(expected_structure)
    replay_sha256 = _sha256(artifacts.structure_path)
    if replay_sha256 != expected_sha256:
        raise ValueError(
            "Frozen graph-search candidate did not reproduce the ranked "
            "initialized assembly"
        )
    return {
        "replay_validated": True,
        "replay_directory": str(replay_directory.resolve()),
        "replay_structure_sha256": replay_sha256,
        "replay_atom_count": artifacts.atom_count,
        "replay_residue_count": artifacts.residue_count,
        "replay_chain_count": artifacts.chain_count,
    }


def search_graph_design(
    design: UserDesignSpec,
    output_directory: str | Path,
    *,
    source_path: str | Path | None = None,
    symmetry_ids: Iterable[str] | None = None,
    interface_ids: Iterable[str] | None = None,
    include_identity: bool = False,
    pose_samples: int = 1,
    seed_start: int = 0,
    top_count: int = 20,
    max_combinations: int = 4096,
) -> dict[str, Any]:
    """Enumerate, compile, rank and freeze graph architecture candidates."""

    if pose_samples < 1:
        raise ValueError("pose_samples must be positive")
    if top_count < 1:
        raise ValueError("top_count must be positive")
    if (
        pose_samples > 1
        and design.sampling.initial_pose is None
        and not design.sampling.initial_poses
    ):
        raise ValueError(
            "pose_samples greater than one requires a declared initial pose"
        )
    searched_interface_ids = (
        tuple(interface_ids) if interface_ids is not None else None
    )
    requested_symmetries = tuple(
        symmetry_ids
        or (
            design.symmetry
            if isinstance(design.symmetry, str)
            else design.symmetry.id,
        )
    )
    if not requested_symmetries:
        raise ValueError("At least one candidate symmetry is required")
    if len(requested_symmetries) != len(set(requested_symmetries)):
        raise ValueError("Candidate symmetry IDs must be unique")
    symmetry_designs = tuple(
        _with_symmetry(design, symmetry_id)
        for symmetry_id in requested_symmetries
    )
    assignments_by_symmetry = tuple(
        (
            symmetry_design,
            graph_neighbour_assignments(
                symmetry_design,
                interface_ids=searched_interface_ids,
                include_identity=include_identity,
                max_combinations=max_combinations,
            ),
        )
        for symmetry_design in symmetry_designs
    )
    total_candidates = sum(
        len(assignments) * pose_samples
        for _, assignments in assignments_by_symmetry
    )
    if total_candidates > max_combinations:
        raise ValueError(
            "Graph search would compile "
            f"{total_candidates} candidates, exceeding "
            f"max_combinations={max_combinations}"
        )

    output = Path(output_directory).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Graph search output directory is not empty: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    candidates_root = output / "candidates"
    candidates_root.mkdir(exist_ok=True)
    summaries: list[dict[str, Any]] = []
    designs: dict[str, UserDesignSpec] = {}
    candidate_index = 0
    for symmetry_design, assignments in assignments_by_symmetry:
        symmetry_id = transform_registry_for_design(
            symmetry_design
        ).group_name
        for assignment in assignments:
            assigned = _with_assignment(symmetry_design, assignment)
            for pose_sample_index in range(pose_samples):
                candidate_id = f"candidate_{candidate_index:06d}"
                candidate_index += 1
                candidate_design = _with_pose_sample(
                    assigned,
                    sample_index=pose_sample_index,
                    seed_start=seed_start,
                )
                designs[candidate_id] = candidate_design
                candidate_directory = candidates_root / candidate_id
                candidate_directory.mkdir(exist_ok=False)
                try:
                    assembly_path = candidate_directory / "assembly.yaml"
                    _write_assembly_design(candidate_design, assembly_path)
                    artifacts = compile_standalone(
                        assembly_path,
                        candidate_directory / "compiled",
                        base_directory=candidate_design.input.parent,
                        strict_validation=False,
                    )
                    manifest = json.loads(
                        artifacts.manifest_path.read_text(encoding="utf-8")
                    )
                    summaries.append(
                        _summary(
                            manifest,
                            candidate_id=candidate_id,
                            symmetry_id=symmetry_id,
                            assignment=assignment,
                            pose_sample_index=pose_sample_index,
                            directory=candidate_directory,
                        )
                    )
                except (OSError, TypeError, ValueError) as error:
                    summaries.append(
                        {
                            "candidate_id": candidate_id,
                            "symmetry": symmetry_id,
                            "accepted": False,
                            "neighbour_transforms": dict(assignment),
                            "pose_sample_index": pose_sample_index,
                            "directory": str(candidate_directory),
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )

    ranking = sorted(summaries, key=_ranking_key)
    selected_root = output / "selected"
    selected_root.mkdir(exist_ok=True)
    selectable = [
        candidate
        for candidate in ranking
        if candidate.get("error") is None and bool(candidate["accepted"])
    ]
    selected_count = 0
    replay_failure_count = 0
    for candidate in selectable:
        if selected_count >= top_count:
            break
        candidate_id = str(candidate["candidate_id"])
        candidate_directory = Path(candidate["directory"])
        frozen_path = candidate_directory / "resolved_design.yaml"
        frozen_path.write_text(
            yaml.safe_dump(
                _public_payload(designs[candidate_id]),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        try:
            replay = _strict_replay_candidate(
                frozen_path,
                candidate_directory / "replay",
                expected_structure=(
                    candidate_directory / "compiled/presymmetrized_input.cif"
                ),
            )
        except (OSError, TypeError, ValueError) as error:
            candidate["replay_validated"] = False
            candidate["replay_error"] = f"{type(error).__name__}: {error}"
            replay_failure_count += 1
            continue
        selected_count += 1
        rank = selected_count
        resolved_path = selected_root / f"rank_{rank:04d}_{candidate_id}.yaml"
        resolved_path.write_text(frozen_path.read_text(encoding="utf-8"))
        candidate["rank"] = rank
        candidate["resolved_design"] = str(resolved_path)
        candidate.update(replay)

    source = Path(source_path).expanduser().resolve() if source_path else None
    payload = {
        "schema_version": 1,
        "compiler": "rfd3_mosaic.graph_search",
        "source_design": str(source) if source is not None else None,
        "source_design_sha256": (
            _sha256(source) if source is not None and source.is_file() else None
        ),
        "symmetry": (
            requested_symmetries[0]
            if len(requested_symmetries) == 1
            else None
        ),
        "searched_symmetries": list(requested_symmetries),
        "searched_interfaces": list(
            searched_interface_ids
            or (item.id for item in design.interfaces)
        ),
        "include_identity": include_identity,
        "pose_samples": pose_samples,
        "seed_start": seed_start,
        "candidate_count": len(ranking),
        "accepted_count": sum(bool(item["accepted"]) for item in ranking),
        "diffusion_interface_formation_count": sum(
            bool(item.get("requires_diffusion_interface_formation"))
            for item in ranking
        ),
        "failed_compilation_count": sum(
            item.get("error") is not None for item in ranking
        ),
        "selected_count": selected_count,
        "replay_failure_count": replay_failure_count,
        "ranking": ranking,
    }
    manifest_path = output / "graph_search.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["manifest_path"] = str(manifest_path)
    return payload


__all__ = ["graph_neighbour_assignments", "search_graph_design"]
