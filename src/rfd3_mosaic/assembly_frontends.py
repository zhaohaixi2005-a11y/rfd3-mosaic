"""Compatibility frontends that lower user inputs to AssemblySpecification.

Topology names belong at the edge of the program.  Once lowered, central
motifs and interface seeds share the same schema, instance expansion, native
RFD3 feature compiler, and runtime sampler.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from rfd3_mosaic.compile import load_assembly_config
from rfd3_mosaic.design_compiler import lower_user_design
from rfd3_mosaic.design_preferences import resolved_preferences_payload
from rfd3_mosaic.geometry import build_transform_registry
from rfd3_mosaic.schema import (
    AssemblySpecification,
    load_user_design,
)

_RFD3_SELECTOR = re.compile(r"^([^0-9,+-]+)([0-9]+)-([0-9]+)$")


class AuditRequirement(str, Enum):
    """Semantic evidence required by one compiled constraint contract."""

    EXACT_CONSTRAINT_ORBIT = "exact_constraint_orbit"
    INTERFACE_GEOMETRY = "interface_geometry"
    ASSEMBLY_INTERFACE_RELATIONS = "assembly_interface_relations"
    GRAPH_INTERFACE_GUIDANCE = "graph_interface_guidance"
    BOUNDED_COMPONENT_MOBILITY = "bounded_component_mobility"
    CYLINDRICAL_COORDINATES = "cylindrical_coordinates"


@dataclass(frozen=True)
class AssemblyCompilationRequest:
    """One normalized request for the native Assembly -> RFD3 compiler."""

    specification_path: Path
    example_id: str
    audit_requirements: tuple[AuditRequirement, ...]
    pose_seed: int | None = None
    pose_candidate_manifest: Path | None = None
    linker_length: int | None = None
    audit_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.audit_requirements:
            raise ValueError(
                "Every assembly compilation request requires an audit"
            )
        if len(self.audit_requirements) != len(set(self.audit_requirements)):
            raise ValueError("Audit requirements cannot repeat")
        if any(
            not isinstance(item, AuditRequirement)
            for item in self.audit_requirements
        ):
            raise TypeError("Audit requirements must use AuditRequirement")


def _single_rfd3_example(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError("Template RFD3 input must contain exactly one example")
    example = next(iter(payload.values()))
    if not isinstance(example, dict):
        raise ValueError("Template RFD3 example must be a JSON object")
    return example


def _parse_rfd3_selector(selector: str) -> tuple[str, int, int]:
    match = _RFD3_SELECTOR.fullmatch(selector)
    if match is None:
        raise ValueError(
            "fixed_selector must be one contiguous range such as B1-31"
        )
    chain, start_text, end_text = match.groups()
    start, end = int(start_text), int(end_text)
    if end < start:
        raise ValueError("fixed_selector range is reversed")
    return chain, start, end


def _axis_and_center(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = matrix[:3, :3]
    skew_axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if np.linalg.norm(skew_axis) > 1e-8:
        axis = skew_axis
    else:
        _, _, right_t = np.linalg.svd(rotation - np.eye(3))
        axis = right_t[-1]
    axis /= np.linalg.norm(axis)
    center, *_ = np.linalg.lstsq(
        np.eye(3) - rotation,
        matrix[:3, 3],
        rcond=None,
    )
    return axis, center


def _transform_set_from_template(
    symmetry_id: str,
    declared_order: list[str],
    declared_matrices: dict[str, Any],
) -> dict[str, Any]:
    if len(symmetry_id) < 2 or not symmetry_id[1:].isdigit():
        raise ValueError(f"Unsupported native symmetry ID {symmetry_id!r}")
    prefix = symmetry_id[0].upper()
    order = int(symmetry_id[1:])
    if prefix not in {"C", "D"} or order < 2:
        raise ValueError(f"Unsupported native symmetry ID {symmetry_id!r}")
    generator_id = f"{symmetry_id}:r1"
    try:
        generator = np.asarray(declared_matrices[generator_id], dtype=float)
    except KeyError as error:
        raise ValueError(
            f"Declared registry lacks cyclic generator {generator_id!r}"
        ) from error
    axis, center = _axis_and_center(generator)
    transform_set: dict[str, Any] = {
        "type": "cyclic" if prefix == "C" else "dihedral",
        "order": order,
        "axis": axis.tolist(),
        "center": center.tolist(),
    }
    if prefix == "D":
        secondary_id = f"{symmetry_id}:s0"
        try:
            secondary = np.asarray(
                declared_matrices[secondary_id], dtype=float
            )
        except KeyError as error:
            raise ValueError(
                f"Declared registry lacks dihedral generator {secondary_id!r}"
            ) from error
        secondary_axis, _ = _axis_and_center(secondary)
        transform_set["secondary_axis"] = secondary_axis.tolist()

    # Parse through the public schema, then prove that this declarative set
    # reproduces the template registry before it is accepted.
    probe = AssemblySpecification.model_validate(
        {
            "schema_version": 2,
            "mode": "constraint_assembly",
            "fragments": {
                "probe": {
                    "source": "probe.pdb",
                    "selection": "A/1/*",
                    "entity_type": "protein",
                    "role": "functional_motif",
                }
            },
            "motion_groups": {
                "probe_group": {"members": ["probe"], "mode": "fixed"}
            },
            "symmetry": {
                "transform_sets": {"native": transform_set},
                "orbits": {
                    "probe_orbit": {
                        "transform_set": "native",
                        "master_groups": ["probe_group"],
                    }
                },
            },
        }
    )
    registry = build_transform_registry(
        probe.symmetry.transform_sets["native"]
    )
    if list(registry.transform_ids) != list(declared_order):
        raise ValueError(
            "Template declared transform order does not match Mosaic registry"
        )
    for transform_id in declared_order:
        expected = np.asarray(declared_matrices[transform_id], dtype=float)
        observed = registry.transform(transform_id)
        if not np.allclose(observed, expected, atol=1e-6):
            raise ValueError(
                "Template declared symmetry matrices cannot be represented "
                f"by the AssemblySpecification: {transform_id}"
            )
    return transform_set


def lower_central_motif_topology(
    topology: Mapping[str, Any],
    output_directory: str | Path,
    *,
    experiment_name: str,
) -> AssemblyCompilationRequest:
    """Translate the small legacy central-motif input into native Assembly IR."""

    template_path = Path(str(topology["template_input"])).resolve()
    template = _single_rfd3_example(template_path)
    chain, start, end = _parse_rfd3_selector(
        str(topology["fixed_selector"])
    )
    source = Path(str(template["input"]))
    if not source.is_absolute():
        source = (template_path.parent / source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Template structure does not exist: {source}")

    symmetry = dict(template.get("symmetry") or {})
    symmetry_id = str(symmetry.get("id") or "")
    extra = dict(template.get("extra") or {})
    declared_order = symmetry.get("declared_transform_order") or extra.get(
        "registry_transform_order"
    )
    declared_matrices = symmetry.get(
        "declared_transform_matrices"
    ) or extra.get("registry_transform_matrices")
    if not isinstance(declared_order, list) or not isinstance(
        declared_matrices, dict
    ):
        raise ValueError(
            "Central motif lowering requires a compiler-validated declared "
            "symmetry registry in the template input"
        )
    transform_set = _transform_set_from_template(
        symmetry_id,
        declared_order,
        declared_matrices,
    )

    n_length = int(topology["n_terminal_length"])
    c_length = int(topology["c_terminal_length"])
    if n_length < 1 or c_length < 1:
        raise ValueError("Terminal diffusion lengths must be positive")
    payload = {
        "assembly": {
            "schema_version": 2,
            "mode": "constraint_assembly",
            "fragments": {
                "central_motif": {
                    "source": str(source),
                    "selection": f"{chain}/{start}-{end}/*",
                    "entity_type": "protein",
                    "role": "functional_motif",
                    "fixed_atoms": "all",
                }
            },
            "motion_groups": {
                "central_motif_group": {
                    "members": ["central_motif"],
                    "mode": "fixed",
                }
            },
            "symmetry": {
                "transform_sets": {"native": transform_set},
                "orbits": {
                    "central_motif_orbit": {
                        "transform_set": "native",
                        "master_groups": ["central_motif_group"],
                    }
                },
            },
            "generated_segments": {
                "n_terminal": {
                    "anchor": {
                        "fragment": "central_motif",
                        "terminus": "N",
                    },
                    "length": {"minimum": n_length, "maximum": n_length},
                },
                "c_terminal": {
                    "anchor": {
                        "fragment": "central_motif",
                        "terminus": "C",
                    },
                    "length": {"minimum": c_length, "maximum": c_length},
                },
            },
        }
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    specification_path = output / "assembly_specification.yaml"
    specification_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    # Validate at the frontend boundary; the native compiler loads the same
    # file again and owns every subsequent step.
    load_assembly_config(specification_path)
    motif_length = end - start + 1
    return AssemblyCompilationRequest(
        specification_path=specification_path,
        example_id=experiment_name,
        audit_requirements=(AuditRequirement.EXACT_CONSTRAINT_ORBIT,),
        audit_metadata={
            "probe_topology": "central_motif_bidirectional_growth",
            "probe_template_input": str(template_path),
            "probe_fixed_selector": f"A1-{motif_length}",
            "probe_n_terminal_length": n_length,
            "probe_c_terminal_length": c_length,
        },
    )


def lower_experiment_topology(
    topology: Mapping[str, Any],
    output_directory: str | Path,
    *,
    project_directory: str | Path,
    experiment_name: str,
) -> AssemblyCompilationRequest:
    """Normalize supported user frontends without compiling RFD3 features."""

    kind = topology.get("kind")
    if kind == "central_motif":
        return lower_central_motif_topology(
            topology,
            output_directory,
            experiment_name=experiment_name,
        )
    if kind == "interface_seed":
        specification_path = Path(str(topology["config"]))
        if not specification_path.is_absolute():
            specification_path = (
                Path(project_directory) / specification_path
            ).resolve()
        load_assembly_config(specification_path)
        manifest = topology.get("pose_candidate_manifest")
        return AssemblyCompilationRequest(
            specification_path=specification_path,
            example_id=str(topology["example_id"]),
            pose_seed=topology.get("pose_seed"),
            pose_candidate_manifest=(
                Path(str(manifest)).resolve() if manifest is not None else None
            ),
            linker_length=topology.get("linker_length"),
            audit_requirements=(AuditRequirement.INTERFACE_GEOMETRY,),
        )
    if kind == "user_design":
        design_path = Path(str(topology["config"])).resolve()
        design = load_user_design(design_path)
        lowered = lower_user_design(design)
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        specification_path = output / "assembly_specification.yaml"
        specification_path.write_text(
            yaml.safe_dump(
                {
                    "assembly": lowered.specification.model_dump(
                        mode="json"
                    )
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        load_assembly_config(specification_path)
        audit_requirements = []
        if any(
            operator.operator == "fixed_xyz"
            for operator in lowered.constraint_plan.operators
        ):
            audit_requirements.append(
                AuditRequirement.EXACT_CONSTRAINT_ORBIT
            )
        if any(
            operator.operator == "cylindrical"
            for operator in lowered.constraint_plan.operators
        ):
            audit_requirements.append(
                AuditRequirement.CYLINDRICAL_COORDINATES
            )
        # Semantic audits must follow the lowered Assembly IR, not only the
        # fields written explicitly in the public YAML.  Simple terminal
        # designs acquire an output-stage symmetry-neighbour interface during
        # lowering, so consulting ``design.interfaces`` here would enable the
        # sampler guidance while silently omitting both interface audits.
        compiled_interfaces = tuple(
            lowered.specification.interfaces.values()
        )
        if compiled_interfaces:
            audit_requirements.append(
                AuditRequirement.ASSEMBLY_INTERFACE_RELATIONS
            )
        if any(
            interface.required
            and interface.satisfaction_stage == "output"
            and interface.target_geometry.mode == "geometric_constraints"
            for interface in compiled_interfaces
        ):
            audit_requirements.append(
                AuditRequirement.GRAPH_INTERFACE_GUIDANCE
            )
        # Audit the effective lowered runtime contract.  Task presets may
        # derive mobility without mutating the user's fixed_xyz declaration,
        # so checking only the public YAML would let a moving component run
        # without its required mobility evidence.
        if any(
            bool(orbit.component_mobility)
            for orbit in lowered.specification.symmetry.orbits.values()
        ):
            audit_requirements.append(
                AuditRequirement.BOUNDED_COMPONENT_MOBILITY
            )
        return AssemblyCompilationRequest(
            specification_path=specification_path,
            example_id=str(topology["example_id"]),
            audit_requirements=tuple(audit_requirements),
            audit_metadata={
                "public_design": str(design_path),
                "public_task": (
                    design.task.value
                    if design.task is not None
                    else None
                ),
                "fixed_arrangement": design.fixed_arrangement.value,
                "assembly_shape": (
                    design.assembly_shape.model_dump(mode="json")
                    if design.assembly_shape is not None
                    else None
                ),
                "resolved_design_preferences": (
                    resolved_preferences_payload(design)
                ),
                "constraint_plan": lowered.constraint_plan.model_dump(
                    mode="json"
                ),
                "sampling_plan": lowered.sampling_plan.model_dump(
                    mode="json"
                ),
                **lowered.runtime_constraint_metadata,
            },
        )
    raise ValueError(f"Unsupported topology kind {kind!r}")


__all__ = [
    "AssemblyCompilationRequest",
    "AuditRequirement",
    "lower_central_motif_topology",
    "lower_experiment_topology",
]
