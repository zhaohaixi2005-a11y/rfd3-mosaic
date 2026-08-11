"""Deterministic CPU optimization of rigid assembly-component poses.

The optimizer lives upstream of RFD3.  It changes only the explicit initial
SE(3) pose of complete rigid components, recompiles the full symmetry
assembly, and evaluates the same clash, linker, interface and objective
reports used by the production standalone compiler.  It never moves atoms
inside a supplied interface seed and never creates a second execution path.

This first implementation deliberately uses a bounded pattern search instead
of a black-box numerical dependency.  The objective contains discontinuous
hard constraints (atom-pair clashes and linker contour feasibility), for
which deterministic coordinate polling is both easier to audit and more
useful than pretending that a smooth gradient exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
import yaml

from rfd3_mosaic.design_compiler import (
    lower_user_design,
    parse_public_selector,
)
from rfd3_mosaic.output import compile_standalone
from rfd3_mosaic.schema import (
    NumericRange,
    UserDesignSpec,
    UserFixedOrientationSpec,
    UserInitialPoseSpec,
)
from rfd3_mosaic.structure import read_structure_atoms


@dataclass(frozen=True)
class ComponentPose:
    """One absolute rigid pose expressed around the declared symmetry axis."""

    radius: float
    azimuth_deg: float
    axial_offset: float
    rotation_deg: tuple[float, float, float]


@dataclass(frozen=True)
class PoseEvaluation:
    """Compiler-derived feasibility and ranking evidence for one pose."""

    score: tuple[float, ...]
    feasible: bool
    hard_clashes: int
    failed_required_interfaces: tuple[str, ...]
    infeasible_links: tuple[str, ...]
    blocked_linker_corridors: tuple[str, ...]
    required_objective_failures: int
    linker_contour_excess: float
    maximum_linker_endpoint_distance: float | None
    minimum_linker_corridor_clearance: float | None
    minimum_linker_axis_clearance: float | None
    maximum_terminal_tangent_angle_deg: float | None
    minimum_inter_group_distance: float
    objective_penalty: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": list(self.score),
            "feasible": self.feasible,
            "hard_clashes": self.hard_clashes,
            "failed_required_interfaces": list(
                self.failed_required_interfaces
            ),
            "infeasible_links": list(self.infeasible_links),
            "blocked_linker_corridors": list(
                self.blocked_linker_corridors
            ),
            "required_objective_failures": (
                self.required_objective_failures
            ),
            "linker_contour_excess": self.linker_contour_excess,
            "maximum_linker_endpoint_distance": (
                self.maximum_linker_endpoint_distance
            ),
            "minimum_linker_corridor_clearance": (
                self.minimum_linker_corridor_clearance
            ),
            "minimum_linker_axis_clearance": (
                self.minimum_linker_axis_clearance
            ),
            "maximum_terminal_tangent_angle_deg": (
                self.maximum_terminal_tangent_angle_deg
            ),
            "minimum_inter_group_distance": (
                self.minimum_inter_group_distance
            ),
            "objective_penalty": self.objective_penalty,
        }


@dataclass(frozen=True)
class PoseOptimizationResult:
    """A normal public design with exact, replayable optimized poses."""

    design: UserDesignSpec
    initial_evaluation: PoseEvaluation
    final_evaluation: PoseEvaluation
    evaluation_count: int
    accepted_update_count: int
    converged: bool
    component_poses: dict[str, ComponentPose]
    trajectory: tuple[dict[str, Any], ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "method": "deterministic_bounded_pattern_search_v1",
            "evaluation_count": self.evaluation_count,
            "accepted_update_count": self.accepted_update_count,
            "converged": self.converged,
            "initial": self.initial_evaluation.to_dict(),
            "final": self.final_evaluation.to_dict(),
            "component_poses": {
                component_id: {
                    "radius": pose.radius,
                    "azimuth_deg": pose.azimuth_deg,
                    "axial_offset": pose.axial_offset,
                    "rotation_deg": list(pose.rotation_deg),
                }
                for component_id, pose in self.component_poses.items()
            },
            "trajectory": list(self.trajectory),
        }


def _symmetry_frame(
    design: UserDesignSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    request = design.symmetry
    if isinstance(request, str):
        axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        center = np.zeros(3, dtype=np.float64)
    else:
        axis = np.asarray(request.axis, dtype=np.float64)
        center = np.asarray(request.center, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    trial = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    if abs(float(np.dot(trial, axis))) > 0.9:
        trial = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    radial_x = trial - np.dot(trial, axis) * axis
    radial_x /= np.linalg.norm(radial_x)
    radial_y = np.cross(axis, radial_x)
    radial_y /= np.linalg.norm(radial_y)
    return center, axis, radial_x, radial_y


def _selected_component_center(
    design: UserDesignSpec,
    component_id: str,
) -> np.ndarray:
    atoms = read_structure_atoms(
        design.input,
        mmcif_identifier_namespace="label",
    )
    component = design.components[component_id]
    ranges = tuple(
        segment
        for selector in component.selectors
        for segment in parse_public_selector(selector)
    )
    coordinates = [
        atom.coordinate
        for atom in atoms
        if any(
            atom.chain_id == segment.chain_id
            and segment.residue_start
            <= atom.residue_number
            <= segment.residue_end
            for segment in ranges
        )
        and not (
            atom.element.strip().upper().startswith("H")
            or atom.atom_name.strip().upper().startswith("H")
        )
    ]
    if not coordinates:
        raise ValueError(
            f"Assembly component {component_id!r} selects no heavy atoms"
        )
    return np.asarray(coordinates, dtype=np.float64).mean(axis=0)


def _starting_poses(design: UserDesignSpec) -> dict[str, ComponentPose]:
    if not design.components:
        raise ValueError("Continuous pose optimization requires components")
    if design.sampling.initial_pose is not None:
        raise ValueError(
            "Continuous multi-component optimization requires "
            "sampling.initial_poses, not one ambiguous initial_pose"
        )
    center, axis, radial_x, radial_y = _symmetry_frame(design)
    poses: dict[str, ComponentPose] = {}
    for component_id in sorted(design.components):
        declared = design.sampling.initial_poses.get(component_id)
        if declared is not None:
            if declared.orientation.method != "fixed":
                raise ValueError(
                    "Continuous pose optimization needs an explicit fixed "
                    f"starting orientation for {component_id!r}; resolve "
                    "uniform_so3 sampling before optimization"
                )
            radial = np.asarray(
                declared.radial_direction,
                dtype=np.float64,
            )
            radial -= np.dot(radial, axis) * axis
            radial_norm = float(np.linalg.norm(radial))
            if radial_norm <= 1e-8:
                raise ValueError(
                    f"Initial radial direction for {component_id!r} is "
                    "parallel to the symmetry axis"
                )
            radial /= radial_norm
            azimuth = math.degrees(
                math.atan2(
                    float(np.dot(radial, radial_y)),
                    float(np.dot(radial, radial_x)),
                )
            )
            poses[component_id] = ComponentPose(
                radius=(declared.radius.minimum + declared.radius.maximum)
                / 2.0,
                azimuth_deg=azimuth,
                axial_offset=(
                    declared.axial_offset.minimum
                    + declared.axial_offset.maximum
                )
                / 2.0,
                rotation_deg=tuple(
                    float(value)
                    for value in declared.orientation.rotation_deg
                ),
            )
            continue
        source_center = _selected_component_center(design, component_id)
        relative = source_center - center
        axial_offset = float(np.dot(relative, axis))
        radial = relative - axial_offset * axis
        radius = float(np.linalg.norm(radial))
        azimuth = (
            math.degrees(
                math.atan2(
                    float(np.dot(radial, radial_y)),
                    float(np.dot(radial, radial_x)),
                )
            )
            if radius > 1e-8
            else 0.0
        )
        poses[component_id] = ComponentPose(
            radius=radius,
            azimuth_deg=azimuth,
            axial_offset=axial_offset,
            rotation_deg=(0.0, 0.0, 0.0),
        )
    return poses


def initialize_global_seed_layout(
    design: UserDesignSpec,
    *,
    sample_index: int,
    sample_count: int,
    diameter_range: tuple[float, float] | None = None,
) -> UserDesignSpec:
    """Place canonical independent seeds without using their file frames.

    This deterministic low-discrepancy family spans ring radius, within-ASU
    azimuth, axial staggering and orientation.  Every state is subsequently
    evaluated on the complete symmetry-expanded assembly and refined by
    :func:`optimize_design_poses`; it is never advertised as a final pose by
    itself.
    """

    if sample_count < 1:
        raise ValueError("global seed layout sample_count must be positive")
    if not 0 <= sample_index < sample_count:
        raise ValueError("global seed layout sample_index is out of range")
    if not isinstance(design.symmetry, str) or not design.symmetry.startswith(
        "C"
    ):
        raise NotImplementedError(
            "The first global seed-layout initializer supports Cn; Dn/T/O/I "
            "require stabilizer-aware placement"
        )
    try:
        order = int(design.symmetry[1:])
    except ValueError as error:
        raise ValueError(
            f"Invalid cyclic symmetry {design.symmetry!r}"
        ) from error
    component_ids = tuple(sorted(design.components))
    if not component_ids:
        raise ValueError("Global seed layout requires assembly components")

    atoms = read_structure_atoms(
        design.input,
        mmcif_identifier_namespace="label",
    )
    component_extents: dict[str, float] = {}
    for component_id in component_ids:
        component = design.components[component_id]
        segments = tuple(
            segment
            for selector in component.selectors
            for segment in parse_public_selector(selector)
        )
        coordinates = np.asarray(
            [
                atom.coordinate
                for atom in atoms
                if any(
                    atom.chain_id == segment.chain_id
                    and segment.residue_start
                    <= atom.residue_number
                    <= segment.residue_end
                    for segment in segments
                )
                and not (
                    atom.element.strip().upper().startswith("H")
                    or atom.atom_name.strip().upper().startswith("H")
                )
            ],
            dtype=np.float64,
        )
        if coordinates.size == 0:
            raise ValueError(
                f"Component {component_id!r} selects no heavy atoms"
            )
        center = coordinates.mean(axis=0)
        component_extents[component_id] = float(
            np.linalg.norm(coordinates - center, axis=1).max()
        )

    physical_slots = order * len(component_ids)
    if diameter_range is not None:
        lower, upper = diameter_range
        if lower <= 0.0 or upper < lower:
            raise ValueError("Invalid requested assembly diameter range")
        base_radius = (lower + upper) / 4.0
    else:
        target_chord = 2.0 * max(component_extents.values()) + 8.0
        base_radius = target_chord / (
            2.0 * math.sin(math.pi / max(physical_slots, 2))
        )
        base_radius = min(120.0, max(12.0, base_radius))

    golden = 0.6180339887498949
    radius_fraction = ((sample_index * golden) % 1.0) - 0.5
    radius = base_radius * (1.0 + 0.4 * radius_fraction)
    asu_angle = 360.0 / order
    slot_angle = asu_angle / len(component_ids)
    phase = (((sample_index + 1) * golden) % 1.0 - 0.5) * slot_angle
    tilt_levels = (-24.0, -12.0, 0.0, 12.0, 24.0)
    initial_poses: dict[str, UserInitialPoseSpec] = {}
    for component_index, component_id in enumerate(component_ids):
        azimuth = component_index * slot_angle + phase
        angle = math.radians(azimuth)
        radial_direction = (math.cos(angle), math.sin(angle), 0.0)
        tilt = tilt_levels[
            (sample_index + 2 * component_index) % len(tilt_levels)
        ]
        roll = (
            ((sample_index * 137 + component_index * 71) % 360) - 180
        )
        axial_scale = min(8.0, component_extents[component_id] * 0.35)
        axial = (((sample_index + component_index) % 3) - 1) * axial_scale
        initial_poses[component_id] = UserInitialPoseSpec(
            radius=NumericRange(minimum=radius, maximum=radius),
            axial_offset=NumericRange(minimum=axial, maximum=axial),
            radial_direction=radial_direction,
            orientation=UserFixedOrientationSpec(
                rotation_deg=(tilt, 0.0, azimuth + roll)
            ),
            seed=int(design.sampling.seed + sample_index),
        )
    sampling = design.sampling.model_copy(
        update={
            "initial_pose": None,
            "initial_poses": initial_poses,
            "seed": int(design.sampling.seed + sample_index),
        }
    )
    return design.model_copy(update={"sampling": sampling})


def _design_with_poses(
    design: UserDesignSpec,
    poses: dict[str, ComponentPose],
) -> UserDesignSpec:
    _, axis, radial_x, radial_y = _symmetry_frame(design)
    initial_poses: dict[str, UserInitialPoseSpec] = {}
    for component_id, pose in poses.items():
        angle = math.radians(pose.azimuth_deg)
        radial = math.cos(angle) * radial_x + math.sin(angle) * radial_y
        # Suppress floating point drift parallel to a non-canonical axis.
        radial -= np.dot(radial, axis) * axis
        radial /= np.linalg.norm(radial)
        initial_poses[component_id] = UserInitialPoseSpec(
            radius=NumericRange(minimum=pose.radius, maximum=pose.radius),
            axial_offset=NumericRange(
                minimum=pose.axial_offset,
                maximum=pose.axial_offset,
            ),
            radial_direction=tuple(float(value) for value in radial),
            orientation=UserFixedOrientationSpec(
                rotation_deg=pose.rotation_deg
            ),
            seed=design.sampling.seed,
        )
    sampling = design.sampling.model_copy(
        update={"initial_pose": None, "initial_poses": initial_poses}
    )
    return design.model_copy(update={"sampling": sampling})


def _write_assembly(design: UserDesignSpec, path: Path) -> None:
    lowered = lower_user_design(design)
    path.write_text(
        yaml.safe_dump(
            {"assembly": lowered.specification.model_dump(mode="json")},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _evaluation_from_manifest(manifest: dict[str, Any]) -> PoseEvaluation:
    validation = manifest["validation"]
    clashes = validation["inter_group_clashes"]
    interfaces = validation["interfaces"]
    linkers = validation["scaffold_link_geometry"]
    objectives = validation["objectives"]
    failed_interfaces = tuple(
        interfaces.get("failed_required_edge_instances", ())
    )
    infeasible_links = tuple(
        linkers.get("infeasible_link_instances", ())
    )
    hard_clashes = int(clashes["total_hard_clashes"])
    required_failures = int(objectives.get("required_failure_count", 0))
    link_reports = tuple(linkers.get("links", ()))
    contour_excess = float(
        sum(
            max(
                0,
                int(link["minimum_required_residues_at_3_8A"])
                - int(link["configured_maximum_length"]),
            )
            for link in link_reports
            if not bool(link.get("chain_break", False))
        )
    )
    endpoint_distances = [
        float(link["endpoint_distance"])
        for link in link_reports
        if link.get("endpoint_distance") is not None
    ]
    maximum_endpoint = max(endpoint_distances) if endpoint_distances else None
    corridor_clearances = [
        float(link["minimum_interior_chord_fixed_atom_clearance"])
        for link in link_reports
        if link.get("minimum_interior_chord_fixed_atom_clearance") is not None
    ]
    minimum_corridor_clearance = (
        min(corridor_clearances) if corridor_clearances else None
    )
    blocked_corridors = tuple(
        str(link["link_instance_id"])
        for link in link_reports
        if not bool(link.get("chain_break", False))
        and link.get("minimum_interior_chord_fixed_atom_clearance") is not None
        and float(link["minimum_interior_chord_fixed_atom_clearance"]) < 2.0
    )
    axis_clearances = [
        float(link["minimum_endpoint_chord_axis_clearance"])
        for link in link_reports
        if link.get("minimum_endpoint_chord_axis_clearance") is not None
    ]
    minimum_axis_clearance = min(axis_clearances) if axis_clearances else None
    tangent_angles = [
        float(value)
        for link in link_reports
        for value in (
            link.get("from_terminal_tangent_to_chord_angle_deg"),
            link.get("to_terminal_tangent_to_chord_angle_deg"),
        )
        if value is not None
    ]
    maximum_tangent_angle = max(tangent_angles) if tangent_angles else None
    minimum_distance = float(clashes["minimum_inter_group_distance"])
    objective_penalty = float(
        objectives.get("total_weighted_penalty", 0.0)
    )
    hard_failure_count = (
        hard_clashes
        + len(failed_interfaces)
        + len(infeasible_links)
        + len(blocked_corridors)
        + required_failures
    )
    feasible = hard_failure_count == 0
    # Feasibility is lexicographically absolute.  The remaining terms only
    # rank equally feasible/infeasible compiler states; no soft improvement
    # can compensate for a newly introduced hard violation.
    score = (
        float(not feasible),
        float(hard_failure_count),
        float(len(failed_interfaces)),
        float(len(infeasible_links)),
        float(len(blocked_corridors)),
        float(hard_clashes),
        contour_excess,
        float(required_failures),
        objective_penalty,
        float("inf")
        if maximum_tangent_angle is None
        else maximum_tangent_angle,
        float("inf")
        if minimum_corridor_clearance is None
        else -minimum_corridor_clearance,
        float("inf")
        if minimum_axis_clearance is None
        else -minimum_axis_clearance,
        float("inf") if maximum_endpoint is None else maximum_endpoint,
        -minimum_distance,
    )
    return PoseEvaluation(
        score=score,
        feasible=feasible,
        hard_clashes=hard_clashes,
        failed_required_interfaces=failed_interfaces,
        infeasible_links=infeasible_links,
        blocked_linker_corridors=blocked_corridors,
        required_objective_failures=required_failures,
        linker_contour_excess=contour_excess,
        maximum_linker_endpoint_distance=maximum_endpoint,
        minimum_linker_corridor_clearance=minimum_corridor_clearance,
        minimum_linker_axis_clearance=minimum_axis_clearance,
        maximum_terminal_tangent_angle_deg=maximum_tangent_angle,
        minimum_inter_group_distance=minimum_distance,
        objective_penalty=objective_penalty,
    )


def evaluate_design_pose(design: UserDesignSpec) -> PoseEvaluation:
    """Compile one complete assembly pose without retaining scratch files."""

    with tempfile.TemporaryDirectory(prefix="rfd3-mosaic-pose-eval-") as raw:
        root = Path(raw)
        assembly = root / "assembly.yaml"
        _write_assembly(design, assembly)
        outputs = compile_standalone(
            assembly,
            root / "compiled",
            base_directory=design.input.parent,
            strict_validation=False,
        )
        manifest = json.loads(
            outputs.manifest_path.read_text(encoding="utf-8")
        )
    return _evaluation_from_manifest(manifest)


def _replace_pose_value(
    pose: ComponentPose,
    variable: str,
    value: float,
) -> ComponentPose:
    if variable == "radius":
        return ComponentPose(
            radius=max(0.0, value),
            azimuth_deg=pose.azimuth_deg,
            axial_offset=pose.axial_offset,
            rotation_deg=pose.rotation_deg,
        )
    if variable == "azimuth":
        return ComponentPose(
            radius=pose.radius,
            azimuth_deg=((value + 180.0) % 360.0) - 180.0,
            axial_offset=pose.axial_offset,
            rotation_deg=pose.rotation_deg,
        )
    if variable == "axial":
        return ComponentPose(
            radius=pose.radius,
            azimuth_deg=pose.azimuth_deg,
            axial_offset=value,
            rotation_deg=pose.rotation_deg,
        )
    rotation_index = {"rx": 0, "ry": 1, "rz": 2}[variable]
    rotation = list(pose.rotation_deg)
    rotation[rotation_index] = value
    return ComponentPose(
        radius=pose.radius,
        azimuth_deg=pose.azimuth_deg,
        axial_offset=pose.axial_offset,
        rotation_deg=tuple(rotation),
    )


def _fixed_xyz_rotation(rotation_deg: tuple[float, float, float]) -> np.ndarray:
    x, y, z = np.radians(np.asarray(rotation_deg, dtype=np.float64))
    rx = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, math.cos(x), -math.sin(x)),
         (0.0, math.sin(x), math.cos(x))),
        dtype=np.float64,
    )
    ry = np.asarray(
        ((math.cos(y), 0.0, math.sin(y)), (0.0, 1.0, 0.0),
         (-math.sin(y), 0.0, math.cos(y))),
        dtype=np.float64,
    )
    rz = np.asarray(
        ((math.cos(z), -math.sin(z), 0.0),
         (math.sin(z), math.cos(z), 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return rz @ ry @ rx


def _pose_within_cumulative_bounds(
    pose: ComponentPose,
    starting: ComponentPose,
    *,
    maximum_translation: float,
    maximum_rotation_deg: float,
) -> bool:
    azimuth = math.radians(pose.azimuth_deg)
    starting_azimuth = math.radians(starting.azimuth_deg)
    translation = np.asarray(
        (
            pose.radius * math.cos(azimuth)
            - starting.radius * math.cos(starting_azimuth),
            pose.radius * math.sin(azimuth)
            - starting.radius * math.sin(starting_azimuth),
            pose.axial_offset - starting.axial_offset,
        ),
        dtype=np.float64,
    )
    if float(np.linalg.norm(translation)) > maximum_translation + 1e-9:
        return False
    start_rotation = _fixed_xyz_rotation(starting.rotation_deg)
    current_rotation = _fixed_xyz_rotation(pose.rotation_deg)
    relative = start_rotation.T @ current_rotation
    cosine = float(
        np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    )
    rotation_distance = math.degrees(math.acos(cosine))
    return rotation_distance <= maximum_rotation_deg + 1e-9


def optimize_design_poses(
    design: UserDesignSpec,
    *,
    levels: int = 3,
    maximum_translation: float = 12.0,
    maximum_rotation_deg: float = 25.0,
    translation_step: float = 4.0,
    rotation_step_deg: float = 10.0,
) -> PoseOptimizationResult:
    """Jointly improve all component poses under hard compiler constraints.

    Each accepted coordinate update is evaluated on the complete expanded
    assembly.  The final returned design contains exact scalar poses and can
    therefore pass through the normal freeze/hash/strict-replay boundary.
    """

    if levels < 1:
        raise ValueError("pose optimization levels must be positive")
    if maximum_translation <= 0.0 or maximum_rotation_deg <= 0.0:
        raise ValueError("pose optimization bounds must be positive")
    if translation_step <= 0.0 or rotation_step_deg <= 0.0:
        raise ValueError("pose optimization steps must be positive")
    starting = _starting_poses(design)
    component_ids = tuple(sorted(starting))
    if not component_ids:
        raise ValueError("pose optimization found no components")
    current_poses = dict(starting)
    current_design = _design_with_poses(design, current_poses)
    current_evaluation = evaluate_design_pose(current_design)
    initial_evaluation = current_evaluation
    evaluation_count = 1
    accepted_updates = 0
    trajectory: list[dict[str, Any]] = [
        {
            "level": -1,
            "component": None,
            "variable": None,
            "delta": 0.0,
            "evaluation": current_evaluation.to_dict(),
        }
    ]
    variables = ("radius", "azimuth", "axial", "rx", "ry", "rz")
    for level in range(levels):
        translation_delta = translation_step / (2.0**level)
        rotation_delta = rotation_step_deg / (2.0**level)
        level_changed = False
        # Poll atomic multi-component directions first.  These proposals can
        # cross a coordinate-wise barrier (for example, two components that
        # must move apart together without temporarily breaking a linker),
        # and make the result independent of which component happens to sort
        # first.  Coordinate polling below is only the local polish stage.
        joint_patterns = (
            tuple(1.0 for _ in component_ids),
            tuple(
                1.0 if index % 2 == 0 else -1.0
                for index in range(len(component_ids))
            ),
        )
        for variable in variables:
            base_delta = (
                rotation_delta
                if variable in {"rx", "ry", "rz"}
                else translation_delta
            )
            for pattern in joint_patterns:
                for direction in (-1.0, 1.0):
                    trial_poses = dict(current_poses)
                    admissible = True
                    for component_id, component_sign in zip(
                        component_ids,
                        pattern,
                        strict=True,
                    ):
                        pose = current_poses[component_id]
                        delta = direction * component_sign * base_delta
                        if variable == "radius":
                            value = pose.radius + delta
                        elif variable == "azimuth":
                            value = pose.azimuth_deg + math.degrees(
                                delta / max(pose.radius, 1.0)
                            )
                        elif variable == "axial":
                            value = pose.axial_offset + delta
                        else:
                            rotation_index = {
                                "rx": 0,
                                "ry": 1,
                                "rz": 2,
                            }[variable]
                            value = pose.rotation_deg[rotation_index] + delta
                        trial_pose = _replace_pose_value(
                            pose,
                            variable,
                            value,
                        )
                        if not _pose_within_cumulative_bounds(
                            trial_pose,
                            starting[component_id],
                            maximum_translation=maximum_translation,
                            maximum_rotation_deg=maximum_rotation_deg,
                        ):
                            admissible = False
                            break
                        trial_poses[component_id] = trial_pose
                    if not admissible:
                        continue
                    trial_design = _design_with_poses(design, trial_poses)
                    trial_evaluation = evaluate_design_pose(trial_design)
                    evaluation_count += 1
                    if trial_evaluation.score < current_evaluation.score:
                        current_poses = trial_poses
                        current_design = trial_design
                        current_evaluation = trial_evaluation
                        accepted_updates += 1
                        level_changed = True
                        trajectory.append(
                            {
                                "level": level,
                                "component": "__joint__",
                                "variable": variable,
                                "delta": direction * base_delta,
                                "pattern": list(pattern),
                                "evaluation": (
                                    current_evaluation.to_dict()
                                ),
                            }
                        )
        for component_id in component_ids:
            for variable in variables:
                pose = current_poses[component_id]
                if variable == "radius":
                    current_value = pose.radius
                    delta = translation_delta
                    lower = max(
                        0.0,
                        starting[component_id].radius
                        - maximum_translation,
                    )
                    upper = (
                        starting[component_id].radius
                        + maximum_translation
                    )
                elif variable == "azimuth":
                    current_value = pose.azimuth_deg
                    # Arc length at the current radius is approximately the
                    # translational trust region used for radial/axial moves.
                    delta = math.degrees(
                        translation_delta / max(pose.radius, 1.0)
                    )
                    angular_bound = math.degrees(
                        maximum_translation / max(
                            starting[component_id].radius,
                            1.0,
                        )
                    )
                    lower = (
                        starting[component_id].azimuth_deg - angular_bound
                    )
                    upper = (
                        starting[component_id].azimuth_deg + angular_bound
                    )
                elif variable == "axial":
                    current_value = pose.axial_offset
                    delta = translation_delta
                    lower = (
                        starting[component_id].axial_offset
                        - maximum_translation
                    )
                    upper = (
                        starting[component_id].axial_offset
                        + maximum_translation
                    )
                else:
                    index = {"rx": 0, "ry": 1, "rz": 2}[variable]
                    current_value = pose.rotation_deg[index]
                    delta = rotation_delta
                    lower = (
                        starting[component_id].rotation_deg[index]
                        - maximum_rotation_deg
                    )
                    upper = (
                        starting[component_id].rotation_deg[index]
                        + maximum_rotation_deg
                    )
                trial_records: list[
                    tuple[
                        tuple[float, ...],
                        float,
                        dict[str, ComponentPose],
                        UserDesignSpec,
                        PoseEvaluation,
                    ]
                ] = []
                for sign in (-1.0, 1.0):
                    trial_value = min(
                        upper,
                        max(lower, current_value + sign * delta),
                    )
                    if math.isclose(
                        trial_value,
                        current_value,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        continue
                    trial_poses = dict(current_poses)
                    trial_poses[component_id] = _replace_pose_value(
                        pose,
                        variable,
                        trial_value,
                    )
                    if not _pose_within_cumulative_bounds(
                        trial_poses[component_id],
                        starting[component_id],
                        maximum_translation=maximum_translation,
                        maximum_rotation_deg=maximum_rotation_deg,
                    ):
                        continue
                    trial_design = _design_with_poses(design, trial_poses)
                    trial_evaluation = evaluate_design_pose(trial_design)
                    evaluation_count += 1
                    trial_records.append(
                        (
                            trial_evaluation.score,
                            sign * delta,
                            trial_poses,
                            trial_design,
                            trial_evaluation,
                        )
                    )
                if not trial_records:
                    continue
                best = min(trial_records, key=lambda item: item[0])
                if best[0] < current_evaluation.score:
                    _, applied_delta, current_poses, current_design, (
                        current_evaluation
                    ) = best
                    accepted_updates += 1
                    level_changed = True
                    trajectory.append(
                        {
                            "level": level,
                            "component": component_id,
                            "variable": variable,
                            "delta": applied_delta,
                            "evaluation": current_evaluation.to_dict(),
                        }
                    )
        if not level_changed and current_evaluation.feasible:
            # A feasible fixed point at this resolution can proceed directly
            # to the next, finer level without extra exploratory restarts.
            continue
    return PoseOptimizationResult(
        design=current_design,
        initial_evaluation=initial_evaluation,
        final_evaluation=current_evaluation,
        evaluation_count=evaluation_count,
        accepted_update_count=accepted_updates,
        converged=current_evaluation.feasible,
        component_poses=current_poses,
        trajectory=tuple(trajectory),
    )


def optimize_candidate_subset(
    candidates: Iterable[tuple[str, UserDesignSpec, dict[str, Any]]],
    *,
    top_count: int,
    levels: int,
    maximum_translation: float,
    maximum_rotation_deg: float,
) -> tuple[tuple[str, UserDesignSpec, dict[str, Any]], ...]:
    """Optimize the most promising initial candidates, preserving all IDs."""

    if top_count < 1:
        raise ValueError("pose optimization top_count must be positive")
    materialized = list(candidates)
    initial: list[
        tuple[tuple[float, ...], int, PoseEvaluation]
    ] = []
    for index, (_, design, metadata) in enumerate(materialized):
        if metadata.get("preflight_failures"):
            continue
        evaluation = evaluate_design_pose(design)
        initial.append((evaluation.score, index, evaluation))
    selected_indices = {
        index
        for _, index, _ in sorted(initial, key=lambda item: item[0])[
            :top_count
        ]
    }
    output: list[tuple[str, UserDesignSpec, dict[str, Any]]] = []
    for index, (candidate_id, design, metadata) in enumerate(materialized):
        updated_metadata = dict(metadata)
        if index not in selected_indices:
            reason = (
                "preflight_failure"
                if metadata.get("preflight_failures")
                else "outside_initial_geometry_shortlist"
            )
            updated_metadata["pose_optimization"] = {
                "attempted": False,
                "reason": reason,
            }
            if reason == "outside_initial_geometry_shortlist":
                updated_metadata["preflight_failures"] = [
                    *(
                        str(value)
                        for value in metadata.get("preflight_failures", ())
                    ),
                    (
                        "candidate was not in the deterministic CPU pose "
                        "optimization shortlist and is therefore not "
                        "eligible for automatic selection"
                    ),
                ]
            output.append((candidate_id, design, updated_metadata))
            continue
        result = optimize_design_poses(
            design,
            levels=levels,
            maximum_translation=maximum_translation,
            maximum_rotation_deg=maximum_rotation_deg,
        )
        updated_metadata["pose_optimization"] = {
            "attempted": True,
            **result.metadata(),
        }
        if not result.converged:
            existing_failures = tuple(
                str(value)
                for value in updated_metadata.get("preflight_failures", ())
            )
            final = result.final_evaluation
            updated_metadata["preflight_failures"] = [
                *existing_failures,
                (
                    "continuous pose optimization found no state satisfying "
                    "all hard CPU geometry contracts: "
                    f"hard_clashes={final.hard_clashes}, "
                    "failed_required_interfaces="
                    f"{len(final.failed_required_interfaces)}, "
                    f"infeasible_links={len(final.infeasible_links)}, "
                    "blocked_linker_corridors="
                    f"{len(final.blocked_linker_corridors)}, "
                    "required_objective_failures="
                    f"{final.required_objective_failures}"
                ),
            ]
        output.append((candidate_id, result.design, updated_metadata))
    return tuple(output)
