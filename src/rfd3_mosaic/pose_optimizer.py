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

import itertools
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from rfd3_mosaic.design_compiler import (
    lower_user_design,
    parse_public_selector,
)
from rfd3_mosaic.design_preferences import compile_design_preferences
from rfd3_mosaic.output import compile_standalone
from rfd3_mosaic.schema import (
    NumericRange,
    UserDesignSpec,
    UserFixedOrientationSpec,
    UserInitialPoseSpec,
)
from rfd3_mosaic.simple_architecture import symmetry_group_action_count
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
    minimum_inter_group_distance: float | None
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
            "method": "deterministic_connection_block_pattern_search_v2",
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


def _joint_component_patterns(
    design: UserDesignSpec,
    component_ids: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    """Return deterministic atomic directions for coupled pose polling.

    A single-coordinate move can be rejected even when moving two connected
    supplied seeds together is feasible (for example, when a linker must
    remain within its contour bound throughout the search).  The original
    optimizer only tried one all-component direction and one order-dependent
    alternating direction.  Here the user's polymer graph supplies the
    physically relevant two-component blocks.  Designs without explicit
    connections retain a complete pair fallback so the optimizer is still
    useful for expert inputs.

    Patterns are canonicalized up to global sign because the caller polls
    both directions explicitly.
    """

    if not component_ids:
        return ()
    index_by_component = {
        component_id: index
        for index, component_id in enumerate(component_ids)
    }
    patterns: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()

    def add(pattern: tuple[float, ...]) -> None:
        if not any(pattern):
            return
        first_nonzero = next(value for value in pattern if value != 0.0)
        canonical = (
            pattern
            if first_nonzero > 0.0
            else tuple(-value for value in pattern)
        )
        if canonical not in seen:
            seen.add(canonical)
            patterns.append(canonical)

    add(tuple(1.0 for _ in component_ids))
    add(tuple(
        1.0 if index % 2 == 0 else -1.0
        for index in range(len(component_ids))
    ))

    connected_pairs = {
        tuple(sorted((
            connection.from_endpoint.component,
            connection.to_endpoint.component,
        )))
        for connection in design.connections
        if connection.from_endpoint.component
        != connection.to_endpoint.component
        and connection.from_endpoint.component in index_by_component
        and connection.to_endpoint.component in index_by_component
    }
    if not connected_pairs:
        connected_pairs = set(itertools.combinations(component_ids, 2))

    for first, second in sorted(connected_pairs):
        same = [0.0] * len(component_ids)
        same[index_by_component[first]] = 1.0
        same[index_by_component[second]] = 1.0
        add(tuple(same))

        opposed = list(same)
        opposed[index_by_component[second]] = -1.0
        add(tuple(opposed))
    return tuple(patterns)


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

    For Cn/Dn this deterministic low-discrepancy family spans ring radius,
    within-ASU azimuth, axial staggering and orientation.  Full-orbit T/O/I
    components use a spherical low-discrepancy family so the initializer does
    not privilege one polyhedral axis.  Every state is subsequently evaluated
    on the complete symmetry-expanded assembly and refined by
    :func:`optimize_design_poses`; it is never advertised as a final pose by
    itself.  Components with an explicit stabilizer/coset action remain a
    separate, fail-closed placement problem.
    """

    if sample_count < 1:
        raise ValueError("global seed layout sample_count must be positive")
    if not 0 <= sample_index < sample_count:
        raise ValueError("global seed layout sample_index is out of range")
    symmetry_id = (
        design.symmetry if isinstance(design.symmetry, str) else design.symmetry.id
    )
    cyclic_or_dihedral = symmetry_id.startswith(("C", "D"))
    polyhedral = symmetry_id in {"T", "O", "I"}
    if not cyclic_or_dihedral and not polyhedral:
        raise NotImplementedError(
            "Global seed-layout initialization supports Cn/Dn/T/O/I"
        )
    if cyclic_or_dihedral:
        try:
            cyclic_order = int(symmetry_id[1:])
        except ValueError as error:
            raise ValueError(
                f"Invalid cyclic/dihedral symmetry {symmetry_id!r}"
            ) from error
    else:
        cyclic_order = 0
    group_order = symmetry_group_action_count(symmetry_id)
    is_dihedral = symmetry_id.startswith("D")
    component_ids = tuple(sorted(design.components))
    if not component_ids:
        raise ValueError("Global seed layout requires assembly components")
    stabilizer_components = tuple(
        component_id
        for component_id in component_ids
        if design.components[component_id].finite_orbit_action is not None
    )
    if stabilizer_components:
        raise NotImplementedError(
            "Global placement of components with explicit stabilizer/coset "
            "actions requires stabilizer-aware local frames; unsupported "
            f"components: {list(stabilizer_components)}"
        )

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

    physical_slots = group_order * len(component_ids)
    if diameter_range is not None:
        lower, upper = diameter_range
        if lower <= 0.0 or upper < lower:
            raise ValueError("Invalid requested assembly diameter range")
        base_radius = (lower + upper) / 4.0
    else:
        target_chord = 2.0 * max(component_extents.values()) + 8.0
        if polyhedral:
            # Equal-area angular spacing for N points on a sphere is roughly
            # sqrt(4*pi/N).  Convert that angle to a chord length rather than
            # pretending a polyhedral orbit is one planar ring.
            angular_spacing = math.sqrt(
                4.0 * math.pi / max(physical_slots, 2)
            )
            base_radius = target_chord / (
                2.0 * math.sin(min(math.pi, angular_spacing) / 2.0)
            )
        else:
            base_radius = target_chord / (
                2.0 * math.sin(math.pi / max(physical_slots, 2))
            )
        base_radius = min(120.0, max(12.0, base_radius))
        base_radius *= compile_design_preferences(
            design
        ).initial_radius_scale

    golden = 0.6180339887498949
    radius_fraction = ((sample_index * golden) % 1.0) - 0.5
    radius = base_radius * (1.0 + 0.4 * radius_fraction)
    asu_angle = (
        360.0 / cyclic_order if cyclic_or_dihedral else 360.0
    )
    slot_angle = asu_angle / len(component_ids)
    phase = (((sample_index + 1) * golden) % 1.0 - 0.5) * slot_angle
    tilt_levels = (-24.0, -12.0, 0.0, 12.0, 24.0)
    initial_poses: dict[str, UserInitialPoseSpec] = {}
    for component_index, component_id in enumerate(component_ids):
        if polyhedral:
            # A Fibonacci-sphere direction with sample-dependent scrambling.
            # The half-slot offset avoids all named symmetry axes, which
            # would otherwise collapse a nominal full orbit onto a stabilizer
            # orbit before the compiler can evaluate it.
            direction_index = (
                sample_index * len(component_ids) + component_index
            )
            z_fraction = (
                ((direction_index + 0.5) * golden) % 1.0
            )
            z_fraction = min(0.95, max(0.05, z_fraction))
            unit_z = 2.0 * z_fraction - 1.0
            azimuth = (
                360.0 * (((direction_index + 1) * golden) % 1.0)
            )
            total_radius = radius
            radial_radius = total_radius * math.sqrt(
                max(0.0, 1.0 - unit_z * unit_z)
            )
            axial = total_radius * unit_z
        else:
            azimuth = component_index * slot_angle + phase
            radial_radius = radius
            axial_scale = min(
                8.0, component_extents[component_id] * 0.35
            )
            axial = (
                ((sample_index + component_index) % 3) - 1
            ) * axial_scale
        angle = math.radians(azimuth)
        radial_direction = (math.cos(angle), math.sin(angle), 0.0)
        tilt = tilt_levels[
            (sample_index + 2 * component_index) % len(tilt_levels)
        ]
        roll = (
            ((sample_index * 137 + component_index * 71) % 360) - 180
        )
        if is_dihedral and abs(axial) <= 1e-8:
            # A point in the dihedral equatorial plane may lie on a C2
            # stabilizer and collapse the reflected coset onto the cyclic
            # coset.  A deterministic non-zero layer offset gives the full
            # Dn action distinct starting copies; the optimizer may later
            # refine it while the compiler checks clashes and group closure.
            axial = axial_scale * (
                0.5 if (sample_index + component_index) % 2 == 0 else -0.5
            )
        initial_poses[component_id] = UserInitialPoseSpec(
            radius=NumericRange(
                minimum=radial_radius,
                maximum=radial_radius,
            ),
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
    raw_minimum_distance = clashes.get("minimum_inter_group_distance")
    minimum_distance = (
        float(raw_minimum_distance)
        if raw_minimum_distance is not None
        else None
    )
    objective_penalty = float(
        objectives.get("total_weighted_penalty", 0.0)
    )
    # A scaffold linker is generated as a flexible polymer.  Its endpoint
    # chord is therefore a useful routing heuristic, but it is not the path
    # the generated backbone is required to follow.  Treating every fixed
    # atom near that straight chord as a hard impossibility rejects viable
    # long linkers that have ample contour length to route around the seed.
    #
    # Keep ``blocked_corridors`` in the lexicographic soft ranking so pose
    # search still prefers clear, direct routes.  Hard feasibility remains
    # fail-closed for actual fixed-group clashes, required interface failure,
    # insufficient linker contour length, and required output objectives.
    # The generated structure is subsequently checked by the authoritative
    # chain-continuity and atom-clash result audits.
    hard_failure_count = (
        hard_clashes
        + len(failed_interfaces)
        + len(infeasible_links)
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
        float(hard_clashes),
        float(required_failures),
        # Straight-chord obstruction is a routing preference, not proof that
        # a flexible linker is geometrically impossible.
        float(len(blocked_corridors)),
        contour_excess,
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
        (
            -minimum_distance
            if minimum_distance is not None
            else float("-inf")
        ),
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
        joint_patterns = _joint_component_patterns(design, component_ids)
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
            # The optimization shortlist is a compute/quality preference,
            # not a scientific hard constraint.  A candidate outside it may
            # already satisfy every fully expanded compiler contract and can
            # still provide a strict-replay fallback when a better-ranked
            # optimized state fails later adapter checks.  Only genuine
            # topology/ownership preflight failures remain disqualifying.
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
                    "straight_chord_obstructions(ranking_only)="
                    f"{len(final.blocked_linker_corridors)}, "
                    "required_objective_failures="
                    f"{final.required_objective_failures}"
                ),
            ]
        output.append((candidate_id, result.design, updated_metadata))
    return tuple(output)
