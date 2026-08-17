"""Structure-aware lowering of the small public design declaration."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re

import numpy as np

from rfd3_mosaic.constraint_plan import (
    ConstraintOperatorPlan,
    ConstraintPlan,
    compile_constraint_plan,
)
from rfd3_mosaic.geometry import build_transform_registry
from rfd3_mosaic.sampling_plan import (
    SamplingPlan,
    assembly_initialization_payload,
    compile_sampling_plan,
)
from rfd3_mosaic.schema import (
    AssemblySpecification,
    AtomScope,
    BetweenGeneration,
    FixedArrangementPolicy,
    SymmetryTransformSetSpec,
    TerminalGeneration,
    UserDesignTask,
    UserDesignSpec,
    UserSymmetrySpec,
)
from rfd3_mosaic.structure import AtomRecord, read_structure_atoms


_PUBLIC_RANGE = re.compile(
    r"^(?P<chain>[^0-9,/\s]+)(?P<start>-?[0-9]+)"
    r"(?:-(?P<end>-?[0-9]+))?$"
)
_ASSEMBLY_RANGE = re.compile(
    r"^(?P<chain>[^/]+)/(?P<start>-?[0-9]+)"
    r"(?:-(?P<end>-?[0-9]+))?/(?:\*|all|backbone|CA)$",
    re.IGNORECASE,
)
_BACKBONE = frozenset({"N", "CA", "C", "O"})


@dataclass(frozen=True, order=True)
class SelectorSegment:
    chain_id: str
    residue_start: int
    residue_end: int

    @property
    def public_expression(self) -> str:
        return f"{self.chain_id}{self.residue_start}-{self.residue_end}"

    @property
    def assembly_expression(self) -> str:
        return f"{self.chain_id}/{self.residue_start}-{self.residue_end}/*"


@dataclass(frozen=True, order=True)
class AtomIdentity:
    chain_id: str
    residue_number: int
    insertion_code: str
    atom_name: str


@dataclass(frozen=True)
class BoundConstraintOperator:
    plan: ConstraintOperatorPlan
    segments: tuple[SelectorSegment, ...]
    atom_ids: frozenset[AtomIdentity]


@dataclass(frozen=True)
class BoundConstraintPlan:
    source: Path
    operators: tuple[BoundConstraintOperator, ...]


@dataclass(frozen=True)
class InterfaceUsageResolution:
    interface_id: str
    requested: str
    physical_instance_count: int
    satisfied: bool


@dataclass(frozen=True)
class LoweredUserDesign:
    specification: AssemblySpecification
    constraint_plan: ConstraintPlan
    sampling_plan: SamplingPlan
    bound_constraints: BoundConstraintPlan
    interface_usage: tuple[InterfaceUsageResolution, ...] = ()
    runtime_constraint_metadata: dict[str, object] = field(
        default_factory=dict
    )


def _resolve_interface_usage(
    design: UserDesignSpec,
    specification: AssemblySpecification,
) -> tuple[InterfaceUsageResolution, ...]:
    """Validate user multiplicities against unique expanded relations."""

    if not design.interfaces:
        return ()
    # Local import keeps the public schema/lowering layer independent from
    # compilation module initialization while using the canonical expansion.
    from rfd3_mosaic.compile import expand_symmetry_instances

    instances = expand_symmetry_instances(specification)
    physical_by_source: dict[str, set[tuple[str, str]]] = {}
    physical_by_hyperedge: dict[
        str,
        set[tuple[str | None, int]],
    ] = {}
    for edge in instances.interfaces.values():
        physical_by_source.setdefault(edge.source_id, set()).add(
            tuple(
                sorted(
                    (
                        edge.left_port_instance_id,
                        edge.right_port_instance_id,
                    )
                )
            )
        )
        group_id = edge.hyperedge_id or edge.source_id
        physical_by_hyperedge.setdefault(group_id, set()).add(
            (
                edge.orbit_id,
                (
                    edge.action_copy_index
                    if edge.action_copy_index is not None
                    else edge.source_copy_index
                ),
            )
        )

    grouped: dict[str, list[object]] = {}
    for interface in design.interfaces:
        group_id = interface.hyperedge_id or interface.id
        grouped.setdefault(group_id, []).append(interface)

    resolutions: list[InterfaceUsageResolution] = []
    for group_id, members in grouped.items():
        usage_payloads = {
            json.dumps(
                member.use.model_dump(mode="json", exclude_none=True),
                sort_keys=True,
            )
            for member in members
        }
        if len(usage_payloads) != 1:
            raise ValueError(
                f"Interface hyperedge {group_id!r} has inconsistent use "
                "requirements across its pairwise runtime members"
            )
        if len(members) > 1:
            observed_counts = {
                len(physical_by_source.get(member.id, set()))
                for member in members
            }
            if len(observed_counts) != 1:
                raise ValueError(
                    f"Interface hyperedge {group_id!r} expands its member "
                    f"relations with inconsistent multiplicities: "
                    f"{sorted(observed_counts)}"
                )
        observed = len(physical_by_hyperedge.get(group_id, set()))
        usage = members[0].use
        satisfied = usage.accepts(observed)
        resolution = InterfaceUsageResolution(
            interface_id=group_id,
            requested=usage.description,
            physical_instance_count=observed,
            satisfied=satisfied,
        )
        resolutions.append(resolution)
        if not satisfied:
            raise ValueError(
                f"Interface {group_id!r} requested "
                f"{usage.description} physical instances, but the "
                f"declared symmetry/copy relation produces {observed}; use "
                "auto, change the requested range, or let architecture "
                "search choose a compatible symmetry"
            )
    return tuple(resolutions)


def parse_public_selector(selector: str) -> tuple[SelectorSegment, ...]:
    """Parse comma-separated public or assembly-style residue ranges."""

    segments: list[SelectorSegment] = []
    for component in selector.split(","):
        normalized = component.strip()
        match = _PUBLIC_RANGE.fullmatch(normalized)
        if match is None:
            match = _ASSEMBLY_RANGE.fullmatch(normalized)
        if match is None:
            raise ValueError(
                "Selector must contain ranges such as A12-20 or "
                f"A/12-20/*, got {normalized!r}"
            )
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if start > end:
            raise ValueError(f"Selector range is reversed: {normalized!r}")
        segment = SelectorSegment(match.group("chain"), start, end)
        if segment in segments:
            raise ValueError(f"Selector repeats range {normalized!r}")
        segments.append(segment)
    if not segments:
        raise ValueError("Selector cannot be empty")
    return tuple(segments)


def _atom_identity(atom: AtomRecord) -> AtomIdentity:
    return AtomIdentity(
        chain_id=atom.chain_id,
        residue_number=atom.residue_number,
        insertion_code=atom.insertion_code,
        atom_name=atom.atom_name.upper(),
    )


def _component_chain_backbone_coordinates(
    atoms: tuple[AtomRecord, ...],
    selectors: tuple[str, ...],
) -> tuple[tuple[str, tuple[tuple[int, str, str], ...], np.ndarray], ...]:
    """Return comparable backbone coordinates for each selected source chain.

    A component stabilizer acts on complete protomers, while public component
    selectors may split one protomer into several disjoint fixed fragments.
    Grouping those fragments by source chain prevents selector count from
    being confused with stabilizer order.
    """

    segments_by_chain: dict[str, list[SelectorSegment]] = {}
    for selector in selectors:
        for segment in parse_public_selector(selector):
            segments_by_chain.setdefault(segment.chain_id, []).append(segment)
    records = []
    for chain_id, segments in segments_by_chain.items():
        selected = [
            atom
            for atom in atoms
            if atom.chain_id == chain_id
            and atom.atom_name.upper() in _BACKBONE
            and any(
                segment.residue_start
                <= atom.residue_number
                <= segment.residue_end
                for segment in segments
            )
        ]
        if not selected:
            raise ValueError(
                f"Component stabilizer selectors on chain {chain_id!r} "
                "matched no backbone atoms"
            )
        residue_ids = sorted(
            {
                (atom.residue_number, atom.insertion_code)
                for atom in selected
            }
        )
        offsets = {
            residue_id: index
            for index, residue_id in enumerate(residue_ids)
        }
        keyed = sorted(
            (
                (
                    offsets[(atom.residue_number, atom.insertion_code)],
                    atom.residue_name,
                    atom.atom_name.upper(),
                ),
                atom,
            )
            for atom in selected
        )
        records.append(
            (
                chain_id,
                tuple(key for key, _ in keyed),
                np.asarray(
                    [atom.coordinate for _, atom in keyed],
                    dtype=np.float64,
                ),
            )
        )
    return tuple(records)


def _validate_component_finite_actions(
    design: UserDesignSpec,
    atoms: tuple[AtomRecord, ...],
    registry,
    *,
    maximum_rmsd: float = 0.25,
) -> None:
    """Prove that every declared component stabilizer exists in the input.

    Component quotient actions are executable only when the selected source
    oligomer is invariant under the declared stabilizer in the declared
    global frame.  This prevents a valid abstract C2--C3 incidence graph from
    being lowered with unrelated, non-symmetric seed coordinates.
    """

    for component_id, component in design.components.items():
        action = component.finite_orbit_action
        if action is None:
            continue
        stabilizer_ids = tuple(action.stabilizer_transform_ids)
        records = _component_chain_backbone_coordinates(
            atoms,
            component.selectors,
        )
        if len(records) != len(stabilizer_ids):
            raise ValueError(
                f"Component {component_id!r} declares a stabilizer of order "
                f"{len(stabilizer_ids)}, but its selectors describe "
                f"{len(records)} source protomer chains; supply every "
                "stabilizer-related protomer explicitly"
            )
        signatures = [record[1] for record in records]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError(
                f"Component {component_id!r} stabilizer-related protomers "
                "do not have identical ordered backbone signatures"
            )
        try:
            transforms = tuple(
                np.asarray(registry.transform(transform_id), dtype=np.float64)
                for transform_id in stabilizer_ids
            )
        except KeyError as error:
            raise ValueError(
                f"Component {component_id!r} finite action references an "
                f"unknown stabilizer transform: {error}"
            ) from error
        observed = tuple(record[2] for record in records)
        best_maximum = float("inf")

        def bottleneck_assignment(costs: np.ndarray) -> float:
            """Return the minimum possible maximum cost of a perfect match."""

            size = costs.shape[0]
            for threshold in sorted(float(value) for value in np.unique(costs)):
                observed_to_expected = [-1] * size

                def augment(expected_index: int, seen: set[int]) -> bool:
                    for observed_index in range(size):
                        if (
                            observed_index in seen
                            or costs[expected_index, observed_index] > threshold
                        ):
                            continue
                        seen.add(observed_index)
                        previous = observed_to_expected[observed_index]
                        if previous < 0 or augment(previous, seen):
                            observed_to_expected[observed_index] = expected_index
                            return True
                    return False

                if all(augment(index, set()) for index in range(size)):
                    return threshold
            return float("inf")

        # The identity-labelled protomer need not be the first chain in the
        # input.  Try each physical protomer as the canonical source and find
        # the best one-to-one assignment of declared stabilizer images.
        for canonical in observed:
            expected = tuple(
                canonical @ transform[:3, :3].T + transform[:3, 3]
                for transform in transforms
            )
            costs = np.asarray(
                [
                    [
                        float(
                            np.sqrt(
                                np.mean(
                                    np.sum((target - candidate) ** 2, axis=1)
                                )
                            )
                        )
                        for candidate in observed
                    ]
                    for target in expected
                ],
                dtype=np.float64,
            )
            best_maximum = min(
                best_maximum,
                bottleneck_assignment(costs),
            )
        if best_maximum > maximum_rmsd:
            raise ValueError(
                f"Component {component_id!r} does not satisfy its declared "
                f"stabilizer in the supplied global frame: best backbone "
                f"RMSD {best_maximum:.3f} A exceeds {maximum_rmsd:.3f} A"
            )


def _select_segment_atoms(
    atoms: tuple[AtomRecord, ...],
    segment: SelectorSegment,
    scope: AtomScope,
) -> frozenset[AtomIdentity]:
    selected = frozenset(
        _atom_identity(atom)
        for atom in atoms
        if atom.chain_id == segment.chain_id
        and segment.residue_start
        <= atom.residue_number
        <= segment.residue_end
        and (
            scope == AtomScope.ALL
            or (scope == AtomScope.CA and atom.atom_name.upper() == "CA")
            or (
                scope == AtomScope.BACKBONE
                and atom.atom_name.upper() in _BACKBONE
            )
        )
    )
    if not selected:
        raise ValueError(
            "Selector resolved to zero atoms: "
            f"{segment.public_expression} atoms={scope.value}"
        )
    residues = {
        (item.chain_id, item.residue_number) for item in selected
    }
    expected = {
        (segment.chain_id, residue)
        for residue in range(segment.residue_start, segment.residue_end + 1)
    }
    missing = sorted(expected - residues)
    if missing:
        raise ValueError(
            f"Selector {segment.public_expression!r} contains missing "
            f"residues: {missing}"
        )
    return selected


def _dofs_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    if "cartesian_xyz" in left or "cartesian_xyz" in right:
        return True
    return bool(set(left) & set(right))


def bind_constraint_plan(
    design: UserDesignSpec,
    plan: ConstraintPlan | None = None,
) -> BoundConstraintPlan:
    """Resolve selectors to atoms and reject partial-selector conflicts."""

    source = design.input.expanduser().resolve()
    atoms = read_structure_atoms(
        source,
        mmcif_identifier_namespace="label",
    )
    compiled = plan or compile_constraint_plan(design)
    bound: list[BoundConstraintOperator] = []
    for operator in compiled.operators:
        segments = parse_public_selector(operator.selector)
        atom_ids = frozenset().union(
            *(
                _select_segment_atoms(atoms, segment, operator.atoms)
                for segment in segments
            )
        )
        bound.append(BoundConstraintOperator(operator, segments, atom_ids))
    for index, left in enumerate(bound):
        for right in bound[index + 1 :]:
            if not left.atom_ids.intersection(right.atom_ids):
                continue
            if _dofs_overlap(
                left.plan.controlled_dofs,
                right.plan.controlled_dofs,
            ):
                raise ValueError(
                    "Constraints overlap on resolved atoms and degrees of "
                    f"freedom: {left.plan.id} and {right.plan.id}"
                )
    return BoundConstraintPlan(source=source, operators=tuple(bound))


def _one_segment(selector: str, *, label: str) -> SelectorSegment:
    segments = parse_public_selector(selector)
    if len(segments) != 1:
        raise ValueError(f"{label} must select one contiguous residue range")
    return segments[0]


def _segment_contains(
    outer: SelectorSegment,
    inner: SelectorSegment,
) -> bool:
    return (
        outer.chain_id == inner.chain_id
        and outer.residue_start <= inner.residue_start
        and outer.residue_end >= inner.residue_end
    )


def _symmetry_payload(design: UserDesignSpec) -> dict[str, object]:
    request = design.symmetry
    if isinstance(request, str):
        symmetry_id = request
        axis = (0.0, 0.0, 1.0)
        center = (0.0, 0.0, 0.0)
        secondary_axis = None
    else:
        assert isinstance(request, UserSymmetrySpec)
        symmetry_id = request.id
        axis = request.axis
        center = request.center
        secondary_axis = request.secondary_axis
    prefix = symmetry_id[0]
    polyhedral = {
        "T": ("tetrahedral", 12),
        "O": ("octahedral", 24),
        "I": ("icosahedral", 60),
    }
    if prefix in {"C", "D"}:
        order = int(symmetry_id[1:])
        symmetry_type = "cyclic" if prefix == "C" else "dihedral"
    elif symmetry_id in polyhedral:
        symmetry_type, order = polyhedral[symmetry_id]
    else:
        raise NotImplementedError(
            f"Unsupported finite symmetry declaration {symmetry_id!r}"
        )
    payload: dict[str, object] = {
        "type": symmetry_type,
        "order": order,
        "axis": axis,
        "center": center,
    }
    if prefix == "D":
        payload["secondary_axis"] = secondary_axis or (1.0, 0.0, 0.0)
    elif symmetry_id in polyhedral and secondary_axis is not None:
        payload["secondary_axis"] = secondary_axis
    return payload


def transform_registry_for_design(design: UserDesignSpec):
    """Build the canonical finite-group registry for one public design.

    Search, planning and lowering must enumerate exactly the same transform
    identifiers.  Keeping this conversion beside the public-to-Assembly-IR
    symmetry lowering prevents a search tool from inventing a second group
    convention.
    """

    return build_transform_registry(
        SymmetryTransformSetSpec.model_validate(_symmetry_payload(design))
    )


def _length_payload(length: object) -> dict[str, int]:
    if isinstance(length, int):
        return {"minimum": length, "maximum": length}
    return {
        "minimum": int(length.minimum),
        "maximum": int(length.maximum),
    }


def _assembly_shape_objectives(
    design: UserDesignSpec,
) -> dict[str, dict[str, object]]:
    """Lower ordinary size intent into normal required IR objectives."""

    shape = design.assembly_shape
    if shape is None:
        return {}
    objectives: dict[str, dict[str, object]] = {}
    for objective_id, metric, bounds in (
        (
            "assembly_outer_diameter",
            "assemblies.outer_diameter",
            shape.diameter_angstrom,
        ),
        (
            "assembly_cavity_diameter",
            "cavities.minimum_central_void_diameter",
            shape.cavity_diameter_angstrom,
        ),
    ):
        if bounds is None:
            continue
        width = float(bounds.maximum - bounds.minimum)
        objectives[objective_id] = {
            "metric": metric,
            "mode": "range",
            "minimum": float(bounds.minimum),
            "maximum": float(bounds.maximum),
            "scale": max(1.0, width),
            "weight": 1.0,
            "required": True,
        }
    return objectives


def _graph_endpoint_segment(
    design: UserDesignSpec,
    endpoint,
) -> SelectorSegment:
    component = design.components[endpoint.component]
    selector = endpoint.selector or component.selectors[0]
    return _one_segment(selector, label="assembly connection endpoint")


def _graph_interface_geometry(relation) -> dict[str, object]:
    if relation.mode == "preserve_input":
        return {
            "mode": "reference_transform",
            "from_reference_seed": True,
            "translation_tolerance": relation.translation_tolerance,
            "rotation_tolerance_deg": relation.rotation_tolerance_deg,
            "minimum_heavy_atom_contacts": (
                relation.minimum_heavy_atom_contacts
            ),
            "contact_cutoff": relation.cutoff,
        }

    geometry: dict[str, object] = {
        "mode": "geometric_constraints",
        # Public contact edges are intent, not a demand that users tune an
        # arbitrary number of atom pairs.  The runtime derives residue-scale
        # coverage and continuity targets from the two generated sides.
        "coverage": {"mode": "auto"},
    }
    if relation.distance is not None:
        minimum = relation.distance.minimum
        maximum = relation.distance.maximum
        geometry["distance"] = {
            "type": "com",
            "target": (minimum + maximum) / 2.0,
            "tolerance": max((maximum - minimum) / 2.0, 1e-6),
        }
    if relation.minimum_heavy_atom_contacts is not None:
        geometry["contacts"] = {
            "min_heavy_atom_contacts": (
                relation.minimum_heavy_atom_contacts
            ),
            "cutoff": relation.cutoff,
        }
    else:
        # Keep the contact cutoff available to the guidance/audit even when
        # the user correctly leaves interface size on automatic mode.
        geometry["contacts"] = {
            "min_heavy_atom_contacts": 0,
            "cutoff": relation.cutoff,
        }
    return geometry


def _symmetry_axis_and_center(
    design: UserDesignSpec,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(design.symmetry, str):
        axis = (0.0, 0.0, 1.0)
        center = (0.0, 0.0, 0.0)
    else:
        axis = design.symmetry.axis
        center = design.symmetry.center
    axis_vector = np.asarray(axis, dtype=np.float64)
    axis_vector /= np.linalg.norm(axis_vector)
    return axis_vector, np.asarray(center, dtype=np.float64)


def _apply_transform(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    return matrix[:3, :3] @ point + matrix[:3, 3]


def _nearest_symmetry_neighbour(
    registry,
    master_center: np.ndarray,
) -> str:
    """Choose the nearest distinct group image of one component centre."""

    candidates: list[tuple[float, str]] = []
    for transform_id in registry.transform_ids[1:]:
        transformed = _apply_transform(
            registry.transform(transform_id),
            master_center,
        )
        distance = float(np.linalg.norm(transformed - master_center))
        if distance > 1e-6:
            candidates.append((distance, transform_id))
    if not candidates:
        raise ValueError(
            "Automatic interface planning could not find a distinct "
            "symmetry neighbour; place the motif away from every symmetry "
            "stabilizer or use expert components/ports/interfaces"
        )
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def _generic_orbit_direction(
    design: UserDesignSpec,
    registry,
) -> tuple[np.ndarray, float]:
    """Find a deterministic generic direction with well-separated copies.

    This is a small geometry planner, not a random pose guess.  It searches a
    fixed set of directions and maximizes the minimum pairwise separation of
    their complete finite-group orbit.  A non-zero axial fraction avoids
    accidentally placing T/O/I motifs on a vertex/edge/face stabilizer.
    """

    axis, _ = _symmetry_axis_and_center(design)
    trial = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    if abs(float(np.dot(trial, axis))) > 0.9:
        trial = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    radial_x = trial - float(np.dot(trial, axis)) * axis
    radial_x /= np.linalg.norm(radial_x)
    radial_y = np.cross(axis, radial_x)

    symmetry_id = (
        design.symmetry
        if isinstance(design.symmetry, str)
        else design.symmetry.id
    )
    axial_fractions = (
        (0.0,) if symmetry_id.startswith("C") else (0.23, 0.47, 0.71)
    )
    best: tuple[float, np.ndarray, float] | None = None
    for axial_fraction in axial_fractions:
        for azimuth_index in range(16):
            angle = 2.0 * np.pi * azimuth_index / 16.0
            radial = np.cos(angle) * radial_x + np.sin(angle) * radial_y
            point = radial + axial_fraction * axis
            point /= np.linalg.norm(point)
            orbit = np.stack(
                [
                    _apply_transform(registry.transform(item), point)
                    for item in registry.transform_ids
                ]
            )
            minimum = min(
                float(np.linalg.norm(orbit[left] - orbit[right]))
                for left in range(len(orbit))
                for right in range(left + 1, len(orbit))
            )
            candidate = (minimum, radial, axial_fraction)
            if best is None or candidate[0] > best[0] + 1e-12:
                best = candidate
    assert best is not None
    return best[1], best[2]


def _automatic_simple_component_plan(
    design: UserDesignSpec,
    registry,
    atoms: tuple[AtomRecord, ...],
) -> tuple[dict[str, object] | None, np.ndarray]:
    """Plan a safe ordinary-user pose only when the input is degenerate.

    Already positioned motifs keep their input frame.  A motif at a symmetry
    stabilizer (for example at the origin) receives a deterministic generic
    orbit pose large enough to prevent immediate copy overlap.
    """

    coordinates = np.asarray(
        [atom.coordinate for atom in atoms if atom.element.upper() != "H"],
        dtype=np.float64,
    )
    if coordinates.size == 0:
        coordinates = np.asarray(
            [atom.coordinate for atom in atoms],
            dtype=np.float64,
        )
    center = coordinates.mean(axis=0)
    extent = float(np.linalg.norm(coordinates - center, axis=1).max())
    image_distances = [
        float(
            np.linalg.norm(
                _apply_transform(registry.transform(item), center) - center
            )
        )
        for item in registry.transform_ids[1:]
    ]
    minimum_image_distance = min(image_distances)
    if minimum_image_distance >= max(4.0, 0.35 * extent):
        return None, center

    radial_direction, axial_fraction = _generic_orbit_direction(
        design,
        registry,
    )
    axis, symmetry_center = _symmetry_axis_and_center(design)
    unit_point = radial_direction + axial_fraction * axis
    orbit = np.stack(
        [
            _apply_transform(registry.transform(item), unit_point)
            for item in registry.transform_ids
        ]
    )
    unit_separation = min(
        float(np.linalg.norm(orbit[left] - orbit[right]))
        for left in range(len(orbit))
        for right in range(left + 1, len(orbit))
    )
    target_separation = max(12.0, 2.0 * extent + 6.0)
    radial = target_separation / unit_separation
    axial_offset = radial * axial_fraction
    planned_center = (
        symmetry_center + radial * radial_direction + axial_offset * axis
    )
    payload: dict[str, object] = {
        "random_seed": design.sampling.seed,
        "center_method": "interface_heavy_atom_com",
        "orientation": {
            "method": "fixed",
            "rotation_deg": (0.0, 0.0, 0.0),
        },
        "placement": {
            "radius": {"mean": radial, "range": 0.0},
            "axial_offset": {"mean": axial_offset, "range": 0.0},
            "radial_direction": tuple(float(item) for item in radial_direction),
        },
    }
    return payload, planned_center


def _initialization_center(
    design: UserDesignSpec,
    payload: dict[str, object],
) -> np.ndarray:
    """Recover the deterministic centre encoded by an IR initialization."""

    axis, symmetry_center = _symmetry_axis_and_center(design)
    placement = payload["placement"]
    assert isinstance(placement, dict)
    radial_value = placement["radius"]
    axial_value = placement["axial_offset"]
    assert isinstance(radial_value, dict)
    assert isinstance(axial_value, dict)
    radial = float(radial_value["mean"])
    axial = float(axial_value["mean"])
    direction = np.asarray(
        placement["radial_direction"],
        dtype=np.float64,
    )
    direction -= float(np.dot(direction, axis)) * axis
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError(
            "Initial pose radial_direction is parallel to the symmetry axis"
        )
    direction /= direction_norm
    return symmetry_center + radial * direction + axial * axis


def lower_user_design(design: UserDesignSpec) -> LoweredUserDesign:
    """Lower public hard-coordinate constraints to the shared Assembly IR.

    Cartesian fixed motifs and per-atom cylindrical coordinates are distinct
    runtime contracts.  The latter retain selected radius/azimuth/axial
    coordinates while allowing every unselected degree of freedom to diffuse;
    they are never approximated by a rigid-component mobility declaration.
    """

    if not design.generation and not design.connections:
        raise ValueError("Executable user designs require generated regions")
    if design.components:
        connected_selectors = {
            _graph_endpoint_segment(design, endpoint)
            for connection in design.connections
            for endpoint in (
                connection.from_endpoint,
                connection.to_endpoint,
            )
        }
        unattached = [
            selector
            for component in design.components.values()
            for selector in component.selectors
            if _one_segment(
                selector,
                label="assembly component selector",
            )
            not in connected_selectors
        ]
        if unattached:
            raise NotImplementedError(
                "The current RFD3 graph backend requires every component "
                "selector to participate in a generated connection; "
                "unattached selectors: "
                + ", ".join(unattached)
            )
    plan = compile_constraint_plan(design)
    sampling_plan = compile_sampling_plan(design)
    plan.require_backend_support({"fixed_xyz", "cylindrical"})
    for operator in plan.operators:
        if (
            operator.operator == "fixed_xyz"
            and operator.atoms != AtomScope.ALL
        ):
            raise ValueError(
                "The first fixed_xyz backend requires atoms=all"
            )
    bound = bind_constraint_plan(design, plan)

    symmetry_id = (
        design.symmetry
        if isinstance(design.symmetry, str)
        else design.symmetry.id
    )
    if any(
        operator.operator == "cylindrical"
        for operator in plan.operators
    ) and symmetry_id[0] not in {"C", "D"}:
        raise NotImplementedError(
            "Public cylindrical projection currently requires a Cn or Dn "
            "symmetry with one invariant principal axis"
        )
    declared_registry = transform_registry_for_design(design)
    named_relations = [
        (f"interface {interface.id!r}", interface.copy_relation.transform)
        for interface in design.interfaces
        if interface.copy_relation.transform is not None
    ]
    named_relations.extend(
        (
            f"connection {connection.id!r}",
            connection.copy_relation.transform,
        )
        for connection in design.connections
        if connection.copy_relation.transform is not None
    )
    for owner, transform_id in named_relations:
        assert transform_id is not None
        try:
            declared_registry.transform(transform_id)
        except KeyError as error:
            raise ValueError(
                f"{owner} uses an invalid symmetry neighbour: {error}"
            ) from error
    if symmetry_id in {"T", "O", "I"}:
        invalid_offsets = [
            clause.orbit_offset
            for clause in design.generation
            if isinstance(clause, BetweenGeneration)
            and clause.orbit_offset != 0
        ]
        invalid_offsets.extend(
            connection.copy_relation.orbit_offset
            for connection in design.connections
            if connection.copy_relation.orbit_offset not in (None, 0)
        )
        if invalid_offsets:
            raise ValueError(
                "T/O/I generation does not define cyclic orbit offsets; "
                "use a named group transform in the AssemblySpecification"
            )

    generation_segments: list[SelectorSegment] = []
    for clause in design.generation:
        if isinstance(clause, TerminalGeneration):
            generation_segments.append(
                _one_segment(clause.anchor, label="terminal anchor")
            )
        else:
            generation_segments.extend(
                (
                    _one_segment(
                        clause.from_selector,
                        label="between from_selector",
                    ),
                    _one_segment(
                        clause.to_selector,
                        label="between to_selector",
                    ),
                )
            )
    for component in design.components.values():
        generation_segments.extend(
            _one_segment(
                selector,
                label="assembly component selector",
            )
            for selector in component.selectors
        )
    for connection in design.connections:
        generation_segments.extend(
            (
                _graph_endpoint_segment(
                    design,
                    connection.from_endpoint,
                ),
                _graph_endpoint_segment(
                    design,
                    connection.to_endpoint,
                ),
            )
        )
    ordered_segments = tuple(dict.fromkeys(generation_segments))
    source_atoms = read_structure_atoms(
        design.input,
        mmcif_identifier_namespace="label",
    )
    _validate_component_finite_actions(
        design,
        source_atoms,
        declared_registry,
    )
    fixed_operators = tuple(
        operator
        for operator in bound.operators
        if operator.plan.operator == "fixed_xyz"
    )
    cylindrical_operators = tuple(
        operator
        for operator in bound.operators
        if operator.plan.operator == "cylindrical"
    )
    structural_operators = fixed_operators + cylindrical_operators
    fixed_atom_ids = frozenset().union(
        *(operator.atom_ids for operator in fixed_operators)
    )
    generation_atom_ids = {
        segment: _select_segment_atoms(
            source_atoms,
            segment,
            AtomScope.ALL,
        )
        for segment in ordered_segments
    }
    operator_by_segment: dict[
        SelectorSegment,
        BoundConstraintOperator,
    ] = {}
    for segment in ordered_segments:
        covering = [
            operator
            for operator in structural_operators
            if (
                generation_atom_ids[segment].issubset(operator.atom_ids)
                if operator.plan.operator == "fixed_xyz"
                else any(
                    _segment_contains(declared_segment, segment)
                    for declared_segment in operator.segments
                )
            )
        ]
        if len(covering) != 1:
            raise ValueError(
                "Every generated-region endpoint requires explicit "
                "fixed_xyz or cylindrical coverage by exactly one "
                "declaration; "
                f"{segment.public_expression!r} has {len(covering)}"
            )
        operator_by_segment[segment] = covering[0]

    if not structural_operators:
        raise ValueError(
            "Current native adapter requires explicit fixed_xyz or "
            "cylindrical constraints for every generated-region endpoint"
        )
    used_atom_ids = frozenset().union(*generation_atom_ids.values())
    unused_fixed_atom_ids = fixed_atom_ids - used_atom_ids
    if unused_fixed_atom_ids:
        raise NotImplementedError(
            "Fixed selections not attached to generated regions are not yet "
            f"supported: {len(unused_fixed_atom_ids)} extra atoms"
        )
    unused_cylindrical_atom_ids = frozenset().union(
        *(operator.atom_ids for operator in cylindrical_operators)
    ) - used_atom_ids
    if unused_cylindrical_atom_ids:
        raise NotImplementedError(
            "Cylindrical selections not attached to generated regions are "
            "not yet supported: "
            f"{len(unused_cylindrical_atom_ids)} extra atoms"
        )

    segment_ids = {
        segment: f"motif_{index:03d}"
        for index, segment in enumerate(ordered_segments, start=1)
    }
    fragments = {
        fragment_id: {
            "source": str(design.input),
            "selection": segment.assembly_expression,
            "entity_type": "protein",
            "role": "functional_motif",
            # ``none`` is a compiler-owned adapter sentinel.  It keeps the
            # indexed input residue in the contig while assigning an empty
            # Cartesian fixed mask; cylindrical reference coordinates travel
            # through a separate runtime feature contract below.
            "fixed_atoms": (
                "all"
                if operator_by_segment[segment].plan.operator == "fixed_xyz"
                else "none"
            ),
        }
        for segment, fragment_id in segment_ids.items()
    }
    component_by_segment: dict[SelectorSegment, str] = {}
    for segment in ordered_segments:
        operator = operator_by_segment[segment].plan
        component_by_segment[segment] = (
            operator.coupling_group or operator.id
        )

    component_members: dict[str, list[str]] = {}
    for segment, fragment_id in segment_ids.items():
        component_members.setdefault(
            component_by_segment[segment], []
        ).append(fragment_id)
    component_ids = tuple(component_members)
    motion_group_ids = {
        component_id: f"fixed_component_{index:03d}"
        for index, component_id in enumerate(component_ids, start=1)
    }
    component_pose: dict[str, dict[str, object]] = {}
    for operator in fixed_operators:
        component_id = operator.plan.coupling_group or operator.plan.id
        pose = dict(operator.plan.parameters["pose"])
        previous = component_pose.setdefault(component_id, pose)
        if previous != pose:
            raise ValueError(
                "All fixed_xyz declarations in coupling_group "
                f"{component_id!r} must declare the same pose settings"
            )
    component_mobility: dict[str, dict[str, object]] = {}
    for component_id, pose in component_pose.items():
        if pose["mode"] != "bounded_mobile":
            continue
        component_mobility[motion_group_ids[component_id]] = {
            "mode": "orbit_rigid",
            "bounds": {
                "max_translation": pose["max_translation"],
                "max_rotation_deg": pose["max_rotation_deg"],
            },
            "subspace": pose["subspace"] or "bounded_se3",
            "proposal": pose["proposal"],
            "schedule": {
                "start_fraction": pose["start_fraction"],
                "end_fraction": pose["end_fraction"],
                "response": pose["response"],
                "max_step_translation": pose[
                    "max_step_translation"
                ],
                "max_step_rotation_deg": pose[
                    "max_step_rotation_deg"
                ],
            },
        }
    generated_segments: dict[str, object] = {}
    for index, clause in enumerate(design.generation, start=1):
        segment_id = f"generated_{index:03d}"
        if isinstance(clause, TerminalGeneration):
            anchor = _one_segment(clause.anchor, label="terminal anchor")
            generated_segments[segment_id] = {
                "anchor": {
                    "fragment": segment_ids[anchor],
                    "terminus": clause.terminus.upper(),
                },
                "length": _length_payload(clause.length),
                "tie_group": clause.tie_group,
            }
        else:
            left = _one_segment(
                clause.from_selector,
                label="between from_selector",
            )
            right = _one_segment(
                clause.to_selector,
                label="between to_selector",
            )
            generated_segments[segment_id] = {
                "from_endpoint": {
                    "fragment": segment_ids[left],
                    "terminus": "C",
                },
                "to_endpoint": {
                    "fragment": segment_ids[right],
                    "terminus": "N",
                },
                "length": _length_payload(clause.length),
                "tie_group": clause.tie_group,
                "copy_relation": {"orbit_offset": clause.orbit_offset},
            }
    for connection in design.connections:
        left = _graph_endpoint_segment(
            design,
            connection.from_endpoint,
        )
        right = _graph_endpoint_segment(
            design,
            connection.to_endpoint,
        )
        generated_segments[f"connection__{connection.id}"] = {
            "from_endpoint": {
                "fragment": segment_ids[left],
                "terminus": "C",
            },
            "to_endpoint": {
                "fragment": segment_ids[right],
                "terminus": "N",
            },
            "length": _length_payload(connection.length),
            "tie_group": connection.tie_group,
            "copy_relation": connection.copy_relation.model_dump(
                mode="json",
                exclude_none=True,
            ),
        }

    # Both public authoring modes converge here.  Expert declarations may
    # provide an explicit initial pose; ordinary contig designs may receive a
    # deterministic pose from the automatic planner below.
    initialization_seed, initialization = assembly_initialization_payload(
        sampling_plan
    )
    if initialization:
        if sampling_plan.initial_pose is not None:
            if len(motion_group_ids) != 1:
                raise ValueError(
                    "One public initial_pose cannot position multiple "
                    "independent fixed coupling groups; use "
                    "sampling.initial_poses keyed by coupling_group"
                )
            initialization = {
                next(iter(motion_group_ids.values())): next(
                    iter(initialization.values())
                )
            }
        else:
            unknown_components = sorted(
                set(initialization) - set(motion_group_ids)
            )
            if unknown_components:
                raise ValueError(
                    "sampling.initial_poses references unknown fixed "
                    "coupling_group(s): "
                    + ", ".join(unknown_components)
                    + "; available groups are "
                    + ", ".join(motion_group_ids)
                )
            initialization = {
                motion_group_ids[component_id]: payload
                for component_id, payload in initialization.items()
            }

    # A usable supplied pose always wins.  If a simple ASU motif lies on a
    # symmetry stabilizer, however, it does not yet define a non-overlapping
    # complete assembly.  Establish one deterministic compile-time pose so
    # ordinary users do not have to invent a radius.  ``locked`` freezes that
    # resolved pose during diffusion; ``optimize_components`` may subsequently
    # adapt it inside the declared mobility bounds.
    if (
        design.user_mode == "simple"
        and len(motion_group_ids) == 1
        and not initialization
    ):
        component_id = next(iter(motion_group_ids))
        component_atom_ids = frozenset().union(
            *(
                generation_atom_ids[segment]
                for segment in ordered_segments
                if component_by_segment[segment] == component_id
            )
        )
        component_atoms = tuple(
            atom
            for atom in source_atoms
            if _atom_identity(atom) in component_atom_ids
        )
        automatic_initialization, _ = _automatic_simple_component_plan(
            design,
            declared_registry,
            component_atoms,
        )
        if automatic_initialization is not None:
            initialization[motion_group_ids[component_id]] = (
                automatic_initialization
            )

    ports: dict[str, object] = {}
    public_port_ids: dict[str, str] = {}
    for port_id, port in design.ports.items():
        internal_port_id = f"port__{port_id}"
        public_port_ids[port_id] = internal_port_id
        ports[internal_port_id] = {
            "group": motion_group_ids[port.component],
            "fragments": [
                segment_ids[
                    _one_segment(
                        selector,
                        label=f"assembly port {port_id!r} selector",
                    )
                ]
                for selector in port.selectors
            ],
            "atoms": "heavy",
            "frame": {"method": "reference_interface_pca"},
        }
    interfaces: dict[str, object] = {}
    for interface in design.interfaces:
        if design.ports:
            node_ports = {
                node: public_port_ids[node] for node in interface.between
            }
        else:
            # Backward-compatible component-as-interface shorthand.  New
            # cage designs should declare reusable named ports explicitly.
            node_ports = {}
            for participant_index, node in enumerate(
                interface.between,
                start=1,
            ):
                port_id = (
                    f"interface__{interface.id}__participant_"
                    f"{participant_index:02d}"
                )
                node_ports[node] = port_id
                ports[port_id] = {
                    "group": motion_group_ids[node],
                    "fragments": component_members[node],
                    "atoms": "heavy",
                    "frame": {"method": "reference_interface_pca"},
                }
        execution_pairs = interface.execution_pairs
        for member_index, (left_node, right_node) in enumerate(
            execution_pairs,
            start=1,
        ):
            member_id = (
                interface.id
                if len(execution_pairs) == 1
                else f"{interface.id}__member_{member_index:02d}"
            )
            left_port = node_ports[left_node]
            right_port = node_ports[right_node]
            if len(execution_pairs) > 1:
                # The current reference-frame compiler requires one partner
                # per runtime port.  A public cooperative hyperedge may use
                # one participant in several contact-tree members, so give
                # each binary compatibility member its own alias onto the
                # same joint-rigid atoms.  This is an execution detail only:
                # public identity, multiplicity and audit remain attached to
                # the one hyperedge.
                left_alias = (
                    f"interface__{interface.id}__member_"
                    f"{member_index:02d}__left"
                )
                right_alias = (
                    f"interface__{interface.id}__member_"
                    f"{member_index:02d}__right"
                )
                ports[left_alias] = dict(ports[left_port])
                ports[right_alias] = dict(ports[right_port])
                left_port = left_alias
                right_port = right_alias
            interfaces[member_id] = {
                "left_port": left_port,
                "right_port": right_port,
                "hyperedge_id": (
                    interface.hyperedge_id
                    or (
                        interface.id
                        if len(execution_pairs) > 1
                        else None
                    )
                ),
                "copy_relation": interface.copy_relation.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                "required": interface.required,
                "satisfaction_stage": (
                    "input"
                    if interface.relation.mode == "preserve_input"
                    else "output"
                ),
                "target_geometry": _graph_interface_geometry(
                    interface.relation
                ),
            }

    # Simple motif-scaffolding tasks do not need a hand-written assembly
    # graph.  Their contig topology already states the design intent:
    # terminal generation around a fixed central motif must create packing
    # against a symmetry neighbour.  Conversely, ``between`` generation
    # joins supplied fixed fragments and therefore does not invent a new
    # interface objective.  Explicit graph interfaces remain available for
    # advanced multi-face cages and take precedence over this inference.
    infer_terminal_interfaces = (
        design.task != UserDesignTask.PRESERVE_SUPPLIED_GEOMETRY
    )
    if not design.interfaces and infer_terminal_interfaces:
        terminal_components: list[str] = []
        for clause in design.generation:
            if not isinstance(clause, TerminalGeneration):
                continue
            anchor = _one_segment(clause.anchor, label="terminal anchor")
            component_id = component_by_segment[anchor]
            if component_id not in terminal_components:
                terminal_components.append(component_id)
        if design.user_mode == "simple" and len(terminal_components) > 1:
            raise ValueError(
                "Simple mode found terminal generation on multiple "
                "independent fixed components. Mosaic will not silently "
                "place several motif orbits on top of one another; use "
                "expert components/ports/interfaces/connections so their "
                "relationships are explicit"
            )
        for component_index, component_id in enumerate(
            terminal_components,
            start=1,
        ):
            motion_group_id = motion_group_ids[component_id]
            component_atom_ids = frozenset().union(
                *(
                    generation_atom_ids[segment]
                    for segment in ordered_segments
                    if component_by_segment[segment] == component_id
                )
            )
            component_atoms = tuple(
                atom
                for atom in source_atoms
                if _atom_identity(atom) in component_atom_ids
            )
            if motion_group_id in initialization:
                master_center = _initialization_center(
                    design,
                    initialization[motion_group_id],
                )
            elif (
                design.fixed_arrangement
                == FixedArrangementPolicy.LOCKED
            ):
                # Preserve the supplied complete-orbit frame exactly.  The
                # generated scaffold may still receive interface guidance,
                # but the fixed target and all inter-copy distances/angles
                # are immutable.
                heavy_coordinates = np.asarray(
                    [
                        atom.coordinate
                        for atom in component_atoms
                        if atom.element.upper() != "H"
                    ],
                    dtype=np.float64,
                )
                if heavy_coordinates.size == 0:
                    heavy_coordinates = np.asarray(
                        [atom.coordinate for atom in component_atoms],
                        dtype=np.float64,
                    )
                master_center = heavy_coordinates.mean(axis=0)
            else:
                automatic_initialization, master_center = (
                    _automatic_simple_component_plan(
                        design,
                        declared_registry,
                        component_atoms,
                    )
                )
                if automatic_initialization is not None:
                    initialization[motion_group_id] = automatic_initialization

            port_id = f"auto_interface_port_{component_index:03d}"
            interface_id = f"auto_generated_interface_{component_index:03d}"
            contact_frame = np.eye(4, dtype=np.float64)
            contact_frame[:3, 3] = master_center
            ports[port_id] = {
                "group": motion_group_id,
                "fragments": component_members[component_id],
                "atoms": "heavy",
                # Output contact guidance is orientation-free.  A
                # translation-only frame avoids imposing a three-atom PCA
                # requirement on tiny motifs used by the simple frontend.
                "frame": {
                    "method": "precomputed",
                    "transform": contact_frame.tolist(),
                },
            }
            copy_relation = {
                "transform": _nearest_symmetry_neighbour(
                    declared_registry,
                    master_center,
                )
            }
            interfaces[interface_id] = {
                "left_port": port_id,
                "right_port": port_id,
                "copy_relation": copy_relation,
                "required": True,
                "satisfaction_stage": "output",
                "target_geometry": {
                    "mode": "geometric_constraints",
                    "contacts": {
                        "min_heavy_atom_contacts": 0,
                        # Heavy-atom output evidence must represent direct
                        # packing, not the much broader CA neighbourhood used
                        # by the differentiable runtime proxy.
                        "cutoff": 4.5,
                    },
                    "coverage": {"mode": "auto"},
                },
            }

    explicit_component_actions = {
        component_id: component.finite_orbit_action
        for component_id, component in design.components.items()
        if component.finite_orbit_action is not None
    }
    if explicit_component_actions and design.finite_orbit_action is not None:
        raise ValueError(
            "A design cannot combine one global finite_orbit_action with "
            "component-level finite_orbit_action declarations"
        )
    if explicit_component_actions:
        symmetry_orbits = {
            f"motif_orbit__{component_id}": {
                "transform_set": "declared",
                "master_groups": [motion_group_ids[component_id]],
                "component_mobility": (
                    {
                        motion_group_ids[component_id]: component_mobility[
                            motion_group_ids[component_id]
                        ]
                    }
                    if motion_group_ids[component_id] in component_mobility
                    else {}
                ),
                "finite_action": (
                    design.components[
                        component_id
                    ].finite_orbit_action.model_dump(mode="json")
                    if design.components[
                        component_id
                    ].finite_orbit_action is not None
                    else None
                ),
            }
            for component_id in component_ids
        }
    else:
        # Preserve the historical single-orbit layout byte-for-byte for all
        # existing designs.  Multiple component orbits are created only when
        # the public graph explicitly declares component finite actions.
        symmetry_orbits = {
            "motif_orbit": {
                "transform_set": "declared",
                "master_groups": list(motion_group_ids.values()),
                "component_mobility": component_mobility,
                "finite_action": (
                    design.finite_orbit_action.model_dump(mode="json")
                    if design.finite_orbit_action is not None
                    else None
                ),
            }
        }

    specification = AssemblySpecification.model_validate(
        {
            "schema_version": 2,
            "mode": "constraint_assembly",
            # Component-level finite actions describe distinct physical
            # quotient orbits (for example the C2 and C3 building blocks in
            # one tetrahedral cage).  Their supplied interface is the only
            # exact runtime object that spans both orbits, so grouping all
            # fixed atoms by motion group would incorrectly collapse the two
            # component actions into one motif orbit.
            "constraint_group_strategy": (
                "interface_edges"
                if explicit_component_actions and interfaces
                else "motion_groups"
            ),
            "random_seed": initialization_seed,
            "fragments": fragments,
            "motion_groups": {
                motion_group_ids[component_id]: {
                    "members": members,
                    "mode": "fixed",
                }
                for component_id, members in component_members.items()
            },
            "symmetry": {
                "transform_sets": {"declared": _symmetry_payload(design)},
                "orbits": symmetry_orbits,
            },
            "ports": ports,
            "interfaces": interfaces,
            "generated_segments": generated_segments,
            "initialization": initialization,
            "objectives": _assembly_shape_objectives(design),
        }
    )
    interface_usage = _resolve_interface_usage(design, specification)
    symmetry_payload = _symmetry_payload(design)
    cylindrical_constraints: list[dict[str, object]] = []
    for operator in cylindrical_operators:
        members: list[dict[str, object]] = []
        for segment, fragment_id in segment_ids.items():
            if operator_by_segment[segment] is not operator:
                continue
            selected_atoms = sorted(
                operator.atom_ids.intersection(
                    generation_atom_ids[segment]
                )
            )
            if not selected_atoms:
                raise ValueError(
                    f"Cylindrical constraint {operator.plan.id!r} matched "
                    f"no atoms in fragment {fragment_id!r}"
                )
            members.append(
                {
                    "fragment_id": fragment_id,
                    "source_atoms": [
                        {
                            "chain_id": atom.chain_id,
                            "residue_number": atom.residue_number,
                            "insertion_code": atom.insertion_code,
                            "atom_name": atom.atom_name,
                        }
                        for atom in selected_atoms
                    ],
                }
            )
        if not members:
            raise ValueError(
                f"Cylindrical constraint {operator.plan.id!r} is not "
                "attached to a materialized input fragment"
            )
        cylindrical_constraints.append(
            {
                "constraint_id": operator.plan.id,
                "orbit_scope": operator.plan.orbit_scope.value,
                "keep": list(operator.plan.controlled_dofs),
                "axis": list(symmetry_payload["axis"]),
                "members": members,
            }
        )
    return LoweredUserDesign(
        specification=specification,
        constraint_plan=plan,
        sampling_plan=sampling_plan,
        bound_constraints=bound,
        interface_usage=interface_usage,
        runtime_constraint_metadata={
            "cylindrical_constraints": cylindrical_constraints,
        },
    )


__all__ = [
    "AtomIdentity",
    "BoundConstraintOperator",
    "BoundConstraintPlan",
    "InterfaceUsageResolution",
    "LoweredUserDesign",
    "SelectorSegment",
    "bind_constraint_plan",
    "lower_user_design",
    "parse_public_selector",
]
