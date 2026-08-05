"""Structure-aware lowering of the small public design declaration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from rfd3_mosaic.constraint_plan import (
    ConstraintOperatorPlan,
    ConstraintPlan,
    compile_constraint_plan,
)
from rfd3_mosaic.sampling_plan import (
    SamplingPlan,
    assembly_initialization_payload,
    compile_sampling_plan,
)
from rfd3_mosaic.schema import (
    AssemblySpecification,
    AtomScope,
    BetweenGeneration,
    TerminalGeneration,
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
class LoweredUserDesign:
    specification: AssemblySpecification
    constraint_plan: ConstraintPlan
    sampling_plan: SamplingPlan
    bound_constraints: BoundConstraintPlan


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
    if prefix not in {"C", "D"}:
        raise NotImplementedError(
            "The native public-design lowerer currently supports Cn and Dn"
        )
    order = int(symmetry_id[1:])
    payload: dict[str, object] = {
        "type": "cyclic" if prefix == "C" else "dihedral",
        "order": order,
        "axis": axis,
        "center": center,
    }
    if prefix == "D":
        payload["secondary_axis"] = secondary_axis or (1.0, 0.0, 0.0)
    return payload


def _length_payload(length: object) -> dict[str, int]:
    if isinstance(length, int):
        return {"minimum": length, "maximum": length}
    return {
        "minimum": int(length.minimum),
        "maximum": int(length.maximum),
    }


def lower_user_design(design: UserDesignSpec) -> LoweredUserDesign:
    """Lower the currently executable fixed-XYZ public subset.

    This first backend binding is intentionally narrow.  It refuses
    cylindrical/mobile operators and any unconstrained generated endpoint
    rather than inheriting the adapter's historical implicit ``ALL`` fixing.
    """

    if not design.generation:
        raise ValueError("Executable user designs require generated regions")
    plan = compile_constraint_plan(design)
    sampling_plan = compile_sampling_plan(design)
    plan.require_backend_support({"fixed_xyz"})
    for operator in plan.operators:
        if operator.atoms != AtomScope.ALL:
            raise ValueError(
                "The first fixed_xyz backend requires atoms=all"
            )
    bound = bind_constraint_plan(design, plan)

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
    ordered_segments = tuple(dict.fromkeys(generation_segments))
    source_atoms = read_structure_atoms(
        design.input,
        mmcif_identifier_namespace="label",
    )
    fixed_operators = tuple(
        operator
        for operator in bound.operators
        if operator.plan.operator == "fixed_xyz"
    )
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
    missing_fixed = [
        segment.public_expression
        for segment in ordered_segments
        if not generation_atom_ids[segment].issubset(fixed_atom_ids)
    ]
    if missing_fixed:
        raise ValueError(
            "Current native adapter requires explicit fixed_xyz constraints "
            "for every generated-region endpoint; missing "
            + ", ".join(missing_fixed)
        )
    used_atom_ids = frozenset().union(*generation_atom_ids.values())
    unused_fixed_atom_ids = fixed_atom_ids - used_atom_ids
    if unused_fixed_atom_ids:
        raise NotImplementedError(
            "Fixed selections not attached to generated regions are not yet "
            f"supported: {len(unused_fixed_atom_ids)} extra atoms"
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
            "fixed_atoms": "all",
        }
        for segment, fragment_id in segment_ids.items()
    }
    component_by_segment: dict[SelectorSegment, str] = {}
    for segment in ordered_segments:
        covering = [
            operator
            for operator in fixed_operators
            if generation_atom_ids[segment].issubset(operator.atom_ids)
        ]
        if len(covering) != 1:
            raise ValueError(
                "Each generated-region endpoint must be covered by exactly "
                "one fixed_xyz declaration; "
                f"{segment.public_expression!r} has {len(covering)}"
            )
        operator = covering[0].plan
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
            "subspace": "bounded_se3",
            "proposal": "denoiser_fit",
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

    initialization_seed, initialization = assembly_initialization_payload(
        sampling_plan
    )
    if initialization:
        if len(motion_group_ids) != 1:
            raise ValueError(
                "One public initial_pose cannot position multiple independent "
                "fixed coupling groups; declare a joint coupling_group or "
                "omit initial_pose"
            )
        initialization = {
            next(iter(motion_group_ids.values())): next(
                iter(initialization.values())
            )
        }
    specification = AssemblySpecification.model_validate(
        {
            "schema_version": 2,
            "mode": "constraint_assembly",
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
                "orbits": {
                    "motif_orbit": {
                        "transform_set": "declared",
                        "master_groups": list(motion_group_ids.values()),
                        "component_mobility": component_mobility,
                    }
                },
            },
            "generated_segments": generated_segments,
            "initialization": initialization,
        }
    )
    return LoweredUserDesign(
        specification=specification,
        constraint_plan=plan,
        sampling_plan=sampling_plan,
        bound_constraints=bound,
    )


__all__ = [
    "AtomIdentity",
    "BoundConstraintOperator",
    "BoundConstraintPlan",
    "LoweredUserDesign",
    "SelectorSegment",
    "bind_constraint_plan",
    "lower_user_design",
    "parse_public_selector",
]
