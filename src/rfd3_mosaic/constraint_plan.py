"""Compile public constraint clauses into topology-neutral runtime intent."""

from enum import Enum

from pydantic import Field

from rfd3_mosaic.design_preferences import compile_design_preferences
from rfd3_mosaic.schema.design import (
    AtomScope,
    BoundedMobileConstraint,
    ConstraintClause,
    ConstraintOrbitScope,
    CylindricalConstraint,
    FixedArrangementPolicy,
    FixedComponentPoseSpec,
    FixedXYZConstraint,
    UserDesignSpec,
    UserDesignTask,
)
from rfd3_mosaic.schema.specs import StrictModel


class ConstraintStage(str, Enum):
    HARD_PROJECTOR = "hard_projector"
    BOUNDED_PROJECTOR = "bounded_projector"


class ConstraintOperatorPlan(StrictModel):
    """Canonical operator consumed later by a backend binder."""

    id: str
    operator: str
    stage: ConstraintStage
    selector: str
    atoms: AtomScope
    orbit_scope: ConstraintOrbitScope
    reference_frame: str
    controlled_dofs: tuple[str, ...]
    coupling_group: str | None = None
    parameters: dict[str, object] = Field(default_factory=dict)


class ConstraintPlan(StrictModel):
    """Deterministic, backend-independent constraint execution plan."""

    schema_version: int = 1
    operators: tuple[ConstraintOperatorPlan, ...] = ()

    @property
    def required_operator_kinds(self) -> tuple[str, ...]:
        return tuple(sorted({item.operator for item in self.operators}))

    def require_backend_support(self, supported: set[str]) -> None:
        missing = sorted(set(self.required_operator_kinds) - supported)
        if missing:
            raise ValueError(
                "Constraint backend does not implement operators: "
                + ", ".join(missing)
            )


def _canonical_operator(constraint: ConstraintClause) -> str:
    if isinstance(constraint, FixedXYZConstraint):
        return "fixed_xyz"
    if isinstance(constraint, CylindricalConstraint):
        return "cylindrical"
    if isinstance(constraint, BoundedMobileConstraint):
        return "bounded_mobile"
    raise TypeError(f"Unsupported constraint clause: {type(constraint)!r}")


def _controlled_dofs(constraint: ConstraintClause) -> tuple[str, ...]:
    if isinstance(constraint, FixedXYZConstraint):
        return ("cartesian_xyz",)
    if isinstance(constraint, CylindricalConstraint):
        return tuple(item.value for item in constraint.keep)
    result: list[str] = []
    for field, dof in (
        ("radial", "radius"),
        ("axial", "axial"),
        ("azimuth_deg", "azimuth"),
        ("tilt_deg", "tilt"),
        ("twist_deg", "twist"),
    ):
        if getattr(constraint, field) is not None:
            result.append(dof)
    return tuple(result)


_CREATE_INTERFACE_ORBIT_POSE = FixedComponentPoseSpec(
    mode="bounded_mobile",
    # Ordinary interface design uses the symmetry axis as a physically
    # meaningful coordinate system: optimize cage radius, axial placement and
    # full rigid-body orientation while suppressing arbitrary tangential drift.
    # Expert declarations can still request unrestricted bounded_se3.
    subspace="radial_axial_rotation",
    proposal="scaffold_objectives",
    max_translation=4.0,
    max_rotation_deg=10.0,
    start_fraction=0.05,
    end_fraction=0.75,
    response=0.2,
    max_step_translation=0.25,
    max_step_rotation_deg=1.0,
)


_CREATE_INTERFACE_FREE_ORBIT_POSE = _CREATE_INTERFACE_ORBIT_POSE.model_copy(
    update={"subspace": "bounded_se3"}
)


def _parameters(
    constraint: ConstraintClause,
    *,
    task: UserDesignTask | None = None,
    fixed_arrangement: FixedArrangementPolicy = FixedArrangementPolicy.LOCKED,
    mobility_subspace: str | None = None,
) -> dict[str, object]:
    if isinstance(constraint, FixedXYZConstraint):
        pose = (
            (
                _CREATE_INTERFACE_FREE_ORBIT_POSE
                if mobility_subspace == "bounded_se3"
                else _CREATE_INTERFACE_ORBIT_POSE
            )
            if (
                task == UserDesignTask.CREATE_SYMMETRIC_INTERFACE
                and fixed_arrangement
                == FixedArrangementPolicy.OPTIMIZE_COMPONENTS
            )
            else constraint.pose
        )
        return {"pose": pose.model_dump(mode="json")}
    if isinstance(constraint, CylindricalConstraint):
        return {"axis": constraint.axis}
    return {
        key: value.model_dump(mode="json")
        for key in (
            "radial",
            "axial",
            "azimuth_deg",
            "tilt_deg",
            "twist_deg",
        )
        if (value := getattr(constraint, key)) is not None
    }


def _atom_scopes_overlap(left: AtomScope, right: AtomScope) -> bool:
    # Every currently supported scope contains CA atoms, so equal selectors
    # always have a non-empty atom intersection.  Keep this explicit rather
    # than accidentally treating CA and backbone as disjoint.
    return True


def _dofs_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    if "cartesian_xyz" in left or "cartesian_xyz" in right:
        return True
    return bool(set(left) & set(right))


def _validate_exact_selector_conflicts(
    operators: tuple[ConstraintOperatorPlan, ...],
) -> None:
    """Reject conflicts visible before structure-aware selector expansion.

    Exact text matches are handled here.  Partial residue overlap is checked
    later by the assembly binder after selectors have become atom identities.
    """

    for index, left in enumerate(operators):
        for right in operators[index + 1 :]:
            if left.selector != right.selector:
                continue
            if not _atom_scopes_overlap(left.atoms, right.atoms):
                continue
            if _dofs_overlap(left.controlled_dofs, right.controlled_dofs):
                raise ValueError(
                    "Conflicting constraints on selector "
                    f"{left.selector!r}: {left.id} ({left.operator}) and "
                    f"{right.id} ({right.operator})"
                )


def compile_constraint_plan(design: UserDesignSpec) -> ConstraintPlan:
    """Compile constraints deterministically without choosing a backend."""

    resolved_preferences = compile_design_preferences(design)

    legacy_operators = tuple(
        ConstraintOperatorPlan(
            id=f"constraint_{index:03d}",
            operator=_canonical_operator(constraint),
            stage=(
                ConstraintStage.BOUNDED_PROJECTOR
                if isinstance(constraint, BoundedMobileConstraint)
                else ConstraintStage.HARD_PROJECTOR
            ),
            selector=constraint.selector,
            atoms=constraint.atoms,
            orbit_scope=constraint.orbit_scope,
            reference_frame=(
                "symmetry_axis"
                if isinstance(
                    constraint,
                    (CylindricalConstraint, BoundedMobileConstraint),
                )
                else "input_rigid_geometry"
            ),
            controlled_dofs=_controlled_dofs(constraint),
            coupling_group=(
                constraint.coupling_group
                if isinstance(constraint, FixedXYZConstraint)
                else None
            ),
            parameters=_parameters(
                constraint,
                task=design.task,
                fixed_arrangement=design.fixed_arrangement,
                mobility_subspace=resolved_preferences.mobility_subspace,
            ),
        )
        for index, constraint in enumerate(design.constraints, start=1)
    )
    graph_operators = tuple(
        ConstraintOperatorPlan(
            id=f"component__{component_id}",
            operator="fixed_xyz",
            stage=ConstraintStage.HARD_PROJECTOR,
            selector=",".join(component.selectors),
            atoms=AtomScope.ALL,
            orbit_scope=ConstraintOrbitScope.COMPLETE_SYMMETRY_ORBIT,
            reference_frame="input_rigid_geometry",
            controlled_dofs=("cartesian_xyz",),
            coupling_group=component_id,
            parameters={"pose": component.pose.model_dump(mode="json")},
        )
        for component_id, component in design.components.items()
    )
    operators = (*legacy_operators, *graph_operators)
    _validate_exact_selector_conflicts(operators)
    return ConstraintPlan(operators=operators)
