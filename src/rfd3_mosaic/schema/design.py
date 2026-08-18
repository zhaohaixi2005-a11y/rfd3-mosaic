"""Small, topology-neutral user design language.

This module deliberately does not expose the complete
``AssemblySpecification``.  It captures the few concepts a user should need
to declare: a structure, a symmetry group, generated regions, and optional
constraint operators.  Lowering into the full assembly IR is a separate,
fail-closed compiler step.
"""

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import AliasChoices, Field, field_validator, model_validator

from rfd3_mosaic.schema.specs import (
    CopyRelationSpec,
    FiniteOrbitActionSpec,
    Identifier,
    LinkLengthSpec,
    StrictModel,
)

Selector = Annotated[str, Field(min_length=1)]
PositiveLength = Annotated[int, Field(ge=1)]
RequestedLength = PositiveLength | LinkLengthSpec


class AtomScope(str, Enum):
    ALL = "all"
    BACKBONE = "backbone"
    CA = "ca"


class ConstraintOrbitScope(str, Enum):
    """Whether a declaration addresses one selection or its complete orbit."""

    SELECTION = "selection"
    COMPLETE_SYMMETRY_ORBIT = "complete_symmetry_orbit"


class UserDesignTask(str, Enum):
    """Optional ordinary-user workflow with explicit physical semantics.

    Omitting ``task`` preserves the legacy/custom declaration exactly.  The
    two named tasks are conveniences only: both are compiled through the
    normal constraint plan and ``AssemblySpecification`` rather than through
    a separate sampler backend. Resolver-emitted expert graphs retain
    ``preserve_supplied_geometry`` as an explicit provenance and safety
    contract for complete user-supplied interface hyperedges.
    """

    PRESERVE_SUPPLIED_GEOMETRY = "preserve_supplied_geometry"
    CREATE_SYMMETRIC_INTERFACE = "create_symmetric_interface"


class FixedArrangementPolicy(str, Enum):
    """Whether exact fixed components retain their assembly placement.

    Interface generation and fixed-component mobility are orthogonal.  A
    locked arrangement still receives generated-atom packing guidance; only
    ``optimize_components`` permits the exact rigid components themselves to
    translate or rotate.
    """

    LOCKED = "locked"
    OPTIMIZE_COMPONENTS = "optimize_components"


class PackingPreference(str, Enum):
    LOOSE = "loose"
    BALANCED = "balanced"
    TIGHT = "tight"


class CavityPreference(str, Enum):
    COMPACT = "compact"
    AUTO = "auto"
    OPEN = "open"


class DiversityPreference(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InterfaceAreaPreference(str, Enum):
    SMALL = "small"
    AUTO = "auto"
    LARGE = "large"


class ComponentMotionPreference(str, Enum):
    LOCKED = "locked"
    GUIDED = "guided"
    FREE = "free"


class UserDesignPreferences(StrictModel):
    """Small, physical ordinary-user controls compiled into safe presets.

    These are not raw loss weights.  Exact motif geometry, symmetry, chain
    continuity and clash rejection remain hard contracts regardless of the
    selected preference.
    """

    packing: PackingPreference = PackingPreference.BALANCED
    cavity: CavityPreference = CavityPreference.AUTO
    diversity: DiversityPreference = DiversityPreference.MEDIUM
    interface_area: InterfaceAreaPreference = InterfaceAreaPreference.AUTO
    component_motion: ComponentMotionPreference | None = None


class ExpertPackingGuidanceSpec(StrictModel):
    """Optional raw controls for expert assembly-graph authors only."""

    weight: Annotated[float, Field(ge=0.0)] | None = None
    coverage_weight: Annotated[float, Field(ge=0.0)] | None = None
    continuity_weight: Annotated[float, Field(ge=0.0)] | None = None
    orientation_weight: Annotated[float, Field(ge=0.0)] | None = None
    shape_weight: Annotated[float, Field(ge=0.0)] | None = None
    backbone_weight: Annotated[float, Field(ge=0.1)] | None = None
    interface_balance_weight: Annotated[float, Field(ge=0.0)] | None = None
    patch_exclusivity_weight: Annotated[float, Field(ge=1.0)] | None = None
    clash_weight: Annotated[float, Field(ge=8.0)] | None = None
    distance_weight: Annotated[float, Field(ge=0.0)] | None = None
    pairs_per_edge: Annotated[int, Field(ge=1)] | None = None
    maximum_token_step: Annotated[float, Field(gt=0.0)] | None = None
    start_fraction: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    end_fraction: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    @model_validator(mode="after")
    def validate_schedule(self) -> "ExpertPackingGuidanceSpec":
        if (
            self.start_fraction is not None
            and self.end_fraction is not None
            and self.start_fraction >= self.end_fraction
        ):
            raise ValueError(
                "expert guidance requires start_fraction < end_fraction"
            )
        return self


class CylindricalDegreeOfFreedom(str, Enum):
    RADIUS = "radius"
    AZIMUTH = "azimuth"
    AXIAL = "axial"


class FixedComponentPoseSpec(StrictModel):
    """Choose whether one rigid geometry component may move as a whole."""

    mode: Literal["fixed", "bounded_mobile"] = "fixed"
    subspace: Literal[
        "bounded_se3",
        "radial",
        "radial_axial",
        "tilt_only",
        "radial_rotation",
        "radial_axial_rotation",
    ] | None = None
    proposal: Literal["denoiser_fit", "scaffold_objectives"] = (
        "denoiser_fit"
    )
    max_translation: Annotated[float, Field(gt=0.0)] | None = None
    max_rotation_deg: Annotated[
        float,
        Field(gt=0.0, le=180.0),
    ] | None = None
    start_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.05
    end_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.75
    response: Annotated[float, Field(gt=0.0, le=1.0)] = 0.2
    max_step_translation: Annotated[float, Field(gt=0.0)] = 0.25
    max_step_rotation_deg: Annotated[
        float,
        Field(gt=0.0, le=180.0),
    ] = 1.0

    @model_validator(mode="after")
    def validate_pose_mode(self) -> "FixedComponentPoseSpec":
        if self.start_fraction >= self.end_fraction:
            raise ValueError(
                "fixed component pose requires start_fraction < "
                "end_fraction"
            )
        bounds = (self.max_translation, self.max_rotation_deg)
        if self.mode == "fixed" and (
            any(value is not None for value in bounds)
            or self.subspace is not None
        ):
            raise ValueError(
                "pose.mode=fixed cannot define a mobility subspace or "
                "translation/rotation bounds"
            )
        if self.mode == "fixed" and self.proposal != "denoiser_fit":
            raise ValueError(
                "pose.proposal is only meaningful when "
                "pose.mode=bounded_mobile"
            )
        if self.mode == "bounded_mobile":
            subspace = self.subspace or "bounded_se3"
            if subspace != "tilt_only" and self.max_translation is None:
                raise ValueError(
                    "pose.mode=bounded_mobile requires max_translation"
                )
            if subspace == "tilt_only":
                if self.max_translation is not None:
                    raise ValueError(
                        "tilt_only mobility cannot define max_translation"
                    )
                if self.max_rotation_deg is None:
                    raise ValueError(
                        "tilt_only mobility requires max_rotation_deg"
                    )
                if self.proposal != "scaffold_objectives":
                    raise ValueError(
                        "tilt_only mobility requires "
                        "proposal=scaffold_objectives"
                    )
            if subspace == "bounded_se3" and self.max_rotation_deg is None:
                raise ValueError(
                    "bounded_se3 mobility requires max_rotation_deg"
                )
            radial_subspaces = {
                "radial",
                "radial_axial",
                "radial_rotation",
                "radial_axial_rotation",
            }
            if subspace in radial_subspaces:
                if (
                    subspace in {"radial", "radial_axial"}
                    and self.max_rotation_deg is not None
                ):
                    raise ValueError(
                        f"{subspace} mobility cannot define "
                        "max_rotation_deg"
                    )
                if (
                    subspace
                    in {"radial_rotation", "radial_axial_rotation"}
                    and self.max_rotation_deg is None
                ):
                    raise ValueError(
                        f"{subspace} mobility requires max_rotation_deg"
                    )
                if self.proposal != "scaffold_objectives":
                    raise ValueError(
                        f"{subspace} mobility currently requires "
                        "proposal=scaffold_objectives"
                    )
        return self


class FixedXYZConstraint(StrictModel):
    """Preserve one selected rigid geometry component during diffusion.

    Selections that share ``coupling_group`` belong to one joint component:
    all of their atoms must be superposable with one common rigid transform.
    An omitted group keeps this declaration independent from other fixed
    declarations.  Absolute laboratory-frame coordinates are deliberately
    not part of the public contract.
    """

    kind: Literal["fixed_xyz", "full_xyz_fixed"] = "fixed_xyz"
    selector: Selector
    atoms: AtomScope = AtomScope.ALL
    orbit_scope: ConstraintOrbitScope = (
        ConstraintOrbitScope.COMPLETE_SYMMETRY_ORBIT
    )
    coupling_group: Identifier | None = None
    pose: FixedComponentPoseSpec = Field(
        default_factory=FixedComponentPoseSpec
    )


class CylindricalConstraint(StrictModel):
    """Lock selected cylindrical coordinates around the symmetry axis."""

    kind: Literal["cylindrical", "ca_cylindrical_fixed"] = "cylindrical"
    selector: Selector
    atoms: AtomScope = AtomScope.CA
    axis: Literal["symmetry"] = "symmetry"
    keep: tuple[CylindricalDegreeOfFreedom, ...]
    orbit_scope: ConstraintOrbitScope = (
        ConstraintOrbitScope.COMPLETE_SYMMETRY_ORBIT
    )

    @field_validator("keep")
    @classmethod
    def validate_keep(
        cls,
        value: tuple[CylindricalDegreeOfFreedom, ...],
    ) -> tuple[CylindricalDegreeOfFreedom, ...]:
        if not value:
            raise ValueError("cylindrical.keep must contain at least one DOF")
        if len(value) != len(set(value)):
            raise ValueError("cylindrical.keep cannot contain duplicate DOFs")
        return value


class NumericRange(StrictModel):
    minimum: float
    maximum: float

    @model_validator(mode="after")
    def validate_order(self) -> "NumericRange":
        if self.minimum > self.maximum:
            raise ValueError("range minimum cannot exceed maximum")
        return self


class UserAssemblyShapeSpec(StrictModel):
    """Optional physical size contract for CPU pose restoration and audit."""

    diameter_angstrom: NumericRange | None = None
    cavity_diameter_angstrom: NumericRange | None = None

    @model_validator(mode="after")
    def require_positive_shape_bounds(self) -> "UserAssemblyShapeSpec":
        if (
            self.diameter_angstrom is None
            and self.cavity_diameter_angstrom is None
        ):
            raise ValueError(
                "assembly_shape requires diameter and/or cavity diameter"
            )
        for name, bounds in (
            ("diameter_angstrom", self.diameter_angstrom),
            ("cavity_diameter_angstrom", self.cavity_diameter_angstrom),
        ):
            if bounds is not None and bounds.minimum <= 0.0:
                raise ValueError(f"assembly_shape.{name} must be positive")
        return self


class BoundedMobileConstraint(StrictModel):
    """Allow one selected rigid group to move inside explicit pose bounds.

    Omitted degrees of freedom remain free.  This is a generic bounded-pose
    operator; ``bounded_mobile_interface`` is accepted as a compatibility
    spelling, not as a separate topology or sampler.
    """

    kind: Literal["bounded_mobile", "bounded_mobile_interface"] = (
        "bounded_mobile"
    )
    selector: Selector
    atoms: AtomScope = AtomScope.ALL
    radial: NumericRange | None = None
    axial: NumericRange | None = None
    azimuth_deg: NumericRange | None = None
    tilt_deg: NumericRange | None = None
    twist_deg: NumericRange | None = None
    orbit_scope: ConstraintOrbitScope = (
        ConstraintOrbitScope.COMPLETE_SYMMETRY_ORBIT
    )

    @model_validator(mode="after")
    def require_a_bound(self) -> "BoundedMobileConstraint":
        bounds = (
            self.radial,
            self.axial,
            self.azimuth_deg,
            self.tilt_deg,
            self.twist_deg,
        )
        if all(value is None for value in bounds):
            raise ValueError(
                "bounded_mobile requires at least one explicit pose bound"
            )
        return self


ConstraintClause = Annotated[
    FixedXYZConstraint | CylindricalConstraint | BoundedMobileConstraint,
    Field(discriminator="kind"),
]


class TerminalGeneration(StrictModel):
    """Generate residues outward from one selected motif terminus."""

    kind: Literal["terminal"] = "terminal"
    anchor: Selector
    terminus: Literal["n", "c"]
    length: RequestedLength
    tie_group: Identifier | None = None


class BetweenGeneration(StrictModel):
    """Generate one directed connection between two motif selections."""

    kind: Literal["between"] = "between"
    from_selector: Selector
    to_selector: Selector
    length: RequestedLength
    tie_group: Identifier | None = None
    orbit_offset: int = 0

    @model_validator(mode="after")
    def reject_self_link(self) -> "BetweenGeneration":
        if self.from_selector == self.to_selector:
            raise ValueError("between generation requires two selections")
        return self


GenerationClause = Annotated[
    TerminalGeneration | BetweenGeneration,
    Field(discriminator="kind"),
]


SymmetryName = Annotated[
    str,
    Field(pattern=r"^(?:C[1-9][0-9]*|D[2-9][0-9]*|T|O|I)$"),
]


class UserSymmetrySpec(StrictModel):
    """Optional explicit frame for a finite point-group declaration."""

    id: SymmetryName
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    secondary_axis: tuple[float, float, float] | None = None

    @model_validator(mode="after")
    def validate_axes(self) -> "UserSymmetrySpec":
        if sum(value * value for value in self.axis) <= 1e-12:
            raise ValueError("symmetry axis cannot be zero")
        if self.id.startswith("C") and self.secondary_axis is not None:
            raise ValueError("secondary_axis is not valid for Cn symmetry")
        if self.secondary_axis is not None:
            if (
                sum(value * value for value in self.secondary_axis)
                <= 1e-12
            ):
                raise ValueError("secondary symmetry axis cannot be zero")
            dot = sum(
                left * right
                for left, right in zip(self.axis, self.secondary_axis)
            )
            if abs(dot) > 1e-6:
                raise ValueError(
                    "secondary symmetry axis must be perpendicular"
                )
        return self


SymmetryRequest = SymmetryName | UserSymmetrySpec


class UserFixedOrientationSpec(StrictModel):
    """One explicit intrinsic XYZ Euler orientation before diffusion."""

    method: Literal["fixed"] = "fixed"
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


class UserUniformSO3OrientationSpec(StrictModel):
    """One reproducible Haar-uniform orientation before diffusion."""

    method: Literal["uniform_so3"] = "uniform_so3"


UserInitialOrientationSpec = Annotated[
    UserFixedOrientationSpec | UserUniformSO3OrientationSpec,
    Field(discriminator="method"),
]


class UserInitialPoseSpec(StrictModel):
    """Rigid initial pose sampled once before the RFD3 timestep loop.

    This is deliberately distinct from diffusion randomness and from motif
    mobility during denoising.  The selected motif motion group is translated
    and rotated as one rigid body; no atom is moved independently.
    """

    radius: NumericRange
    axial_offset: NumericRange = Field(
        default_factory=lambda: NumericRange(minimum=0.0, maximum=0.0)
    )
    radial_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    orientation: UserInitialOrientationSpec = Field(
        default_factory=UserFixedOrientationSpec
    )
    seed: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def validate_pose(self) -> "UserInitialPoseSpec":
        if self.radius.minimum < 0.0:
            raise ValueError("initial_pose.radius cannot be negative")
        squared_norm = sum(value * value for value in self.radial_direction)
        if squared_norm <= 1e-12:
            raise ValueError("initial_pose.radial_direction cannot be zero")
        return self


class UserSamplingSpec(StrictModel):
    """Separate pre-diffusion pose choice from RFD3 diffusion sampling."""

    initial_pose: UserInitialPoseSpec | None = None
    initial_poses: dict[Identifier, UserInitialPoseSpec] = Field(
        default_factory=dict
    )
    timesteps: Annotated[int, Field(ge=2, le=200)] = 200
    designs: Annotated[int, Field(ge=1, le=10000)] = 1
    seed: Annotated[int, Field(ge=0)] = 42
    preset: Literal["exact_mosaic"] = "exact_mosaic"
    low_memory_mode: bool = True
    execution_backend: Literal[
        "explicit_all_copy",
        "local_neighbourhood",
    ] = "explicit_all_copy"
    neighbour_radius: Annotated[int, Field(ge=0)] = 1

    @model_validator(mode="after")
    def reject_ambiguous_pose_declarations(self) -> "UserSamplingSpec":
        if self.initial_pose is not None and self.initial_poses:
            raise ValueError(
                "sampling cannot define both initial_pose and "
                "initial_poses"
            )
        return self


class UserResourceSpec(StrictModel):
    profile: str | Path = "local"
    walltime: str | None = None
    memory: str | None = None
    cpus: Annotated[int, Field(ge=1)] | None = None
    partition: str | None = None


class UserOutputSpec(StrictModel):
    root: Path
    campaign: Identifier = "rfd3-mosaic"


class UserAssemblyComponentSpec(StrictModel):
    """One rigid node in the public assembly graph."""

    selectors: Annotated[tuple[Selector, ...], Field(min_length=1)]
    geometry: Literal["rigid", "joint_rigid"] = "rigid"
    finite_orbit_action: FiniteOrbitActionSpec | None = None
    pose: FixedComponentPoseSpec = Field(
        default_factory=FixedComponentPoseSpec
    )

    @model_validator(mode="after")
    def require_unique_selectors(self) -> "UserAssemblyComponentSpec":
        if len(self.selectors) != len(set(self.selectors)):
            raise ValueError("Assembly component selectors must be unique")
        if self.geometry == "rigid" and len(self.selectors) != 1:
            raise ValueError(
                "geometry=rigid requires exactly one selector; use "
                "geometry=joint_rigid when several fragments must retain "
                "one common relative pose"
            )
        return self


class UserAssemblyPortSpec(StrictModel):
    """One named interface face owned by a rigid graph component.

    Public ports deliberately select complete component fragments.  This
    keeps interface identity independent from motion identity: several ports
    may belong to one joint-rigid component while participating in different
    symmetry-neighbour relations.
    """

    component: Identifier
    selectors: Annotated[tuple[Selector, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_unique_selectors(self) -> "UserAssemblyPortSpec":
        if len(self.selectors) != len(set(self.selectors)):
            raise ValueError("Assembly port selectors must be unique")
        return self


class UserPreserveInputRelationSpec(StrictModel):
    """Preserve a contacting reference interface between two components."""

    mode: Literal["preserve_input"] = "preserve_input"
    translation_tolerance: Annotated[float, Field(gt=0.0)] = 2.0
    rotation_tolerance_deg: Annotated[
        float,
        Field(gt=0.0, le=180.0),
    ] = 10.0
    minimum_heavy_atom_contacts: Annotated[int, Field(ge=0)] = 1
    cutoff: Annotated[float, Field(gt=0.0)] = 4.5


class UserContactRelationSpec(StrictModel):
    """Ask Mosaic to design an interface between two components.

    A plain ``mode: contact`` is deliberately sufficient.  Mosaic derives a
    scale-aware contact-coverage and continuity target from the generated
    regions at runtime.  ``distance`` and ``minimum_heavy_atom_contacts`` are
    retained as optional expert overrides and for backwards compatibility;
    ordinary users should not have to guess either value.
    """

    mode: Literal["contact"] = "contact"
    distance: NumericRange | None = None
    minimum_heavy_atom_contacts: Annotated[
        int,
        Field(ge=1),
    ] | None = None
    # This is a heavy-atom contact cutoff.  Eight angstrom is useful for a
    # coarse CA neighbourhood, but is far too permissive for claiming a
    # physical all-atom interface: unrelated surface atoms can satisfy it.
    # Keep the ordinary default at a conventional direct-contact distance;
    # experts may still override it explicitly.
    cutoff: Annotated[float, Field(gt=0.0)] = 4.5


UserInterfaceRelationSpec = Annotated[
    UserPreserveInputRelationSpec | UserContactRelationSpec,
    Field(discriminator="mode"),
]


class UserInterfaceUsageSpec(StrictModel):
    """Requested physical multiplicity of one interface in the assembly.

    Ordinary users may write ``use: auto``, ``use: 12``,
    ``use: {exact: 12}``, or a minimum/maximum range.  The compiler validates
    the request against unique physical interface instances after symmetry
    expansion; it is not interpreted as an instruction to copy coordinates
    blindly.
    """

    mode: Literal["auto", "exact", "range"] = "auto"
    exact: Annotated[int, Field(ge=1)] | None = None
    minimum: Annotated[int, Field(ge=1)] | None = None
    maximum: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="before")
    @classmethod
    def parse_compact_usage(cls, value: object) -> object:
        if value is None or value == "auto":
            return {"mode": "auto"}
        if isinstance(value, int) and not isinstance(value, bool):
            return {"mode": "exact", "exact": value}
        if isinstance(value, dict):
            normalized = dict(value)
            if not normalized:
                return {"mode": "auto"}
            if "mode" not in normalized:
                normalized["mode"] = (
                    "exact" if "exact" in normalized else "range"
                )
            return normalized
        return value

    @model_validator(mode="after")
    def validate_usage(self) -> "UserInterfaceUsageSpec":
        if self.mode == "auto":
            if any(
                value is not None
                for value in (self.exact, self.minimum, self.maximum)
            ):
                raise ValueError("use=auto cannot define multiplicity bounds")
            return self
        if self.mode == "exact":
            if self.exact is None:
                raise ValueError("exact interface use requires exact")
            if self.minimum is not None or self.maximum is not None:
                raise ValueError(
                    "exact interface use cannot define minimum/maximum"
                )
            return self
        if self.exact is not None:
            raise ValueError("range interface use cannot define exact")
        if self.minimum is None and self.maximum is None:
            raise ValueError(
                "range interface use requires minimum and/or maximum"
            )
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(
                "interface use minimum cannot exceed maximum"
            )
        return self

    def accepts(self, observed: int) -> bool:
        if self.mode == "auto":
            return True
        if self.mode == "exact":
            return observed == self.exact
        return (
            (self.minimum is None or observed >= self.minimum)
            and (self.maximum is None or observed <= self.maximum)
        )

    @property
    def description(self) -> str:
        if self.mode == "auto":
            return "auto"
        if self.mode == "exact":
            return f"exactly {self.exact}"
        lower = self.minimum if self.minimum is not None else "unbounded"
        upper = self.maximum if self.maximum is not None else "unbounded"
        return f"{lower}..{upper}"


class UserAssemblyInterfaceSpec(StrictModel):
    """One complete interface relation between two or more ports.

    ``between`` is the biological interface identity.  For a supplied
    cooperative interface with three or more participants, ``contact_pairs``
    records a connected contact-supported execution tree.  Those pairs are
    compiler details: multiplicity, provenance and audit identity remain
    attached to this one interface hyperedge.
    """

    id: Identifier
    # Several pairwise runtime edges may be the contact-supported spanning
    # tree of one supplied multi-participant interface.  ``use`` then refers
    # to physical instances of this hyperedge, not to the number of member
    # pairs.
    hyperedge_id: Identifier | None = None
    between: Annotated[tuple[Identifier, ...], Field(min_length=2)]
    contact_pairs: tuple[tuple[Identifier, Identifier], ...] = ()
    relation: UserInterfaceRelationSpec = Field(
        default_factory=UserPreserveInputRelationSpec
    )
    use: UserInterfaceUsageSpec = Field(
        default_factory=UserInterfaceUsageSpec
    )
    copy_relation: CopyRelationSpec = Field(
        default_factory=lambda: CopyRelationSpec(orbit_offset=0)
    )
    required: bool = True

    @model_validator(mode="after")
    def reject_identity_self_interface(self) -> "UserAssemblyInterfaceSpec":
        if len(self.between) == 2 and self.between[0] == self.between[1]:
            relation = self.copy_relation
            if relation.orbit_offset == 0 or (
                relation.transform is not None
                and relation.transform.endswith(":e")
            ):
                raise ValueError(
                    "A self-interface must target a non-identity symmetry "
                    "copy"
                )
            if self.contact_pairs:
                raise ValueError(
                    "A symmetry-neighbour self-interface does not use "
                    "contact_pairs"
                )
            return self

        if len(set(self.between)) != len(self.between):
            raise ValueError(
                "Multi-participant interface ports must be unique"
            )
        if len(self.between) == 2:
            allowed = {frozenset(self.between)}
            declared = {frozenset(pair) for pair in self.contact_pairs}
            if declared and declared != allowed:
                raise ValueError(
                    "Binary interface contact_pairs must contain only its "
                    "declared port pair"
                )
            return self
        if self.relation.mode != "preserve_input":
            raise ValueError(
                "Multi-participant generated contact guidance is not yet "
                "supported; supplied hyperedges require preserve_input"
            )
        if not self.contact_pairs:
            raise ValueError(
                "A multi-participant supplied interface requires a "
                "contact-supported contact_pairs tree"
            )
        node_set = set(self.between)
        adjacency = {node: set() for node in self.between}
        seen_pairs: set[frozenset[str]] = set()
        for left, right in self.contact_pairs:
            if left == right:
                raise ValueError("Interface contact_pairs cannot self-pair")
            if left not in node_set or right not in node_set:
                raise ValueError(
                    "Interface contact_pairs must reference ports in between"
                )
            key = frozenset((left, right))
            if key in seen_pairs:
                raise ValueError("Interface contact_pairs must be unique")
            seen_pairs.add(key)
            adjacency[left].add(right)
            adjacency[right].add(left)
        reached = {self.between[0]}
        pending = [self.between[0]]
        while pending:
            current = pending.pop()
            for neighbour in adjacency[current]:
                if neighbour not in reached:
                    reached.add(neighbour)
                    pending.append(neighbour)
        if reached != node_set:
            raise ValueError(
                "Interface contact_pairs must connect every participant"
            )
        return self

    @property
    def execution_pairs(self) -> tuple[tuple[Identifier, Identifier], ...]:
        """Return binary runtime members without changing hyperedge identity."""

        if self.contact_pairs:
            return self.contact_pairs
        return ((self.between[0], self.between[1]),)


class UserComponentEndpointSpec(StrictModel):
    """A chain terminus belonging to one assembly-graph component."""

    component: Identifier
    terminus: Literal["n", "c"]
    selector: Selector | None = None

    @model_validator(mode="before")
    @classmethod
    def parse_compact_endpoint(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        component, separator, terminus = value.rpartition(".")
        if not separator or not component or terminus.lower() not in {"n", "c"}:
            raise ValueError(
                "Compact connection endpoints must use component.N or "
                "component.C"
            )
        return {"component": component, "terminus": terminus.lower()}


class UserAssemblyConnectionSpec(StrictModel):
    """One generated protein connection edge in the assembly graph."""

    id: Identifier
    from_endpoint: UserComponentEndpointSpec = Field(
        validation_alias=AliasChoices("from", "from_endpoint"),
        serialization_alias="from",
    )
    to_endpoint: UserComponentEndpointSpec = Field(
        validation_alias=AliasChoices("to", "to_endpoint"),
        serialization_alias="to",
    )
    length: RequestedLength
    tie_group: Identifier | None = None
    copy_relation: CopyRelationSpec = Field(
        default_factory=lambda: CopyRelationSpec(orbit_offset=0)
    )

    @model_validator(mode="after")
    def require_chain_direction(self) -> "UserAssemblyConnectionSpec":
        if self.from_endpoint.terminus != "c":
            raise ValueError("connection from_endpoint must use terminus c")
        if self.to_endpoint.terminus != "n":
            raise ValueError("connection to_endpoint must use terminus n")
        return self


class UserDesignSpec(StrictModel):
    """Stable public design declaration before assembly-IR lowering."""

    schema_version: Literal[1] = 1
    name: Identifier
    input: Path
    symmetry: SymmetryRequest
    finite_orbit_action: FiniteOrbitActionSpec | None = None
    task: UserDesignTask | None = None
    fixed_arrangement: FixedArrangementPolicy = (
        FixedArrangementPolicy.LOCKED
    )
    preferences: UserDesignPreferences = Field(
        default_factory=UserDesignPreferences
    )
    guidance: ExpertPackingGuidanceSpec | None = None
    assembly_shape: UserAssemblyShapeSpec | None = None
    generation: tuple[GenerationClause, ...] = ()
    constraints: tuple[ConstraintClause, ...] = ()
    components: dict[Identifier, UserAssemblyComponentSpec] = Field(
        default_factory=dict
    )
    ports: dict[Identifier, UserAssemblyPortSpec] = Field(
        default_factory=dict
    )
    interfaces: tuple[UserAssemblyInterfaceSpec, ...] = ()
    connections: tuple[UserAssemblyConnectionSpec, ...] = ()
    sampling: UserSamplingSpec = Field(default_factory=UserSamplingSpec)
    resources: UserResourceSpec = Field(default_factory=UserResourceSpec)
    output: UserOutputSpec | None = None

    @model_validator(mode="before")
    @classmethod
    def infer_arrangement_from_motion_preference(cls, value: object) -> object:
        """Keep the short ordinary YAML free of duplicate motion switches."""

        if (
            not isinstance(value, dict)
            or "fixed_arrangement" in value
            or value.get("task")
            != UserDesignTask.CREATE_SYMMETRIC_INTERFACE.value
        ):
            return value
        preferences = value.get("preferences")
        if not isinstance(preferences, dict):
            return value
        motion = preferences.get("component_motion")
        if motion is None:
            return value
        updated = dict(value)
        updated["fixed_arrangement"] = (
            FixedArrangementPolicy.LOCKED.value
            if motion == ComponentMotionPreference.LOCKED.value
            else FixedArrangementPolicy.OPTIMIZE_COMPONENTS.value
        )
        return updated

    @property
    def user_mode(self) -> Literal["simple", "expert"]:
        """Return the public authoring mode without creating two backends.

        The mode is inferred from what the user chose to declare.  A compact
        contig-style design is the ordinary-user surface.  Declaring an
        assembly graph opts into the expert surface.  Both lower through the
        same ``AssemblySpecification`` compiler and sampler runtime.
        """

        return (
            "expert"
            if self.components
            or self.ports
            or self.interfaces
            or self.connections
            else "simple"
        )

    @model_validator(mode="after")
    def validate_assembly_graph_mode(self) -> "UserDesignSpec":
        graph_declared = bool(
            self.components
            or self.ports
            or self.interfaces
            or self.connections
        )
        if not graph_declared:
            return self
        if not self.components:
            raise ValueError(
                "interfaces/connections require assembly components"
            )
        if self.generation or self.constraints:
            raise ValueError(
                "components/ports/interfaces/connections cannot be mixed "
                "with legacy generation/constraints"
            )

        selectors: dict[str, str] = {}
        for component_id, component in self.components.items():
            for selector in component.selectors:
                previous = selectors.setdefault(selector, component_id)
                if previous != component_id:
                    raise ValueError(
                        f"Selector {selector!r} belongs to both "
                        f"{previous!r} and {component_id!r}"
                    )

        for port_id, port in self.ports.items():
            component = self.components.get(port.component)
            if component is None:
                raise ValueError(
                    f"Port {port_id!r} references unknown component "
                    f"{port.component!r}"
                )
            unknown_selectors = sorted(
                set(port.selectors) - set(component.selectors)
            )
            if unknown_selectors:
                raise ValueError(
                    f"Port {port_id!r} selectors do not belong to "
                    f"component {port.component!r}: {unknown_selectors}"
                )

        interface_ids: set[str] = set()
        interface_nodes = self.ports or self.components
        interface_node_kind = "ports" if self.ports else "components"
        for interface in self.interfaces:
            if interface.id in interface_ids:
                raise ValueError(
                    f"Duplicate assembly interface id {interface.id!r}"
                )
            interface_ids.add(interface.id)
            unknown = sorted(set(interface.between) - set(interface_nodes))
            if unknown:
                raise ValueError(
                    f"Interface {interface.id!r} references unknown "
                    f"{interface_node_kind}: {unknown}"
                )

        connection_ids: set[str] = set()
        for connection in self.connections:
            if connection.id in connection_ids:
                raise ValueError(
                    f"Duplicate assembly connection id {connection.id!r}"
                )
            connection_ids.add(connection.id)
            for endpoint in (
                connection.from_endpoint,
                connection.to_endpoint,
            ):
                component = self.components.get(endpoint.component)
                if component is None:
                    raise ValueError(
                        f"Connection {connection.id!r} references unknown "
                        f"component {endpoint.component!r}"
                    )
                if endpoint.selector is None:
                    if len(component.selectors) != 1:
                        raise ValueError(
                            f"Connection {connection.id!r} endpoint "
                            f"{endpoint.component!r} must select one of its "
                            "multiple component selectors"
                        )
                elif endpoint.selector not in component.selectors:
                    raise ValueError(
                        f"Connection {connection.id!r} selector "
                        f"{endpoint.selector!r} does not belong to component "
                        f"{endpoint.component!r}"
                    )
        return self

    @model_validator(mode="after")
    def validate_task_contract(self) -> "UserDesignSpec":
        """Keep workflow presets explicit, narrow and fail-closed.

        ``fixed_xyz`` always preserves the selected atoms as exact rigid
        geometry.  Creating generated interface material does not imply that
        fixed components may move: ``fixed_arrangement=locked`` is the safe
        default.  Only ``optimize_components`` changes complete rigid poses.
        Expert authors declare component mobility directly. Resolver-emitted
        expert graphs may retain ``preserve_supplied_geometry`` so downstream
        stages never confuse supplied hyperedges with generated contacts.
        """

        motion = self.preferences.component_motion
        if self.guidance is not None and self.user_mode != "expert":
            raise ValueError(
                "raw guidance weights require expert assembly components; "
                "ordinary designs use preferences"
            )
        if (
            motion is not None
            and self.task is not None
            and self.user_mode == "simple"
        ):
            expected = (
                FixedArrangementPolicy.LOCKED
                if motion == ComponentMotionPreference.LOCKED
                else FixedArrangementPolicy.OPTIMIZE_COMPONENTS
            )
            if self.fixed_arrangement != expected:
                raise ValueError(
                    "preferences.component_motion conflicts with "
                    "fixed_arrangement"
                )

        if self.task is None:
            if self.fixed_arrangement != FixedArrangementPolicy.LOCKED:
                raise ValueError(
                    "fixed_arrangement=optimize_components requires "
                    "task=create_symmetric_interface; expert designs use "
                    "component pose declarations"
                )
            if motion is not None and not self.components:
                raise ValueError(
                    "preferences.component_motion requires either "
                    "task=create_symmetric_interface or assembly components"
                )
            if motion is not None and self.components:
                expected_mode = (
                    "fixed"
                    if motion == ComponentMotionPreference.LOCKED
                    else "bounded_mobile"
                )
                wrong_modes = sorted(
                    component_id
                    for component_id, component in self.components.items()
                    if component.pose.mode != expected_mode
                )
                if wrong_modes:
                    raise ValueError(
                        "preferences.component_motion does not match "
                        "component pose declarations: "
                        + ", ".join(wrong_modes)
                    )
                if motion != ComponentMotionPreference.LOCKED:
                    expected_subspace = (
                        "bounded_se3"
                        if motion == ComponentMotionPreference.FREE
                        else "radial_axial_rotation"
                    )
                    wrong_subspaces = sorted(
                        component_id
                        for component_id, component in self.components.items()
                        if (component.pose.subspace or "bounded_se3")
                        != expected_subspace
                    )
                    if wrong_subspaces:
                        raise ValueError(
                            "preferences.component_motion does not match "
                            "component mobility subspaces: "
                            + ", ".join(wrong_subspaces)
                        )
            return self
        if self.user_mode != "simple":
            if self.task != UserDesignTask.PRESERVE_SUPPLIED_GEOMETRY:
                raise ValueError(
                    "task=create_symmetric_interface is a compact motif "
                    "workflow; expert assembly graphs declare generated "
                    "contact interfaces explicitly"
                )
            if self.fixed_arrangement != FixedArrangementPolicy.LOCKED:
                raise ValueError(
                    "expert preserve_supplied_geometry uses component.pose "
                    "for whole-seed motion; fixed_arrangement must remain "
                    "locked"
                )
            if not self.interfaces:
                raise ValueError(
                    "expert preserve_supplied_geometry requires at least "
                    "one supplied interface"
                )
            generated_interfaces = sorted(
                interface.id
                for interface in self.interfaces
                if interface.relation.mode != "preserve_input"
            )
            if generated_interfaces:
                raise ValueError(
                    "preserve_supplied_geometry cannot redesign supplied "
                    "interfaces: " + ", ".join(generated_interfaces)
                )
            non_joint_components = sorted(
                component_id
                for component_id, component in self.components.items()
                if component.geometry != "joint_rigid"
            )
            if non_joint_components:
                raise ValueError(
                    "preserve_supplied_geometry requires every supplied "
                    "interface component to be joint_rigid: "
                    + ", ".join(non_joint_components)
                )
            independently_mobile_interfaces: list[str] = []
            for interface in self.interfaces:
                participant_components = {
                    (
                        self.ports[participant].component
                        if self.ports
                        else participant
                    )
                    for participant in interface.between
                }
                if len(participant_components) == 1:
                    continue
                if any(
                    self.components[component_id].pose.mode
                    == "bounded_mobile"
                    for component_id in participant_components
                ):
                    independently_mobile_interfaces.append(interface.id)
            if independently_mobile_interfaces:
                raise ValueError(
                    "A supplied interface spanning different components "
                    "cannot move those components independently; keep them "
                    "fixed or place every participant in one joint-rigid "
                    "motion group: "
                    + ", ".join(sorted(independently_mobile_interfaces))
                )
            # The preserved object is the complete interface geometry, not
            # necessarily its global assembly pose.  A resolved expert graph
            # may therefore move each joint-rigid seed as one SE(3) body.
            if motion is not None:
                expected_mode = (
                    "fixed"
                    if motion == ComponentMotionPreference.LOCKED
                    else "bounded_mobile"
                )
                wrong_modes = sorted(
                    component_id
                    for component_id, component in self.components.items()
                    if component.pose.mode != expected_mode
                )
                if wrong_modes:
                    raise ValueError(
                        "preferences.component_motion does not match "
                        "supplied-interface component poses: "
                        + ", ".join(wrong_modes)
                    )
                if motion != ComponentMotionPreference.LOCKED:
                    expected_subspace = (
                        "bounded_se3"
                        if motion == ComponentMotionPreference.FREE
                        else "radial_axial_rotation"
                    )
                    wrong_subspaces = sorted(
                        component_id
                        for component_id, component in self.components.items()
                        if (component.pose.subspace or "bounded_se3")
                        != expected_subspace
                    )
                    if wrong_subspaces:
                        raise ValueError(
                            "preferences.component_motion does not match "
                            "supplied-interface component mobility "
                            "subspaces: " + ", ".join(wrong_subspaces)
                        )
            return self
        fixed_constraints = tuple(
            constraint
            for constraint in self.constraints
            if isinstance(constraint, FixedXYZConstraint)
        )
        non_fixed_constraints = tuple(
            constraint
            for constraint in self.constraints
            if not isinstance(constraint, FixedXYZConstraint)
        )
        if not fixed_constraints:
            raise ValueError(
                f"task={self.task.value} requires at least one fixed_xyz "
                "motif declaration"
            )
        if non_fixed_constraints:
            raise ValueError(
                "ordinary task presets currently accept fixed_xyz motif "
                "declarations only; use explicit constraints for a custom "
                "expert design"
            )
        mobile = tuple(
            constraint
            for constraint in fixed_constraints
            if constraint.pose.mode != "fixed"
        )
        if mobile:
            raise ValueError(
                f"task={self.task.value} owns the ordinary motif-orbit pose "
                "contract; omit fixed_xyz.pose or omit task and declare "
                "expert mobility explicitly"
            )
        if self.task == UserDesignTask.CREATE_SYMMETRIC_INTERFACE:
            if not self.generation or not all(
                isinstance(clause, TerminalGeneration)
                for clause in self.generation
            ):
                raise ValueError(
                    "task=create_symmetric_interface requires terminal "
                    "generation around one internal fixed motif; supplied "
                    "fixed interfaces connected by between generation use "
                    "task=preserve_supplied_geometry"
                )
            symmetry_id = (
                self.symmetry
                if isinstance(self.symmetry, str)
                else self.symmetry.id
            )
            if (
                self.fixed_arrangement
                == FixedArrangementPolicy.OPTIMIZE_COMPONENTS
                and not symmetry_id.startswith(("C", "D"))
            ):
                raise ValueError(
                    "fixed_arrangement=optimize_components currently "
                    "supports Cn and Dn because its safe orbit-adaptation "
                    "controller requires a principal symmetry axis; locked "
                    "generated-interface guidance is topology-neutral"
                )
        elif self.fixed_arrangement != FixedArrangementPolicy.LOCKED:
            raise ValueError(
                "preserve_supplied_geometry requires "
                "fixed_arrangement=locked"
            )
        if (
            self.task == UserDesignTask.PRESERVE_SUPPLIED_GEOMETRY
            and motion not in {None, ComponentMotionPreference.LOCKED}
        ):
            raise ValueError(
                "preserve_supplied_geometry requires "
                "preferences.component_motion=locked"
            )
        return self


def load_user_design(path: str | Path) -> UserDesignSpec:
    """Load one public design file and resolve its structure path.

    Loading is intentionally side-effect free.  It validates only the public
    declaration; assembly lowering and backend capability checks happen in
    later compiler stages.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"User design does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("User design must contain a YAML mapping")
    design = UserDesignSpec.model_validate(payload)
    structure = design.input.expanduser()
    if not structure.is_absolute():
        structure = source.parent / structure
    structure = structure.resolve()
    if not structure.is_file():
        raise FileNotFoundError(f"Design input does not exist: {structure}")
    output = design.output
    if output is not None:
        root = output.root.expanduser()
        if not root.is_absolute():
            root = source.parent / root
        output = output.model_copy(update={"root": root.resolve()})
    return design.model_copy(update={"input": structure, "output": output})
