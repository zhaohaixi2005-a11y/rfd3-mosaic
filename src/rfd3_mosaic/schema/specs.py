from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Identifier = Annotated[
    str,
    Field(
        min_length=1,
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$",
    ),
]

TransformIdentifier = Annotated[
    str,
    Field(
        min_length=3,
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]*:[A-Za-z0-9_.-]+$",
    ),
]


class StrictModel(BaseModel):
    """Base class for immutable, strict configuration objects."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
    )


class EntityType(str, Enum):
    PROTEIN = "protein"
    LIGAND = "ligand"
    METAL = "metal"
    DNA = "dna"
    RNA = "rna"


class FragmentRole(str, Enum):
    INTERFACE_MOTIF = "interface_motif"
    FUNCTIONAL_MOTIF = "functional_motif"
    FUNCTIONAL_COMPONENT = "functional_component"
    SUPPORT = "support"


class MotionMode(str, Enum):
    FIXED = "fixed"
    RIGID = "rigid"
    SOFT_RIGID = "soft_rigid"


class OrbitMobilityMode(str, Enum):
    FIXED = "fixed"
    ORBIT_RIGID = "orbit_rigid"


class MobilitySubspace(str, Enum):
    """Allowed physical degrees of freedom for one master orbit pose."""

    RADIAL = "radial"
    RADIAL_AXIAL = "radial_axial"
    RADIAL_ROTATION = "radial_rotation"
    RADIAL_AXIAL_ROTATION = "radial_axial_rotation"
    TILT_ONLY = "tilt_only"
    BOUNDED_SE3 = "bounded_se3"


class MobilityProposal(str, Enum):
    """Source of one timestep's bounded rigid-pose proposal."""

    DENOISER_FIT = "denoiser_fit"
    SCAFFOLD_OBJECTIVES = "scaffold_objectives"
    HOYEUNG_DRAG_COMPAT = "hoyeung_drag_compat"


class FrameMethod(str, Enum):
    ANCHORS = "anchors"
    REFERENCE_INTERFACE_PCA = "reference_interface_pca"
    PRINCIPAL_AXIS_WITH_ANCHOR = "principal_axis_with_anchor"
    PRECOMPUTED = "precomputed"


class FragmentSpec(StrictModel):
    """Describes what a structural fragment is."""

    source: Path
    selection: Annotated[str, Field(min_length=1)]
    entity_type: EntityType
    role: FragmentRole
    fixed_atoms: str | list[str] | None = None
    provenance_tags: tuple[str, ...] = ()


class MotionBounds(StrictModel):
    max_translation: Annotated[float, Field(ge=0.0)] | None = None
    max_rotation_deg: Annotated[float, Field(ge=0.0, le=180.0)] | None = None

    @model_validator(mode="after")
    def require_at_least_one_bound(self) -> "MotionBounds":
        if self.max_translation is None and self.max_rotation_deg is None:
            raise ValueError("At least one motion bound must be provided")
        return self


class MotionGroupSpec(StrictModel):
    """Defines how one or more fragments are allowed to move."""

    members: Annotated[list[Identifier], Field(min_length=1)]
    mode: MotionMode
    bounds: MotionBounds | None = None

    @model_validator(mode="after")
    def validate_members_and_bounds(self) -> "MotionGroupSpec":
        if len(self.members) != len(set(self.members)):
            raise ValueError("Motion-group members must be unique")

        if self.mode == MotionMode.SOFT_RIGID and self.bounds is None:
            raise ValueError("soft_rigid motion groups require bounds")

        if self.mode != MotionMode.SOFT_RIGID and self.bounds is not None:
            raise ValueError(
                "Motion bounds are only valid for soft_rigid groups"
            )

        return self


class InterfacePortFrameSpec(StrictModel):
    """Defines a deterministic local coordinate frame for an interface port."""

    method: FrameMethod

    origin_atoms: list[str] | None = None
    x_axis_atoms: list[str] | None = None
    xy_plane_atoms: list[str] | None = None

    anchor_atom: str | None = None
    transform: list[list[float]] | None = None

    @model_validator(mode="after")
    def validate_frame_definition(self) -> "InterfacePortFrameSpec":
        if self.method == FrameMethod.ANCHORS:
            if not self.origin_atoms:
                raise ValueError("anchors frame requires origin_atoms")
            if self.x_axis_atoms is None or len(self.x_axis_atoms) != 2:
                raise ValueError(
                    "anchors frame requires exactly two x_axis_atoms"
                )
            if self.xy_plane_atoms is None or len(self.xy_plane_atoms) != 3:
                raise ValueError(
                    "anchors frame requires exactly three xy_plane_atoms"
                )
            if self.transform is not None:
                raise ValueError(
                    "anchors frame cannot also define a transform"
                )

        elif self.method == FrameMethod.PRECOMPUTED:
            if self.transform is None:
                raise ValueError(
                    "precomputed frame requires a transform"
                )
            if len(self.transform) != 4:
                raise ValueError("Precomputed transform must be 4x4")
            if any(len(row) != 4 for row in self.transform):
                raise ValueError("Precomputed transform must be 4x4")

        elif self.method == FrameMethod.PRINCIPAL_AXIS_WITH_ANCHOR:
            if self.anchor_atom is None:
                raise ValueError(
                    "principal_axis_with_anchor requires anchor_atom"
                )
            if self.transform is not None:
                raise ValueError(
                    "principal-axis frame cannot define a transform"
                )

        elif self.method == FrameMethod.REFERENCE_INTERFACE_PCA:
            if self.transform is not None:
                raise ValueError(
                    "PCA frame cannot define a precomputed transform"
                )

        return self


class InterfacePortSpec(StrictModel):
    """Defines an interface-bearing region on a motion group."""

    group: Identifier
    fragments: Annotated[list[Identifier], Field(min_length=1)]
    atoms: str = "heavy"
    frame: InterfacePortFrameSpec

    @model_validator(mode="after")
    def require_unique_fragments(self) -> "InterfacePortSpec":
        if len(self.fragments) != len(set(self.fragments)):
            raise ValueError("Interface-port fragments must be unique")
        return self
class CopyRelationSpec(StrictModel):
    """Identifies which symmetry-related copy a port connects to."""

    orbit_offset: int | None = None
    transform: TransformIdentifier | None = None

    @model_validator(mode="after")
    def require_exactly_one_relation(self) -> "CopyRelationSpec":
        provided = [
            self.orbit_offset is not None,
            self.transform is not None,
        ]
        if sum(provided) != 1:
            raise ValueError(
                "Exactly one of orbit_offset or transform must be provided"
            )
        return self


class ReferenceTransformGeometry(StrictModel):
    """Preserves an interface pose from a reference seed."""

    mode: Literal["reference_transform"] = "reference_transform"
    from_reference_seed: bool = True
    target_transform: list[list[float]] | None = None

    translation_tolerance: Annotated[float, Field(gt=0.0)] = 2.0
    rotation_tolerance_deg: Annotated[
        float,
        Field(gt=0.0, le=180.0),
    ] = 10.0
    # Assembly IR remains backwards-compatible for non-interface geometric
    # relations. The public `preserve_input` interface frontend lowers an
    # explicit positive requirement by default.
    minimum_heavy_atom_contacts: Annotated[int, Field(ge=0)] = 0
    contact_cutoff: Annotated[float, Field(gt=0.0)] = 4.5

    @model_validator(mode="after")
    def validate_reference_source(self) -> "ReferenceTransformGeometry":
        if self.from_reference_seed and self.target_transform is not None:
            raise ValueError(
                "Do not provide target_transform when "
                "from_reference_seed is true"
            )

        if not self.from_reference_seed and self.target_transform is None:
            raise ValueError(
                "target_transform is required when "
                "from_reference_seed is false"
            )

        if self.target_transform is not None:
            if len(self.target_transform) != 4:
                raise ValueError("target_transform must be 4x4")
            if any(len(row) != 4 for row in self.target_transform):
                raise ValueError("target_transform must be 4x4")

        return self


class DistanceConstraint(StrictModel):
    type: Literal[
        "anchor",
        "com",
        "plane_to_plane",
        "axis_to_axis",
    ]
    target: Annotated[float, Field(ge=0.0)]
    tolerance: Annotated[float, Field(gt=0.0)]


class AngleConstraint(StrictModel):
    target: Annotated[float, Field(ge=0.0, le=180.0)]
    tolerance: Annotated[float, Field(gt=0.0, le=180.0)]


class AngleRangeConstraint(StrictModel):
    minimum: Annotated[float, Field(ge=-180.0, le=180.0)]
    maximum: Annotated[float, Field(ge=-180.0, le=180.0)]

    @model_validator(mode="after")
    def validate_range(self) -> "AngleRangeConstraint":
        if self.minimum > self.maximum:
            raise ValueError("minimum cannot be greater than maximum")
        return self


class ContactConstraint(StrictModel):
    min_heavy_atom_contacts: Annotated[int, Field(ge=0)] = 0
    cutoff: Annotated[float, Field(gt=0.0)] = 8.0


class InterfaceCoverageConstraint(StrictModel):
    """Scale-aware interface-quality contract used by the sampler and audit.

    ``auto`` is the public default.  Optional numeric values exist only as an
    internal/expert replay mechanism; the normal compiler leaves them unset
    so the runtime can derive targets from the available generated residues.
    """

    mode: Literal["auto"] = "auto"
    minimum_contact_residues_per_side: Annotated[
        int,
        Field(ge=1),
    ] | None = None
    minimum_contiguous_contact_residues_per_side: Annotated[
        int,
        Field(ge=1),
    ] | None = None

    @model_validator(mode="after")
    def contiguous_target_cannot_exceed_coverage(
        self,
    ) -> "InterfaceCoverageConstraint":
        if (
            self.minimum_contact_residues_per_side is not None
            and self.minimum_contiguous_contact_residues_per_side is not None
            and self.minimum_contiguous_contact_residues_per_side
            > self.minimum_contact_residues_per_side
        ):
            raise ValueError(
                "minimum_contiguous_contact_residues_per_side cannot exceed "
                "minimum_contact_residues_per_side"
            )
        return self


class GeometricConstraintsGeometry(StrictModel):
    """Defines an exploratory interface without a reference pose."""

    mode: Literal["geometric_constraints"] = "geometric_constraints"

    distance: DistanceConstraint | None = None
    normal_angle_deg: AngleConstraint | None = None
    twist_deg: AngleRangeConstraint | None = None
    contacts: ContactConstraint | None = None
    coverage: InterfaceCoverageConstraint | None = None

    @model_validator(mode="after")
    def require_at_least_one_constraint(
        self,
    ) -> "GeometricConstraintsGeometry":
        constraints = [
            self.distance,
            self.normal_angle_deg,
            self.twist_deg,
            self.contacts,
            self.coverage,
        ]

        if all(value is None for value in constraints):
            raise ValueError(
                "At least one geometric constraint must be provided"
            )

        return self


TargetGeometrySpec = Annotated[
    ReferenceTransformGeometry | GeometricConstraintsGeometry,
    Field(discriminator="mode"),
]


class MobilityScheduleSpec(StrictModel):
    """Time window and per-step trust region for orbit motion."""

    start_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.05
    end_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.75
    response: Annotated[float, Field(gt=0.0, le=1.0)] = 0.2
    max_step_translation: Annotated[float, Field(gt=0.0)] = 0.25
    max_step_rotation_deg: Annotated[
        float,
        Field(gt=0.0, le=180.0),
    ] = 1.0

    @model_validator(mode="after")
    def validate_window(self) -> "MobilityScheduleSpec":
        if self.start_fraction >= self.end_fraction:
            raise ValueError(
                "Mobility schedule requires start_fraction < end_fraction"
            )
        return self


class OrbitMobilitySpec(StrictModel):
    """Optional bounded motion of one complete symmetry constraint orbit.

    Mobility belongs to an orbit, not to an interface.  The legacy interface
    field still accepts this same type and is migrated into the orbit IR by
    the compiler.
    """

    mode: OrbitMobilityMode = OrbitMobilityMode.FIXED
    bounds: MotionBounds | None = None
    subspace: MobilitySubspace | None = None
    proposal: MobilityProposal | None = None
    schedule: MobilityScheduleSpec | None = None
    objectives: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_bounds(self) -> "OrbitMobilitySpec":
        if (
            self.mode == OrbitMobilityMode.ORBIT_RIGID
            and self.bounds is None
        ):
            raise ValueError(
                "orbit_rigid interface mobility requires cumulative bounds"
            )
        if self.mode == OrbitMobilityMode.ORBIT_RIGID:
            assert self.bounds is not None
            subspace = self.effective_subspace
            translation = float(self.bounds.max_translation or 0.0)
            rotation = float(self.bounds.max_rotation_deg or 0.0)
            if subspace in {
                MobilitySubspace.RADIAL,
                MobilitySubspace.RADIAL_AXIAL,
                MobilitySubspace.RADIAL_ROTATION,
                MobilitySubspace.RADIAL_AXIAL_ROTATION,
            } and translation <= 0.0:
                raise ValueError(
                    f"{subspace.value} mobility requires a positive "
                    "translation bound"
                )
            if (
                subspace
                in {
                    MobilitySubspace.RADIAL_ROTATION,
                    MobilitySubspace.RADIAL_AXIAL_ROTATION,
                }
                and rotation <= 0.0
            ):
                raise ValueError(
                    f"{subspace.value} mobility requires a positive "
                    "rotation bound"
                )
            if (
                subspace == MobilitySubspace.TILT_ONLY
                and rotation <= 0.0
            ):
                raise ValueError(
                    "tilt_only mobility requires a positive rotation bound"
                )
            if (
                subspace == MobilitySubspace.BOUNDED_SE3
                and (translation <= 0.0 or rotation <= 0.0)
            ):
                raise ValueError(
                    "bounded_se3 mobility requires positive translation "
                    "and rotation bounds"
                )
        if (
            self.mode == OrbitMobilityMode.FIXED
            and any(
                value is not None
                for value in (
                    self.bounds,
                    self.subspace,
                    self.proposal,
                    self.schedule,
                )
            )
        ):
            raise ValueError(
                "fixed orbit mobility cannot define motion controls"
            )
        if self.mode == OrbitMobilityMode.FIXED and self.objectives:
            raise ValueError(
                "fixed orbit mobility cannot define mobility objectives"
            )
        if len(self.objectives) != len(set(self.objectives)):
            raise ValueError("Orbit mobility objectives must be unique")
        return self

    @property
    def effective_subspace(self) -> MobilitySubspace | None:
        if self.mode == OrbitMobilityMode.FIXED:
            return None
        return self.subspace or MobilitySubspace.BOUNDED_SE3

    @property
    def effective_proposal(self) -> MobilityProposal | None:
        if self.mode == OrbitMobilityMode.FIXED:
            return None
        return self.proposal or MobilityProposal.DENOISER_FIT

    @property
    def effective_schedule(self) -> MobilityScheduleSpec | None:
        if self.mode == OrbitMobilityMode.FIXED:
            return None
        return self.schedule or MobilityScheduleSpec()


# Compatibility public names used by existing Interface-Seed 2.0 configs.
InterfaceMobilityMode = OrbitMobilityMode
InterfaceMobilitySpec = OrbitMobilitySpec


class InterfaceEdgeSpec(StrictModel):
    """Defines a target relationship between two interface ports."""

    left_port: Identifier
    right_port: Identifier
    hyperedge_id: Identifier | None = None
    copy_relation: CopyRelationSpec
    required: bool = True
    # Input-stage edges must already be satisfied by the compiled seed.
    # Output-stage edges are design objectives: preflight records their
    # residual but the post-diffusion audit enforces the contract.
    satisfaction_stage: Literal["input", "output"] = "input"
    target_geometry: TargetGeometrySpec
    mobility: OrbitMobilitySpec = OrbitMobilitySpec()


class SymmetryType(str, Enum):
    CYCLIC = "cyclic"
    DIHEDRAL = "dihedral"
    TETRAHEDRAL = "tetrahedral"
    OCTAHEDRAL = "octahedral"
    ICOSAHEDRAL = "icosahedral"


class SymmetryTransformSetSpec(StrictModel):
    """Defines a named set of symmetry group transformations."""

    type: SymmetryType
    order: Annotated[int, Field(ge=2)]
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    secondary_axis: tuple[float, float, float] | None = None
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def require_nonzero_axis(self) -> "SymmetryTransformSetSpec":
        squared_norm = sum(component * component for component in self.axis)

        if squared_norm <= 1e-12:
            raise ValueError("Symmetry axis cannot be the zero vector")

        if self.type == SymmetryType.CYCLIC:
            if self.secondary_axis is not None:
                raise ValueError(
                    "secondary_axis is not valid for cyclic symmetry"
                )
            return self

        expected_polyhedral_orders = {
            SymmetryType.TETRAHEDRAL: 12,
            SymmetryType.OCTAHEDRAL: 24,
            SymmetryType.ICOSAHEDRAL: 60,
        }
        expected_order = expected_polyhedral_orders.get(self.type)
        if expected_order is not None and self.order != expected_order:
            raise ValueError(
                f"{self.type.value} symmetry requires exactly "
                f"{expected_order} proper rotations"
            )

        if self.secondary_axis is not None:
            secondary_norm = sum(
                component * component for component in self.secondary_axis
            )
            if secondary_norm <= 1e-12:
                raise ValueError("Secondary symmetry axis cannot be zero")
            dot_product = sum(
                left * right
                for left, right in zip(self.axis, self.secondary_axis)
            )
            normalized_dot = abs(dot_product) / (
                squared_norm * secondary_norm
            ) ** 0.5
            if normalized_dot > 1e-6:
                raise ValueError(
                    "secondary_axis must be perpendicular to axis"
                )

        return self


class FiniteOrbitActionSpec(StrictModel):
    """One transitive finite-group action represented by coset data.

    ``coset_representative_ids`` are the distinct physical copies.  The
    stabilizer and complete transform-to-representative map make quotient
    action resolution deterministic for non-commutative groups.
    """

    coset_representative_ids: Annotated[
        tuple[TransformIdentifier, ...],
        Field(min_length=1),
    ]
    stabilizer_transform_ids: Annotated[
        tuple[TransformIdentifier, ...],
        Field(min_length=1),
    ]
    transform_to_coset_representative: dict[
        TransformIdentifier,
        TransformIdentifier,
    ]
    stabilizer_path_transform_ids: dict[
        Identifier,
        TransformIdentifier,
    ] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
        description=(
            "Optional compiler-frozen assignment from generated scaffold "
            "path IDs to transforms inside the component stabilizer. This "
            "is used when a complete stabilized oligomer is already "
            "materialized as one preexpanded RFD3 ASU."
        ),
    )

    @model_validator(mode="after")
    def validate_identifiers(self) -> "FiniteOrbitActionSpec":
        representatives = self.coset_representative_ids
        stabilizer = self.stabilizer_transform_ids
        if len(representatives) != len(set(representatives)):
            raise ValueError("Coset representative IDs must be unique")
        if len(stabilizer) != len(set(stabilizer)):
            raise ValueError("Stabilizer transform IDs must be unique")
        unknown_values = set(
            self.transform_to_coset_representative.values()
        ) - set(representatives)
        if unknown_values:
            raise ValueError(
                "Transform-to-coset map references unknown representatives: "
                f"{sorted(unknown_values)}"
            )
        unknown_path_transforms = set(
            self.stabilizer_path_transform_ids.values()
        ) - set(stabilizer)
        if unknown_path_transforms:
            raise ValueError(
                "Stabilizer path assignments reference transforms outside "
                f"the declared stabilizer: {sorted(unknown_path_transforms)}"
            )
        if (
            self.stabilizer_path_transform_ids
            and len(representatives) != 1
        ):
            raise ValueError(
                "Stabilizer path assignments currently require a single "
                "physical quotient representative (a complete G/G "
                "stabilized ASU)"
            )
        return self


class SymmetryOrbitSpec(StrictModel):
    """Expands one or more master groups through a transform set."""

    transform_set: Identifier
    master_groups: Annotated[list[Identifier], Field(min_length=1)]
    mobility: OrbitMobilitySpec = OrbitMobilitySpec()
    component_mobility: dict[Identifier, OrbitMobilitySpec] = Field(
        default_factory=dict
    )
    finite_action: FiniteOrbitActionSpec | None = None

    @model_validator(mode="after")
    def require_unique_master_groups(self) -> "SymmetryOrbitSpec":
        if len(self.master_groups) != len(set(self.master_groups)):
            raise ValueError("master_groups must be unique")
        unknown = set(self.component_mobility) - set(self.master_groups)
        if unknown:
            raise ValueError(
                "component_mobility references groups outside this orbit: "
                f"{sorted(unknown)}"
            )
        if (
            self.mobility.mode != OrbitMobilityMode.FIXED
            and self.component_mobility
        ):
            raise ValueError(
                "An orbit cannot combine orbit-wide mobility with "
                "per-component mobility"
            )
        return self


class ObjectiveMode(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    AT_MOST = "at_most"
    AT_LEAST = "at_least"
    TARGET = "target"
    RANGE = "range"


class ObjectiveSpec(StrictModel):
    """Backend-independent scalar objective used to rank candidate poses."""

    metric: Annotated[
        str,
        Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$"),
    ]
    mode: ObjectiveMode
    weight: Annotated[float, Field(gt=0.0)] = 1.0
    scale: Annotated[float, Field(gt=0.0)] = 1.0
    required: bool = False
    threshold: float | None = None
    target: float | None = None
    tolerance: Annotated[float, Field(ge=0.0)] | None = None
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_mode_parameters(self) -> "ObjectiveSpec":
        provided = {
            "threshold": self.threshold,
            "target": self.target,
            "tolerance": self.tolerance,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }
        required_fields: dict[ObjectiveMode, set[str]] = {
            ObjectiveMode.MINIMIZE: set(),
            ObjectiveMode.MAXIMIZE: set(),
            ObjectiveMode.AT_MOST: {"threshold"},
            ObjectiveMode.AT_LEAST: {"threshold"},
            ObjectiveMode.TARGET: {"target", "tolerance"},
            ObjectiveMode.RANGE: {"minimum", "maximum"},
        }
        expected = required_fields[self.mode]
        actual = {key for key, value in provided.items() if value is not None}
        if actual != expected:
            raise ValueError(
                f"Objective mode {self.mode.value!r} requires exactly "
                f"{sorted(expected)}, got {sorted(actual)}"
            )
        if (
            self.mode == ObjectiveMode.RANGE
            and self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("Objective minimum cannot exceed maximum")
        if self.required and self.mode in {
            ObjectiveMode.MINIMIZE,
            ObjectiveMode.MAXIMIZE,
        }:
            raise ValueError(
                "Directional objectives cannot be required because they "
                "do not define a satisfaction boundary"
            )
        return self


class CenterMethod(str, Enum):
    NONE = "none"
    INTERFACE_HEAVY_ATOM_COM = "interface_heavy_atom_com"


class SampleRangeSpec(StrictModel):
    """A reproducible scalar sampling interval around a mean."""

    mean: float
    range: Annotated[float, Field(ge=0.0)] = 0.0


class FixedOrientationSpec(StrictModel):
    """Explicit intrinsic XYZ Euler orientation for a master group."""

    method: Literal["fixed"] = "fixed"
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


class UniformSO3OrientationSpec(StrictModel):
    """Haar-uniform rigid orientation sampled on SO(3)."""

    method: Literal["uniform_so3"] = "uniform_so3"


OrientationSpec = Annotated[
    FixedOrientationSpec | UniformSO3OrientationSpec,
    Field(discriminator="method"),
]


class RadialPlacementSpec(StrictModel):
    radius: SampleRangeSpec
    axial_offset: SampleRangeSpec = SampleRangeSpec(mean=0.0)
    radial_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)

    @model_validator(mode="after")
    def validate_radial_direction(self) -> "RadialPlacementSpec":
        squared_norm = sum(
            component * component for component in self.radial_direction
        )
        if squared_norm <= 1e-12:
            raise ValueError("radial_direction cannot be the zero vector")
        if self.radius.mean - self.radius.range < 0.0:
            raise ValueError("Sampled placement radius cannot be negative")
        return self


class MotionGroupInitializationSpec(StrictModel):
    """Initial master pose before applying a symmetry group action."""

    random_seed: Annotated[int, Field(ge=0)] | None = None
    center_method: CenterMethod = CenterMethod.INTERFACE_HEAVY_ATOM_COM
    orientation: OrientationSpec = FixedOrientationSpec()
    placement: RadialPlacementSpec


class Terminus(str, Enum):
    N = "N"
    C = "C"


class ScaffoldEndpointSpec(StrictModel):
    fragment: Identifier
    terminus: Terminus


class LinkLengthSpec(StrictModel):
    minimum: Annotated[int, Field(ge=0)]
    maximum: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_length_range(self) -> "LinkLengthSpec":
        if self.minimum > self.maximum:
            raise ValueError(
                "Link minimum cannot be greater than maximum"
            )
        return self


class ScaffoldLinkSpec(StrictModel):
    """Defines a directed protein connection."""

    from_endpoint: ScaffoldEndpointSpec
    to_endpoint: ScaffoldEndpointSpec
    length: LinkLengthSpec
    tie_group: Identifier | None = None
    chain_break: bool = False
    copy_relation: CopyRelationSpec = CopyRelationSpec(orbit_offset=0)

    @model_validator(mode="after")
    def validate_direction(self) -> "ScaffoldLinkSpec":
        if not self.chain_break:
            if self.from_endpoint.terminus != Terminus.C:
                raise ValueError(
                    "A continuous scaffold link must start at C terminus"
                )
            if self.to_endpoint.terminus != Terminus.N:
                raise ValueError(
                    "A continuous scaffold link must end at N terminus"
                )

        if self.from_endpoint == self.to_endpoint:
            raise ValueError(
                "Scaffold link endpoints cannot be identical"
            )

        return self


class TerminalExtensionSpec(StrictModel):
    """A generated N- or C-terminal segment anchored to one fragment."""

    anchor: ScaffoldEndpointSpec
    length: LinkLengthSpec
    tie_group: Identifier | None = None


GeneratedSegmentSpec = ScaffoldLinkSpec | TerminalExtensionSpec

class SymmetrySpec(StrictModel):
    transform_sets: Annotated[
        dict[Identifier, SymmetryTransformSetSpec],
        Field(min_length=1),
    ]
    orbits: Annotated[
        dict[Identifier, SymmetryOrbitSpec],
        Field(min_length=1),
    ]


class AssemblySpecification(StrictModel):
    """Generator-independent description of a constrained assembly.

    ``InterfaceSeedSpec`` remains as a compatibility alias below.  The
    neutral name is intentional: a central motif, a cross-protomer interface
    seed, and a multi-orbit cage are all represented by the same fragments,
    motion groups, symmetry orbits, scaffold links, and constraint edges.
    """

    schema_version: Literal[2] = 2
    mode: Literal[
        "constraint_assembly",
        "se3_static",
        "multi_interface_se3",
        "legacy_rfd1",
    ] = "se3_static"
    random_seed: int | None = None
    constraint_group_strategy: Literal[
        "auto",
        "interface_edges",
        "motion_groups",
    ] = "auto"

    fragments: Annotated[
        dict[Identifier, FragmentSpec],
        Field(min_length=1),
    ]
    motion_groups: Annotated[
        dict[Identifier, MotionGroupSpec],
        Field(min_length=1),
    ]
    ports: dict[Identifier, InterfacePortSpec] = Field(
        default_factory=dict
    )
    symmetry: SymmetrySpec
    interfaces: dict[Identifier, InterfaceEdgeSpec] = Field(
        default_factory=dict
    )
    scaffold_links: dict[Identifier, ScaffoldLinkSpec] = Field(
        default_factory=dict
    )
    generated_segments: dict[
        Identifier,
        GeneratedSegmentSpec,
    ] = Field(default_factory=dict)
    initialization: dict[
        Identifier,
        MotionGroupInitializationSpec,
    ] = Field(default_factory=dict)
    objectives: dict[Identifier, ObjectiveSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_cross_references(self) -> "AssemblySpecification":
        fragment_owners: dict[str, list[str]] = {
            fragment_id: [] for fragment_id in self.fragments
        }

        # Every motion-group member must reference an existing fragment.
        for group_id, group in self.motion_groups.items():
            for fragment_id in group.members:
                if fragment_id not in self.fragments:
                    raise ValueError(
                        f"Motion group {group_id!r} references unknown "
                        f"fragment {fragment_id!r}"
                    )
                fragment_owners[fragment_id].append(group_id)

        # Every fragment must belong to exactly one motion group.
        for fragment_id, owners in fragment_owners.items():
            if not owners:
                raise ValueError(
                    f"Fragment {fragment_id!r} has no motion-group owner"
                )
            if len(owners) > 1:
                raise ValueError(
                    f"Fragment {fragment_id!r} belongs to multiple "
                    f"motion groups: {owners}"
                )

        # Ports must reference an existing group and fragments owned by it.
        for port_id, port in self.ports.items():
            if port.group not in self.motion_groups:
                raise ValueError(
                    f"Port {port_id!r} references unknown motion group "
                    f"{port.group!r}"
                )

            group_members = set(
                self.motion_groups[port.group].members
            )

            for fragment_id in port.fragments:
                if fragment_id not in self.fragments:
                    raise ValueError(
                        f"Port {port_id!r} references unknown fragment "
                        f"{fragment_id!r}"
                    )

                if fragment_id not in group_members:
                    raise ValueError(
                        f"Fragment {fragment_id!r} is not a member of "
                        f"port {port_id!r}'s motion group {port.group!r}"
                    )

        # Interface edges must reference existing ports.
        for edge_id, edge in self.interfaces.items():
            if edge.left_port not in self.ports:
                raise ValueError(
                    f"Interface {edge_id!r} references unknown left port "
                    f"{edge.left_port!r}"
                )

            if edge.right_port not in self.ports:
                raise ValueError(
                    f"Interface {edge_id!r} references unknown right port "
                    f"{edge.right_port!r}"
                )

        # Symmetry orbits must reference valid transform sets and groups.
        group_orbits: dict[str, list[str]] = {
            group_id: [] for group_id in self.motion_groups
        }

        for orbit_id, orbit in self.symmetry.orbits.items():
            if orbit.transform_set not in self.symmetry.transform_sets:
                raise ValueError(
                    f"Orbit {orbit_id!r} references unknown transform set "
                    f"{orbit.transform_set!r}"
                )

            for group_id in orbit.master_groups:
                if group_id not in self.motion_groups:
                    raise ValueError(
                        f"Orbit {orbit_id!r} references unknown motion "
                        f"group {group_id!r}"
                    )
                group_orbits[group_id].append(orbit_id)

            for objective_id in orbit.mobility.objectives:
                if objective_id not in self.objectives:
                    raise ValueError(
                        f"Symmetry orbit {orbit_id!r} mobility references "
                        f"unknown objective {objective_id!r}"
                    )

        # One master group cannot belong to multiple symmetry orbits.
        for group_id, orbit_ids in group_orbits.items():
            if len(orbit_ids) > 1:
                raise ValueError(
                    f"Motion group {group_id!r} belongs to multiple "
                    f"symmetry orbits: {orbit_ids}"
                )

        for edge_id, edge in self.interfaces.items():
            for objective_id in edge.mobility.objectives:
                if objective_id not in self.objectives:
                    raise ValueError(
                        f"Interface {edge_id!r} mobility references unknown "
                        f"objective {objective_id!r}"
                    )

        # Scaffold links must reference existing fragments.
        used_endpoints: set[tuple[str, Terminus]] = set()

        for link_id, link in self.scaffold_links.items():
            endpoints = [
                link.from_endpoint,
                link.to_endpoint,
            ]

            for endpoint in endpoints:
                if endpoint.fragment not in self.fragments:
                    raise ValueError(
                        f"Scaffold link {link_id!r} references unknown "
                        f"fragment {endpoint.fragment!r}"
                    )

                endpoint_key = (
                    endpoint.fragment,
                    endpoint.terminus,
                )

                if endpoint_key in used_endpoints:
                    raise ValueError(
                        f"Scaffold endpoint {endpoint.fragment!r}:"
                        f"{endpoint.terminus.value} is used more than once"
                    )

                used_endpoints.add(endpoint_key)

        for segment_id, segment in self.generated_segments.items():
            if isinstance(segment, ScaffoldLinkSpec):
                endpoints = [
                    segment.from_endpoint,
                    segment.to_endpoint,
                ]
            else:
                endpoints = [segment.anchor]

            for endpoint in endpoints:
                if endpoint.fragment not in self.fragments:
                    raise ValueError(
                        f"Generated segment {segment_id!r} references "
                        f"unknown fragment {endpoint.fragment!r}"
                    )
                endpoint_key = (endpoint.fragment, endpoint.terminus)
                if endpoint_key in used_endpoints:
                    raise ValueError(
                        f"Scaffold endpoint {endpoint.fragment!r}:"
                        f"{endpoint.terminus.value} is used more than once"
                    )
                used_endpoints.add(endpoint_key)

        for group_id in self.initialization:
            if group_id not in self.motion_groups:
                raise ValueError(
                    f"Initialization references unknown motion group "
                    f"{group_id!r}"
                )

        return self


# Backwards-compatible public name used by the original Interface-Seed 2.0
# compiler and existing user configurations.  New code should use the
# topology-neutral ``AssemblySpecification`` name.
InterfaceSeedSpec = AssemblySpecification
