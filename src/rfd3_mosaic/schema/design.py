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
from pydantic import Field, field_validator, model_validator

from rfd3_mosaic.schema.specs import Identifier, LinkLengthSpec, StrictModel


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


class CylindricalDegreeOfFreedom(str, Enum):
    RADIUS = "radius"
    AZIMUTH = "azimuth"
    AXIAL = "axial"


class FixedComponentPoseSpec(StrictModel):
    """Choose whether one rigid geometry component may move as a whole."""

    mode: Literal["fixed", "bounded_mobile"] = "fixed"
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
        if self.mode == "fixed" and any(value is not None for value in bounds):
            raise ValueError(
                "pose.mode=fixed cannot define translation/rotation bounds"
            )
        if self.mode == "bounded_mobile" and any(
            value is None for value in bounds
        ):
            raise ValueError(
                "pose.mode=bounded_mobile requires max_translation and "
                "max_rotation_deg"
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
    """Optional explicit frame for a simple Cn/Dn public declaration."""

    id: SymmetryName
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    secondary_axis: tuple[float, float, float] | None = None

    @model_validator(mode="after")
    def validate_axes(self) -> "UserSymmetrySpec":
        if sum(value * value for value in self.axis) <= 1e-12:
            raise ValueError("symmetry axis cannot be zero")
        if self.id.startswith("C") and self.secondary_axis is not None:
            raise ValueError("secondary_axis is only valid for Dn symmetry")
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
    timesteps: Annotated[int, Field(ge=2, le=200)] = 200
    seed: Annotated[int, Field(ge=0)] = 42
    preset: Literal["exact_mosaic"] = "exact_mosaic"
    low_memory_mode: bool = True
    execution_backend: Literal[
        "explicit_all_copy",
        "local_neighbourhood",
    ] = "explicit_all_copy"
    neighbour_radius: Annotated[int, Field(ge=0)] = 1


class UserResourceSpec(StrictModel):
    profile: str | Path = "p100"
    walltime: str | None = None
    memory: str | None = None
    cpus: Annotated[int, Field(ge=1)] | None = None
    partition: str | None = None


class UserOutputSpec(StrictModel):
    root: Path
    campaign: Identifier = "rfd3-mosaic"


class UserDesignSpec(StrictModel):
    """Stable public design declaration before assembly-IR lowering."""

    schema_version: Literal[1] = 1
    name: Identifier
    input: Path
    symmetry: SymmetryRequest
    generation: tuple[GenerationClause, ...] = ()
    constraints: tuple[ConstraintClause, ...] = ()
    sampling: UserSamplingSpec = Field(default_factory=UserSamplingSpec)
    resources: UserResourceSpec = Field(default_factory=UserResourceSpec)
    output: UserOutputSpec | None = None


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
