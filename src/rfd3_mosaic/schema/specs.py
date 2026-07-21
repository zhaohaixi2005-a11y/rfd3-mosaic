from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


Identifier = Annotated[
    str,
    Field(
        min_length=1,
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$",
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