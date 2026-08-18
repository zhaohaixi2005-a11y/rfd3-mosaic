"""Local functional geometry before symmetry or topology is prescribed.

The schema describes chemical/geometric intent only. It deliberately does
not choose a global symmetry group, chain partition, linker topology, runtime
backend or Slurm profile. Structure binding and residual evaluation are
separate fail-closed stages.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from rfd3_mosaic.geometry.se3 import validate_transform
from rfd3_mosaic.schema.specs import Identifier, StrictModel

Selector = Annotated[str, Field(min_length=1)]
Nonnegative = Annotated[float, Field(ge=0.0)]


class FunctionalFragmentMotion(str, Enum):
    RIGID = "rigid"
    SOFT_RIGID = "soft_rigid"


class FunctionalFragmentSpec(StrictModel):
    """One chemically meaningful atom group with optional unknown ownership."""

    id: Identifier
    selector: Selector
    motion: FunctionalFragmentMotion = FunctionalFragmentMotion.RIGID
    maximum_internal_rmsd: Nonnegative = 0.0
    component: Identifier | None = None
    subunit: Identifier | None = None

    @model_validator(mode="after")
    def validate_motion_bound(self) -> "FunctionalFragmentSpec":
        if (
            self.motion == FunctionalFragmentMotion.RIGID
            and self.maximum_internal_rmsd != 0.0
        ):
            raise ValueError(
                "rigid functional fragments require maximum_internal_rmsd=0"
            )
        if (
            self.motion == FunctionalFragmentMotion.SOFT_RIGID
            and self.maximum_internal_rmsd <= 0.0
        ):
            raise ValueError(
                "soft_rigid functional fragments require a positive "
                "maximum_internal_rmsd"
            )
        return self


class FunctionalAtomSpec(StrictModel):
    """Stable symbolic atom reference resolved later against one structure."""

    id: Identifier
    fragment: Identifier
    selector: Selector
    element: Annotated[str, Field(pattern=r"^[A-Z][a-z]?$")] | None = None
    role: str | None = None


class DistanceGeometry(StrictModel):
    kind: Literal["distance"] = "distance"
    id: Identifier
    atoms: tuple[Identifier, Identifier]
    target: Annotated[float, Field(gt=0.0)]
    tolerance: Nonnegative = 0.25

    @field_validator("atoms")
    @classmethod
    def reject_repeated_atoms(
        cls,
        value: tuple[Identifier, Identifier],
    ) -> tuple[Identifier, Identifier]:
        if value[0] == value[1]:
            raise ValueError("distance geometry requires two different atoms")
        return value


class AngleGeometry(StrictModel):
    kind: Literal["angle"] = "angle"
    id: Identifier
    atoms: tuple[Identifier, Identifier, Identifier]
    target_deg: Annotated[float, Field(ge=0.0, le=180.0)]
    tolerance_deg: Annotated[float, Field(ge=0.0, le=180.0)] = 5.0

    @field_validator("atoms")
    @classmethod
    def reject_repeated_atoms(
        cls,
        value: tuple[Identifier, Identifier, Identifier],
    ) -> tuple[Identifier, Identifier, Identifier]:
        if len(set(value)) != 3:
            raise ValueError("angle geometry requires three different atoms")
        return value


class DihedralGeometry(StrictModel):
    kind: Literal["dihedral"] = "dihedral"
    id: Identifier
    atoms: tuple[Identifier, Identifier, Identifier, Identifier]
    target_deg: Annotated[float, Field(ge=-180.0, le=180.0)]
    tolerance_deg: Annotated[float, Field(ge=0.0, le=180.0)] = 10.0

    @field_validator("atoms")
    @classmethod
    def reject_repeated_atoms(
        cls,
        value: tuple[Identifier, Identifier, Identifier, Identifier],
    ) -> tuple[Identifier, Identifier, Identifier, Identifier]:
        if len(set(value)) != 4:
            raise ValueError(
                "dihedral geometry requires four different atoms"
            )
        return value


class ChiralitySign(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class ChiralityGeometry(StrictModel):
    kind: Literal["chirality"] = "chirality"
    id: Identifier
    # Ordered as center, first, second, third substituent. The sign applies to
    # dot((first-center), cross(second-center, third-center)).
    atoms: tuple[Identifier, Identifier, Identifier, Identifier]
    sign: ChiralitySign
    minimum_abs_volume: Nonnegative = 0.0

    @field_validator("atoms")
    @classmethod
    def reject_repeated_atoms(
        cls,
        value: tuple[Identifier, Identifier, Identifier, Identifier],
    ) -> tuple[Identifier, Identifier, Identifier, Identifier]:
        if len(set(value)) != 4:
            raise ValueError(
                "chirality geometry requires four different atoms"
            )
        return value


class RelativePoseGeometry(StrictModel):
    kind: Literal["relative_pose"] = "relative_pose"
    id: Identifier
    fragments: tuple[Identifier, Identifier]
    target_transform: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    translation_tolerance: Nonnegative = 0.5
    rotation_tolerance_deg: Annotated[
        float,
        Field(ge=0.0, le=180.0),
    ] = 5.0

    @model_validator(mode="after")
    def validate_pose(self) -> "RelativePoseGeometry":
        if self.fragments[0] == self.fragments[1]:
            raise ValueError("relative_pose requires two different fragments")
        validate_transform(self.target_transform)
        return self


class CoordinationShape(str, Enum):
    LINEAR = "linear"
    TRIGONAL_PLANAR = "trigonal_planar"
    TETRAHEDRAL = "tetrahedral"
    SQUARE_PLANAR = "square_planar"
    OCTAHEDRAL = "octahedral"
    CUSTOM = "custom"


class CoordinationGeometry(StrictModel):
    """True multi-atom hyperedge, not decomposed into unrelated contacts."""

    kind: Literal["coordination"] = "coordination"
    id: Identifier
    center: Identifier
    ligands: Annotated[tuple[Identifier, ...], Field(min_length=2)]
    shape: CoordinationShape
    distance_target: Annotated[float, Field(gt=0.0)]
    distance_tolerance: Nonnegative = 0.25
    angle_tolerance_deg: Annotated[
        float,
        Field(ge=0.0, le=180.0),
    ] = 10.0
    require_cross_fragment: bool = True

    @model_validator(mode="after")
    def validate_members(self) -> "CoordinationGeometry":
        members = (self.center, *self.ligands)
        if len(members) != len(set(members)):
            raise ValueError("coordination members must be unique")
        expected_ligands = {
            CoordinationShape.LINEAR: 2,
            CoordinationShape.TRIGONAL_PLANAR: 3,
            CoordinationShape.TETRAHEDRAL: 4,
            CoordinationShape.SQUARE_PLANAR: 4,
            CoordinationShape.OCTAHEDRAL: 6,
        }
        expected = expected_ligands.get(self.shape)
        if expected is not None and len(self.ligands) != expected:
            raise ValueError(
                f"{self.shape.value} coordination requires exactly "
                f"{expected} ligands"
            )
        return self


FunctionalGeometryRelation = Annotated[
    DistanceGeometry
    | AngleGeometry
    | DihedralGeometry
    | ChiralityGeometry
    | RelativePoseGeometry
    | CoordinationGeometry,
    Field(discriminator="kind"),
]


class FunctionalGeometrySpec(StrictModel):
    """A local functional constraint hypergraph independent of architecture."""

    schema_version: Literal[1] = 1
    name: Identifier
    input: Path
    fragments: Annotated[
        tuple[FunctionalFragmentSpec, ...],
        Field(min_length=1),
    ]
    atoms: Annotated[tuple[FunctionalAtomSpec, ...], Field(min_length=1)]
    relations: Annotated[
        tuple[FunctionalGeometryRelation, ...],
        Field(min_length=1),
    ]

    @model_validator(mode="after")
    def validate_graph(self) -> "FunctionalGeometrySpec":
        fragments = {item.id: item for item in self.fragments}
        if len(fragments) != len(self.fragments):
            raise ValueError("functional fragment IDs must be unique")
        atoms = {item.id: item for item in self.atoms}
        if len(atoms) != len(self.atoms):
            raise ValueError("functional atom IDs must be unique")
        relation_ids = {item.id for item in self.relations}
        if len(relation_ids) != len(self.relations):
            raise ValueError("functional relation IDs must be unique")
        for atom in self.atoms:
            if atom.fragment not in fragments:
                raise ValueError(
                    f"Functional atom {atom.id!r} references unknown "
                    f"fragment {atom.fragment!r}"
                )
        for relation in self.relations:
            if isinstance(relation, RelativePoseGeometry):
                unknown = set(relation.fragments) - set(fragments)
                if unknown:
                    raise ValueError(
                        f"Relation {relation.id!r} references unknown "
                        f"fragments {sorted(unknown)}"
                    )
                continue
            references = (
                (relation.center, *relation.ligands)
                if isinstance(relation, CoordinationGeometry)
                else relation.atoms
            )
            unknown = set(references) - set(atoms)
            if unknown:
                raise ValueError(
                    f"Relation {relation.id!r} references unknown atoms "
                    f"{sorted(unknown)}"
                )
            if (
                isinstance(relation, CoordinationGeometry)
                and relation.require_cross_fragment
            ):
                member_fragments = {
                    atoms[atom_id].fragment for atom_id in references
                }
                if len(member_fragments) < 2:
                    raise ValueError(
                        f"Coordination relation {relation.id!r} must span "
                        "at least two functional fragments"
                    )
        return self


def load_functional_geometry(path: str | Path) -> FunctionalGeometrySpec:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Functional geometry does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Functional geometry must contain a YAML mapping")
    specification = FunctionalGeometrySpec.model_validate(payload)
    structure = specification.input.expanduser()
    if not structure.is_absolute():
        structure = source.parent / structure
    structure = structure.resolve()
    if not structure.is_file():
        raise FileNotFoundError(
            f"Functional geometry input does not exist: {structure}"
        )
    return specification.model_copy(update={"input": structure})


__all__ = [
    "AngleGeometry",
    "ChiralityGeometry",
    "ChiralitySign",
    "CoordinationGeometry",
    "CoordinationShape",
    "DihedralGeometry",
    "DistanceGeometry",
    "FunctionalAtomSpec",
    "FunctionalFragmentMotion",
    "FunctionalFragmentSpec",
    "FunctionalGeometryRelation",
    "FunctionalGeometrySpec",
    "RelativePoseGeometry",
    "load_functional_geometry",
]
