"""Deterministic symmetry-transform generation and lookup.

The registry uses homogeneous transforms acting on Cartesian coordinates.  A
cyclic transform rotates around an arbitrary axis passing through ``center``;
copy zero is always the identity and transform IDs are stable (``C3:e``,
``C3:r1``, ...).  A dihedral registry contains the proper rotational Dn group:
the n-fold rotations followed by n perpendicular two-fold rotations.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

from rfd3_mosaic.geometry.se3 import (
    FloatArray,
    axis_angle_rotation,
    compose_transforms,
    make_transform,
    validate_transform,
)
from rfd3_mosaic.schema.specs import (
    SymmetryTransformSetSpec,
    SymmetryType,
)


def cyclic_transform_id(order: int, copy_index: int) -> str:
    """Return the canonical transform ID for a cyclic copy index."""

    if order < 2:
        raise ValueError("Cyclic symmetry order must be at least 2")

    normalized_index = copy_index % order
    suffix = "e" if normalized_index == 0 else f"r{normalized_index}"
    return f"C{order}:{suffix}"


@dataclass(frozen=True)
class SymmetryTransformRegistry:
    """Ordered, immutable transform set for one finite symmetry group."""

    group_name: str
    transform_ids: tuple[str, ...]
    transforms: Mapping[str, FloatArray]
    orbit_period: int | None = None

    def __post_init__(self) -> None:
        if len(self.transform_ids) < 2:
            raise ValueError(
                "A symmetry registry requires at least two transforms"
            )
        if len(self.transform_ids) != len(set(self.transform_ids)):
            raise ValueError("Symmetry transform IDs must be unique")
        if set(self.transform_ids) != set(self.transforms):
            raise ValueError("Transform IDs and transform mapping do not match")

        orbit_period = self.orbit_period or len(self.transform_ids)
        if orbit_period < 1 or len(self.transform_ids) % orbit_period != 0:
            raise ValueError(
                "orbit_period must be a positive divisor of registry order"
            )

        validated: dict[str, FloatArray] = {}
        for transform_id in self.transform_ids:
            matrix = validate_transform(self.transforms[transform_id]).copy()
            matrix.setflags(write=False)
            validated[transform_id] = matrix

        object.__setattr__(self, "transforms", MappingProxyType(validated))
        object.__setattr__(self, "orbit_period", orbit_period)

    @property
    def order(self) -> int:
        return len(self.transform_ids)

    @property
    def identity_id(self) -> str:
        return self.transform_ids[0]

    def transform(self, transform_id: str) -> FloatArray:
        """Return a read-only transform by canonical ID."""

        try:
            return self.transforms[transform_id]
        except KeyError as error:
            raise KeyError(
                f"Unknown transform {transform_id!r} in {self.group_name}"
            ) from error

    def transform_id_for_offset(
        self,
        orbit_offset: int,
        *,
        source_copy_index: int = 0,
    ) -> str:
        """Resolve a signed orbit offset relative to a source copy."""

        if source_copy_index < 0 or source_copy_index >= self.order:
            raise IndexError(
                f"Source copy index {source_copy_index} is outside "
                f"registry {self.group_name!r}"
            )
        assert self.orbit_period is not None
        orbit_block, index_in_orbit = divmod(
            source_copy_index,
            self.orbit_period,
        )
        target_index = (
            orbit_block * self.orbit_period
            + (index_in_orbit + orbit_offset) % self.orbit_period
        )
        return self.transform_ids[target_index]

    def transform_for_offset(
        self,
        orbit_offset: int,
        *,
        source_copy_index: int = 0,
    ) -> FloatArray:
        transform_id = self.transform_id_for_offset(
            orbit_offset,
            source_copy_index=source_copy_index,
        )
        return self.transform(transform_id)

    def transform_id_for_relation(
        self,
        relation_id: str,
        *,
        source_copy_index: int = 0,
    ) -> str:
        """Apply a named group relation to a source copy.

        Relations act on the left: ``target = relation @ source``.  This is
        important for non-commutative groups such as Dn, where a two-fold
        relation pairs the two cyclic cosets deterministically.
        """

        if source_copy_index < 0 or source_copy_index >= self.order:
            raise IndexError(
                f"Source copy index {source_copy_index} is outside "
                f"registry {self.group_name!r}"
            )
        source_id = self.transform_ids[source_copy_index]
        return self.compose_ids(relation_id, source_id)

    def compose_ids(self, left_id: str, right_id: str) -> str:
        """Return the registered group element equal to left @ right."""

        composed = compose_transforms(
            self.transform(left_id),
            self.transform(right_id),
        )
        for candidate_id in self.transform_ids:
            if np.allclose(
                composed,
                self.transform(candidate_id),
                atol=1e-6,
            ):
                return candidate_id
        raise ValueError(
            f"Composition of {left_id!r} and {right_id!r} is not in "
            f"registry {self.group_name!r}"
        )


def build_cyclic_registry(
    order: int,
    *,
    axis: ArrayLike = (0.0, 0.0, 1.0),
    center: ArrayLike = (0.0, 0.0, 0.0),
) -> SymmetryTransformRegistry:
    """Build the complete Cn action around an arbitrary axis and center."""

    if order < 2:
        raise ValueError("Cyclic symmetry order must be at least 2")

    axis_vector = np.asarray(axis, dtype=np.float64)
    center_vector = np.asarray(center, dtype=np.float64)
    if center_vector.shape != (3,):
        raise ValueError(
            f"Symmetry center must have shape (3,), got {center_vector.shape}"
        )
    if not np.isfinite(center_vector).all():
        raise ValueError("Symmetry center contains NaN or Inf")

    transform_ids: list[str] = []
    transforms: dict[str, FloatArray] = {}
    for copy_index in range(order):
        transform_id = cyclic_transform_id(order, copy_index)
        angle = 2.0 * np.pi * copy_index / order
        rotation = axis_angle_rotation(axis_vector, angle)
        translation = center_vector - rotation @ center_vector
        transform_ids.append(transform_id)
        transforms[transform_id] = make_transform(rotation, translation)

    return SymmetryTransformRegistry(
        group_name=f"C{order}",
        transform_ids=tuple(transform_ids),
        transforms=transforms,
        orbit_period=order,
    )


def dihedral_transform_id(
    order: int,
    copy_index: int,
) -> str:
    """Return the canonical ID for one element of the proper Dn group."""

    if order < 2:
        raise ValueError("Dihedral symmetry order must be at least 2")
    normalized_index = copy_index % (2 * order)
    if normalized_index == 0:
        return f"D{order}:e"
    if normalized_index < order:
        return f"D{order}:r{normalized_index}"
    return f"D{order}:s{normalized_index - order}"


def _deterministic_perpendicular_axis(axis: FloatArray) -> FloatArray:
    unit_axis = axis / np.linalg.norm(axis)
    basis = np.eye(3, dtype=np.float64)[np.argmin(np.abs(unit_axis))]
    perpendicular = np.cross(unit_axis, basis)
    return perpendicular / np.linalg.norm(perpendicular)


def build_dihedral_registry(
    order: int,
    *,
    axis: ArrayLike = (0.0, 0.0, 1.0),
    secondary_axis: ArrayLike | None = None,
    center: ArrayLike = (0.0, 0.0, 0.0),
) -> SymmetryTransformRegistry:
    """Build the 2n proper rotations of Dn about a common center.

    ``axis`` is the principal n-fold axis.  ``secondary_axis`` is a
    perpendicular two-fold axis; when omitted a deterministic perpendicular
    direction is selected.
    """

    if order < 2:
        raise ValueError("Dihedral symmetry order must be at least 2")

    axis_vector = np.asarray(axis, dtype=np.float64)
    if axis_vector.shape != (3,) or not np.isfinite(axis_vector).all():
        raise ValueError("Dihedral axis must be a finite vector of shape (3,)")
    axis_norm = np.linalg.norm(axis_vector)
    if axis_norm <= 1e-12:
        raise ValueError("Dihedral axis cannot be zero")
    unit_axis = axis_vector / axis_norm

    if secondary_axis is None:
        unit_secondary = _deterministic_perpendicular_axis(axis_vector)
    else:
        secondary_vector = np.asarray(secondary_axis, dtype=np.float64)
        if (
            secondary_vector.shape != (3,)
            or not np.isfinite(secondary_vector).all()
        ):
            raise ValueError(
                "Dihedral secondary axis must be a finite vector of shape (3,)"
            )
        secondary_norm = np.linalg.norm(secondary_vector)
        if secondary_norm <= 1e-12:
            raise ValueError("Dihedral secondary axis cannot be zero")
        unit_secondary = secondary_vector / secondary_norm
        if not np.isclose(
            np.dot(unit_axis, unit_secondary),
            0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "Dihedral secondary axis must be perpendicular to axis"
            )

    center_vector = np.asarray(center, dtype=np.float64)
    if center_vector.shape != (3,):
        raise ValueError(
            f"Symmetry center must have shape (3,), got {center_vector.shape}"
        )
    if not np.isfinite(center_vector).all():
        raise ValueError("Symmetry center contains NaN or Inf")

    rotations = [
        axis_angle_rotation(
            unit_axis,
            2.0 * np.pi * copy_index / order,
        )
        for copy_index in range(order)
    ]
    flip = axis_angle_rotation(unit_secondary, np.pi)
    group_rotations = rotations + [
        flip @ rotation for rotation in rotations
    ]

    transform_ids: list[str] = []
    transforms: dict[str, FloatArray] = {}
    for copy_index, rotation in enumerate(group_rotations):
        transform_id = dihedral_transform_id(order, copy_index)
        translation = center_vector - rotation @ center_vector
        transform_ids.append(transform_id)
        transforms[transform_id] = make_transform(rotation, translation)

    return SymmetryTransformRegistry(
        group_name=f"D{order}",
        transform_ids=tuple(transform_ids),
        transforms=transforms,
        orbit_period=order,
    )


def build_transform_registry(
    spec: SymmetryTransformSetSpec,
) -> SymmetryTransformRegistry:
    """Compile a schema transform-set specification into a registry."""

    if spec.type == SymmetryType.CYCLIC:
        return build_cyclic_registry(
            spec.order,
            axis=spec.axis,
            center=spec.center,
        )

    if spec.type == SymmetryType.DIHEDRAL:
        return build_dihedral_registry(
            spec.order,
            axis=spec.axis,
            secondary_axis=spec.secondary_axis,
            center=spec.center,
        )

    raise NotImplementedError(
        f"Symmetry type {spec.type.value!r} is not implemented yet"
    )


def validate_group_closure(
    registry: SymmetryTransformRegistry,
) -> None:
    """Raise if any pairwise composition leaves the registered group."""

    for left_id in registry.transform_ids:
        for right_id in registry.transform_ids:
            registry.compose_ids(left_id, right_id)
