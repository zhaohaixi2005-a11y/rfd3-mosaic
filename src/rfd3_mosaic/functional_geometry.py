"""Backend-independent evaluation of bound functional geometry."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field

from rfd3_mosaic.geometry.se3 import (
    compose_transforms,
    invert_transform,
    validate_transform,
)
from rfd3_mosaic.schema.functional_geometry import (
    AngleGeometry,
    ChiralityGeometry,
    ChiralitySign,
    CoordinationGeometry,
    CoordinationShape,
    DihedralGeometry,
    DistanceGeometry,
    FunctionalGeometrySpec,
    RelativePoseGeometry,
)
from rfd3_mosaic.schema.specs import StrictModel


FloatArray = NDArray[np.float64]


class FunctionalRelationEvaluation(StrictModel):
    id: str
    kind: str
    passed: bool
    observed: dict[str, object] = Field(default_factory=dict)
    errors: dict[str, float] = Field(default_factory=dict)
    normalized_violation: float


class FunctionalGeometryEvaluation(StrictModel):
    schema_version: int = 1
    name: str
    passed: bool
    evaluated_relations: int
    maximum_normalized_violation: float
    relations: tuple[FunctionalRelationEvaluation, ...]


def _coordinate(value: ArrayLike, *, atom_id: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(
            f"Coordinate for functional atom {atom_id!r} must be finite "
            "with shape (3,)"
        )
    return result


def _angle_deg(first: FloatArray, center: FloatArray, last: FloatArray) -> float:
    left = first - center
    right = last - center
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        raise ValueError("Cannot evaluate an angle with a zero-length arm")
    cosine = np.clip(float(np.dot(left, right)) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _dihedral_deg(
    first: FloatArray,
    second: FloatArray,
    third: FloatArray,
    fourth: FloatArray,
) -> float:
    b0 = second - first
    b1 = third - second
    b2 = fourth - third
    norm = float(np.linalg.norm(b1))
    if norm <= 1e-12:
        raise ValueError("Cannot evaluate a dihedral with a zero central bond")
    unit = b1 / norm
    first_plane = b0 - np.dot(b0, unit) * unit
    second_plane = b2 - np.dot(b2, unit) * unit
    if (
        float(np.linalg.norm(first_plane)) <= 1e-12
        or float(np.linalg.norm(second_plane)) <= 1e-12
    ):
        raise ValueError("Cannot evaluate a dihedral from collinear atoms")
    x = float(np.dot(first_plane, second_plane))
    y = float(np.dot(np.cross(unit, first_plane), second_plane))
    return float(np.degrees(np.arctan2(y, x)))


def _periodic_error_deg(observed: float, target: float) -> float:
    return abs((observed - target + 180.0) % 360.0 - 180.0)


def _rotation_error_deg(transform: FloatArray) -> float:
    rotation = transform[:3, :3]
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _normalized_violation(error: float, tolerance: float) -> float:
    outside = max(error - tolerance, 0.0)
    return outside / max(tolerance, 1e-6)


def _result(
    relation_id: str,
    kind: str,
    *,
    observed: dict[str, object],
    errors: dict[str, float],
    violations: tuple[float, ...],
) -> FunctionalRelationEvaluation:
    maximum = max(violations, default=0.0)
    return FunctionalRelationEvaluation(
        id=relation_id,
        kind=kind,
        passed=maximum <= 1e-12,
        observed=observed,
        errors=errors,
        normalized_violation=maximum,
    )


def _ideal_coordination_angles(shape: CoordinationShape) -> tuple[float, ...]:
    if shape == CoordinationShape.LINEAR:
        return (180.0,)
    if shape == CoordinationShape.TRIGONAL_PLANAR:
        return (120.0,)
    if shape == CoordinationShape.TETRAHEDRAL:
        return (109.47122063449069,)
    if shape in {
        CoordinationShape.SQUARE_PLANAR,
        CoordinationShape.OCTAHEDRAL,
    }:
        return (90.0, 180.0)
    return ()


def _evaluate_relation(
    relation: object,
    coordinates: Mapping[str, FloatArray],
    fragment_transforms: Mapping[str, FloatArray],
) -> FunctionalRelationEvaluation:
    if isinstance(relation, DistanceGeometry):
        observed = float(
            np.linalg.norm(
                coordinates[relation.atoms[0]]
                - coordinates[relation.atoms[1]]
            )
        )
        error = abs(observed - relation.target)
        return _result(
            relation.id,
            relation.kind,
            observed={"distance": observed},
            errors={"absolute_distance_error": error},
            violations=(_normalized_violation(error, relation.tolerance),),
        )
    if isinstance(relation, AngleGeometry):
        observed = _angle_deg(*(coordinates[item] for item in relation.atoms))
        error = abs(observed - relation.target_deg)
        return _result(
            relation.id,
            relation.kind,
            observed={"angle_deg": observed},
            errors={"absolute_angle_error_deg": error},
            violations=(
                _normalized_violation(error, relation.tolerance_deg),
            ),
        )
    if isinstance(relation, DihedralGeometry):
        observed = _dihedral_deg(
            *(coordinates[item] for item in relation.atoms)
        )
        error = _periodic_error_deg(observed, relation.target_deg)
        return _result(
            relation.id,
            relation.kind,
            observed={"dihedral_deg": observed},
            errors={"periodic_dihedral_error_deg": error},
            violations=(
                _normalized_violation(error, relation.tolerance_deg),
            ),
        )
    if isinstance(relation, ChiralityGeometry):
        center, first, second, third = (
            coordinates[item] for item in relation.atoms
        )
        volume = float(
            np.dot(first - center, np.cross(second - center, third - center))
        )
        signed = (
            volume
            if relation.sign == ChiralitySign.POSITIVE
            else -volume
        )
        violation = max(relation.minimum_abs_volume - signed, 0.0)
        normalized = violation / max(relation.minimum_abs_volume, 1.0)
        return _result(
            relation.id,
            relation.kind,
            observed={"signed_volume": volume},
            errors={"required_sign_margin": violation},
            violations=(normalized,),
        )
    if isinstance(relation, RelativePoseGeometry):
        first, second = relation.fragments
        if (
            first not in fragment_transforms
            or second not in fragment_transforms
        ):
            raise ValueError(
                f"Relative-pose relation {relation.id!r} requires bound "
                f"frames for fragments {first!r} and {second!r}"
            )
        # Pose of the second fragment expressed in the first fragment frame.
        observed_transform = compose_transforms(
            invert_transform(fragment_transforms[first]),
            fragment_transforms[second],
        )
        target = validate_transform(relation.target_transform)
        delta = compose_transforms(invert_transform(target), observed_transform)
        translation_error = float(np.linalg.norm(delta[:3, 3]))
        rotation_error = _rotation_error_deg(delta)
        return _result(
            relation.id,
            relation.kind,
            observed={"relative_transform": observed_transform.tolist()},
            errors={
                "translation_error": translation_error,
                "rotation_error_deg": rotation_error,
            },
            violations=(
                _normalized_violation(
                    translation_error,
                    relation.translation_tolerance,
                ),
                _normalized_violation(
                    rotation_error,
                    relation.rotation_tolerance_deg,
                ),
            ),
        )
    if isinstance(relation, CoordinationGeometry):
        center = coordinates[relation.center]
        ligands = [coordinates[item] for item in relation.ligands]
        distances = [float(np.linalg.norm(item - center)) for item in ligands]
        distance_errors = [
            abs(item - relation.distance_target) for item in distances
        ]
        observed_angles: list[float] = []
        angle_errors: list[float] = []
        ideal_angles = _ideal_coordination_angles(relation.shape)
        if ideal_angles:
            for left_index, left in enumerate(ligands):
                for right in ligands[left_index + 1 :]:
                    angle = _angle_deg(left, center, right)
                    observed_angles.append(angle)
                    angle_errors.append(
                        min(abs(angle - target) for target in ideal_angles)
                    )
        violations = tuple(
            _normalized_violation(item, relation.distance_tolerance)
            for item in distance_errors
        ) + tuple(
            _normalized_violation(item, relation.angle_tolerance_deg)
            for item in angle_errors
        )
        return _result(
            relation.id,
            relation.kind,
            observed={
                "distances": distances,
                "pairwise_angles_deg": observed_angles,
                "shape": relation.shape.value,
            },
            errors={
                "maximum_distance_error": max(distance_errors, default=0.0),
                "maximum_angle_error_deg": max(angle_errors, default=0.0),
            },
            violations=violations,
        )
    raise TypeError(f"Unsupported functional relation {type(relation)!r}")


def evaluate_functional_geometry(
    specification: FunctionalGeometrySpec,
    atom_coordinates: Mapping[str, ArrayLike],
    *,
    fragment_transforms: Mapping[str, ArrayLike] | None = None,
) -> FunctionalGeometryEvaluation:
    """Evaluate an already-bound local functional constraint hypergraph."""

    required_atoms = {item.id for item in specification.atoms}
    missing = sorted(required_atoms - set(atom_coordinates))
    if missing:
        raise ValueError(f"Missing functional atom coordinates: {missing}")
    coordinates = {
        atom_id: _coordinate(atom_coordinates[atom_id], atom_id=atom_id)
        for atom_id in required_atoms
    }
    transforms = {
        fragment_id: validate_transform(transform)
        for fragment_id, transform in (fragment_transforms or {}).items()
    }
    evaluations = tuple(
        _evaluate_relation(relation, coordinates, transforms)
        for relation in specification.relations
    )
    maximum = max(
        (item.normalized_violation for item in evaluations),
        default=0.0,
    )
    return FunctionalGeometryEvaluation(
        name=specification.name,
        passed=all(item.passed for item in evaluations),
        evaluated_relations=len(evaluations),
        maximum_normalized_violation=maximum,
        relations=evaluations,
    )


__all__ = [
    "FunctionalGeometryEvaluation",
    "FunctionalRelationEvaluation",
    "evaluate_functional_geometry",
]
