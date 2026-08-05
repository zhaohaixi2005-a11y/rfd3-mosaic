"""Internal compatibility checks between one SE(3) relation and Cn actions.

This module deliberately does not infer a complete assembly architecture.
One pairwise relation can establish a finite cyclic subgroup while leaving
additional copies, component ownership, connectivity and scaffold feasibility
unobserved.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field, field_validator

from rfd3_mosaic.geometry.se3 import validate_transform
from rfd3_mosaic.schema.specs import StrictModel


FloatArray = NDArray[np.float64]


class CyclicRelationSearchSpace(StrictModel):
    """Finite Cn domain and fail-closed pairwise-relation gates."""

    orders: tuple[int, ...] = tuple(range(2, 9))
    max_angle_error_deg: float = Field(default=2.0, ge=0.0)
    max_screw_translation: float = Field(default=0.25, ge=0.0)
    max_closure_rotation_deg: float = Field(default=2.0, ge=0.0)
    max_closure_translation: float = Field(default=0.5, ge=0.0)

    @field_validator("orders")
    @classmethod
    def validate_orders(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("Cyclic relation search requires at least one order")
        if any(order < 2 for order in value):
            raise ValueError("Cyclic relation orders must be at least C2")
        if len(value) != len(set(value)):
            raise ValueError("Cyclic relation orders cannot repeat")
        return tuple(sorted(value))


class RelationCompatibilityHypothesis(StrictModel):
    """One full-group interpretation compatible with one observed relation."""

    id: str
    proposed_group: str
    proposed_group_order: int
    equivalent_orbit_offsets: tuple[int, ...]
    observed_relation_subgroup_order: int
    subgroup_index_in_proposed_group: int
    unobserved_cosets: int
    relation_generates_complete_group: bool
    axis: tuple[float, float, float]
    axis_point: tuple[float, float, float]
    measured_principal_angle_deg: float
    expected_principal_angle_deg: float
    angle_error_deg: float
    screw_translation: float
    axis_line_fit_residual: float
    closure_rotation_error_deg: float
    closure_translation_error: float
    relation_compatibility_score: float
    evidence_scope: str = "single_pairwise_relative_transform"
    unresolved_evidence: tuple[str, ...] = (
        "functional_atom_identity",
        "component_ownership",
        "connectivity",
        "whole_assembly_clashes",
        "scaffold_generation_feasibility",
        "sequence_designability",
    )


class CyclicRelationCompatibilityReport(StrictModel):
    schema_version: int = 1
    search_space: CyclicRelationSearchSpace
    evaluated_offset_classes: int
    compatible_relations: tuple[RelationCompatibilityHypothesis, ...]
    interpretation_note: str


def _principal_rotation_angle(rotation: FloatArray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _canonical_axis(rotation: FloatArray) -> FloatArray:
    eigenvalues, eigenvectors = np.linalg.eig(rotation)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    axis = np.real(eigenvectors[:, index]).astype(np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-10:
        raise ValueError("Cannot recover a finite rotation axis")
    axis /= norm
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0.0:
        axis *= -1.0
    return axis


def _rotation_axis_line(
    transform: FloatArray,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Fit the closest point on the axis and expose point-group violations."""

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    axis = _canonical_axis(rotation)
    system = np.eye(3, dtype=np.float64) - rotation
    axis_point, *_ = np.linalg.lstsq(system, translation, rcond=None)
    predicted = system @ axis_point
    residual = translation - predicted
    screw = abs(float(np.dot(residual, axis)))
    fit_residual = float(np.linalg.norm(residual))
    # Position along the rotation axis is gauge freedom. Report the unique
    # point on that line nearest the global origin.
    axis_point = axis_point - np.dot(axis_point, axis) * axis
    return axis, axis_point, screw, fit_residual


def _expected_principal_angle(order: int, offset: int) -> float:
    directed = (2.0 * np.pi * offset / order) % (2.0 * np.pi)
    return float(min(directed, 2.0 * np.pi - directed))


def _equivalent_offsets(order: int, offset: int) -> tuple[int, ...]:
    inverse = (-offset) % order
    return tuple(sorted({offset, inverse}))


def _normalized(value: float, maximum: float) -> float:
    if maximum == 0.0:
        return 0.0 if value <= 1e-12 else float("inf")
    return value / maximum


def enumerate_cyclic_relation_compatibility(
    relative_transform: ArrayLike,
    search_space: CyclicRelationSearchSpace | None = None,
) -> CyclicRelationCompatibilityReport:
    """Enumerate Cn group-element classes compatible with one relation.

    Orbit offsets related by inversion are collapsed because reversing the
    axis direction or swapping source and target fragments does not provide a
    distinct full-assembly interpretation. A 120-degree relation therefore
    supports a complete C3 generator and a three-element subgroup of C6; it
    does not distinguish which full assembly actually exists.
    """

    transform = validate_transform(relative_transform)
    space = search_space or CyclicRelationSearchSpace()
    measured_angle = _principal_rotation_angle(transform[:3, :3])
    if measured_angle <= 1e-8:
        raise ValueError(
            "Identity-like relations do not determine a cyclic subgroup"
        )
    axis, axis_point, screw, fit_residual = _rotation_axis_line(transform)
    hypotheses: list[RelationCompatibilityHypothesis] = []
    evaluated = 0
    for order in space.orders:
        seen_classes: set[tuple[int, ...]] = set()
        for offset in range(1, order):
            offset_class = _equivalent_offsets(order, offset)
            if offset_class in seen_classes:
                continue
            seen_classes.add(offset_class)
            evaluated += 1
            expected_angle = _expected_principal_angle(order, offset)
            angle_error_deg = float(
                np.degrees(abs(measured_angle - expected_angle))
            )
            subgroup_order = order // gcd(order, offset)
            subgroup_index = order // subgroup_order
            closure = np.linalg.matrix_power(transform, subgroup_order)
            closure_rotation_deg = float(
                np.degrees(_principal_rotation_angle(closure[:3, :3]))
            )
            closure_translation = float(np.linalg.norm(closure[:3, 3]))
            if (
                angle_error_deg > space.max_angle_error_deg
                or screw > space.max_screw_translation
                or closure_rotation_deg > space.max_closure_rotation_deg
                or closure_translation > space.max_closure_translation
            ):
                continue
            score = sum(
                (
                    _normalized(angle_error_deg, space.max_angle_error_deg),
                    _normalized(screw, space.max_screw_translation),
                    _normalized(
                        closure_rotation_deg,
                        space.max_closure_rotation_deg,
                    ),
                    _normalized(
                        closure_translation,
                        space.max_closure_translation,
                    ),
                )
            )
            offsets_text = "|".join(f"k{item}" for item in offset_class)
            hypotheses.append(
                RelationCompatibilityHypothesis(
                    id=f"C{order}:{offsets_text}",
                    proposed_group=f"C{order}",
                    proposed_group_order=order,
                    equivalent_orbit_offsets=offset_class,
                    observed_relation_subgroup_order=subgroup_order,
                    subgroup_index_in_proposed_group=subgroup_index,
                    unobserved_cosets=subgroup_index - 1,
                    relation_generates_complete_group=subgroup_index == 1,
                    axis=tuple(float(value) for value in axis),
                    axis_point=tuple(float(value) for value in axis_point),
                    measured_principal_angle_deg=float(
                        np.degrees(measured_angle)
                    ),
                    expected_principal_angle_deg=float(
                        np.degrees(expected_angle)
                    ),
                    angle_error_deg=angle_error_deg,
                    screw_translation=screw,
                    axis_line_fit_residual=fit_residual,
                    closure_rotation_error_deg=closure_rotation_deg,
                    closure_translation_error=closure_translation,
                    relation_compatibility_score=score,
                )
            )
    hypotheses.sort(
        key=lambda item: (
            item.relation_compatibility_score,
            item.unobserved_cosets,
            item.proposed_group_order,
            item.equivalent_orbit_offsets,
        )
    )
    return CyclicRelationCompatibilityReport(
        search_space=space,
        evaluated_offset_classes=evaluated,
        compatible_relations=tuple(hypotheses),
        interpretation_note=(
            "These are pairwise relation compatibilities, not inferred "
            "assembly architectures. Additional functional, ownership, "
            "topology, clash and generation evidence is unresolved."
        ),
    )


__all__ = [
    "CyclicRelationCompatibilityReport",
    "CyclicRelationSearchSpace",
    "RelationCompatibilityHypothesis",
    "enumerate_cyclic_relation_compatibility",
]
