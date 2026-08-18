"""Topology-neutral morphology descriptors for finite symmetric assemblies.

The generated scaffold validity audit historically checked each chain's
radius of gyration.  That is useful for detecting an extended chain, but it
does not describe the *assembly*: a ring with a very large central opening
can still contain individually compact chains.  This module derives the
finite rotation group's fixed point and axes directly from the compiler's
declared homogeneous transforms, then measures the final structure relative
to that geometry.

No morphology value is a hard quality threshold by default.  In particular,
an intended pore and an unintended hole can have the same diameter.  The
descriptors are therefore explicit diagnostics which downstream design goals
may later constrain without baking a ring-specific assumption into the
scaffold audit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rfd3_mosaic.structure import AtomRecord

_AXIS_COSINE_TOLERANCE = 1.0e-6
_FIXED_POINT_TOLERANCE = 1.0e-5


def _canonical_axis(vector: np.ndarray) -> np.ndarray:
    axis = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12 or not np.isfinite(norm):
        raise ValueError("Rotation axis is undefined")
    axis = axis / norm
    for value in axis:
        if abs(float(value)) <= 1.0e-10:
            continue
        if value < 0.0:
            axis = -axis
        break
    return axis


def _rotation_order(rotation: np.ndarray, *, maximum: int) -> int:
    """Return the finite order of one declared proper rotation."""

    accumulated = np.eye(3, dtype=float)
    for order in range(1, maximum + 1):
        accumulated = rotation @ accumulated
        if np.allclose(accumulated, np.eye(3), atol=1.0e-6):
            return order
    raise ValueError(
        "Declared rotation does not close within the symmetry multiplicity"
    )


def _rotation_axis(rotation: np.ndarray) -> np.ndarray:
    _, _, right = np.linalg.svd(rotation - np.eye(3, dtype=float))
    return _canonical_axis(right[-1])


def _normalize_transforms(
    transforms: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    if len(transforms) < 2:
        raise ValueError("Morphology audit requires at least two transforms")
    normalized: list[np.ndarray] = []
    for index, raw_transform in enumerate(transforms):
        transform = np.asarray(raw_transform, dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(
                f"Symmetry transform {index} must be a finite 4x4 matrix"
            )
        rotation = transform[:3, :3]
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1.0e-6,
        ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
            raise ValueError(
                f"Symmetry transform {index} is not a proper rotation"
            )
        if not np.allclose(
            transform[3],
            np.asarray((0.0, 0.0, 0.0, 1.0)),
            atol=1.0e-6,
        ):
            raise ValueError(
                f"Symmetry transform {index} has an invalid homogeneous row"
            )
        normalized.append(transform)
    return tuple(normalized)


def _common_fixed_point(
    transforms: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, int, float]:
    """Find one point fixed by all rotations and report solution rank.

    A cyclic group fixes an entire line, so its rank is two and the returned
    point is the least-norm point on that line.  Dihedral and polyhedral
    groups normally have rank three and therefore a unique center.
    """

    coefficients: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for transform in transforms:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        if np.allclose(rotation, np.eye(3), atol=1.0e-8) and np.allclose(
            translation,
            0.0,
            atol=1.0e-8,
        ):
            continue
        # R @ center + t = center.
        coefficients.append(np.eye(3) - rotation)
        targets.append(translation)
    if not coefficients:
        raise ValueError("Declared transforms contain no non-identity rotation")
    matrix = np.vstack(coefficients)
    target = np.concatenate(targets)
    center, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    residuals = [
        float(
            np.linalg.norm(
                transform[:3, :3] @ center
                + transform[:3, 3]
                - center
            )
        )
        for transform in transforms
    ]
    maximum_residual = max(residuals, default=0.0)
    if rank < 2 or maximum_residual > _FIXED_POINT_TOLERANCE:
        raise ValueError(
            "Declared symmetry transforms do not share a finite rotation "
            f"axis/center (rank={rank}, residual={maximum_residual:.6g} A)"
        )
    return center, int(rank), maximum_residual


def _unique_rotation_axes(
    transforms: tuple[np.ndarray, ...],
) -> list[dict[str, Any]]:
    axes: list[dict[str, Any]] = []
    maximum_order = len(transforms)
    for transform_index, transform in enumerate(transforms):
        rotation = transform[:3, :3]
        if np.allclose(rotation, np.eye(3), atol=1.0e-8):
            continue
        axis = _rotation_axis(rotation)
        order = _rotation_order(rotation, maximum=maximum_order)
        existing = next(
            (
                record
                for record in axes
                if abs(float(np.dot(record["axis"], axis)))
                >= 1.0 - _AXIS_COSINE_TOLERANCE
            ),
            None,
        )
        if existing is None:
            axes.append(
                {
                    "axis": axis,
                    "fold": order,
                    "transform_indices": [transform_index],
                }
            )
        else:
            existing["fold"] = max(int(existing["fold"]), order)
            existing["transform_indices"].append(transform_index)
    axes.sort(
        key=lambda record: (
            -int(record["fold"]),
            tuple(float(value) for value in record["axis"]),
        )
    )
    return axes


def _selected_coordinates(
    atoms: tuple[AtomRecord, ...],
    *,
    atom_name: str,
) -> np.ndarray:
    selected = [
        np.asarray(atom.coordinate, dtype=float)
        for atom in atoms
        if atom.record_type == "ATOM"
        and atom.atom_name.upper() == atom_name.upper()
    ]
    if not selected:
        raise ValueError(
            f"Assembly morphology audit matched no {atom_name.upper()} atoms"
        )
    coordinates = np.asarray(selected, dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Assembly morphology coordinates contain NaN or Inf")
    return coordinates


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "p05": float(np.percentile(values, 5.0)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "maximum": float(np.max(values)),
    }


def _validated_bounds(
    minimum: float | None,
    maximum: float | None,
    *,
    label: str,
) -> tuple[float | None, float | None]:
    values = [value for value in (minimum, maximum) if value is not None]
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{label} bounds must be finite and non-negative")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{label} minimum cannot exceed maximum")
    return minimum, maximum


def _criterion(
    *,
    metric: str,
    observed: float | None,
    minimum: float | None,
    maximum: float | None,
) -> dict[str, Any]:
    available = observed is not None
    passed = bool(
        available
        and (minimum is None or observed >= minimum)
        and (maximum is None or observed <= maximum)
    )
    return {
        "metric": metric,
        "observed": observed,
        "minimum": minimum,
        "maximum": maximum,
        "available": available,
        "passed": passed,
    }


def audit_assembly_morphology(
    atoms: tuple[AtomRecord, ...],
    *,
    symmetry_transforms: tuple[np.ndarray, ...],
    atom_name: str = "CA",
    minimum_central_pore_diameter: float | None = None,
    maximum_central_pore_diameter: float | None = None,
    minimum_outer_radial_diameter: float | None = None,
    maximum_outer_radial_diameter: float | None = None,
    minimum_spherical_inner_diameter: float | None = None,
    maximum_spherical_inner_diameter: float | None = None,
    minimum_spherical_outer_diameter: float | None = None,
    maximum_spherical_outer_diameter: float | None = None,
) -> dict[str, Any]:
    """Measure a final assembly relative to its declared finite group action.

    For Cn and Dn (n > 2) there is one highest-fold principal axis, so the
    report includes an unambiguous central-pore and outer-radial diameter.
    Polyhedral groups have several equivalent highest-fold axes; their report
    retains every axis and deliberately leaves the singular principal-axis
    fields unset rather than inventing an arbitrary cage orientation.
    """

    bounds = {
        "central_pore_diameter": _validated_bounds(
            minimum_central_pore_diameter,
            maximum_central_pore_diameter,
            label="central pore diameter",
        ),
        "outer_radial_diameter": _validated_bounds(
            minimum_outer_radial_diameter,
            maximum_outer_radial_diameter,
            label="outer radial diameter",
        ),
        "spherical_inner_diameter": _validated_bounds(
            minimum_spherical_inner_diameter,
            maximum_spherical_inner_diameter,
            label="spherical inner diameter",
        ),
        "spherical_outer_diameter": _validated_bounds(
            minimum_spherical_outer_diameter,
            maximum_spherical_outer_diameter,
            label="spherical outer diameter",
        ),
    }
    measurement_only = not any(
        minimum is not None or maximum is not None
        for minimum, maximum in bounds.values()
    )
    threshold_payload = {
        metric: {"minimum": minimum, "maximum": maximum}
        for metric, (minimum, maximum) in bounds.items()
    }
    try:
        transforms = _normalize_transforms(symmetry_transforms)
        coordinates = _selected_coordinates(atoms, atom_name=atom_name)
        center, fixed_point_rank, center_residual = _common_fixed_point(
            transforms
        )
        axes = _unique_rotation_axes(transforms)
        if not axes:
            raise ValueError("Declared transforms contain no rotation axes")
    except ValueError as error:
        return {
            "audit": "rfd3_mosaic.assembly_morphology",
            "schema_version": 1,
            "passed": False,
            "measurement_only": measurement_only,
            "diagnostic_only": measurement_only,
            "summary": None,
            "thresholds": threshold_payload,
            "criteria": [],
            "axes": [],
            "failures": [str(error)],
        }

    offsets = coordinates - center
    center_distances = np.linalg.norm(offsets, axis=1)
    axis_records: list[dict[str, Any]] = []
    for axis_index, record in enumerate(axes):
        axis = np.asarray(record["axis"], dtype=float)
        axial_positions = offsets @ axis
        radial_vectors = offsets - axial_positions[:, None] * axis[None, :]
        radial_distances = np.linalg.norm(radial_vectors, axis=1)
        axis_records.append(
            {
                "axis_index": axis_index,
                "fold": int(record["fold"]),
                "direction": [float(value) for value in axis],
                "transform_indices": list(record["transform_indices"]),
                "radial_clearance": _distribution(radial_distances),
                "central_pore_diameter": 2.0 * float(np.min(radial_distances)),
                "central_pore_diameter_p05": 2.0
                * float(np.percentile(radial_distances, 5.0)),
                "outer_radial_diameter": 2.0 * float(np.max(radial_distances)),
                "outer_radial_diameter_p95": 2.0
                * float(np.percentile(radial_distances, 95.0)),
                "axial_minimum": float(np.min(axial_positions)),
                "axial_maximum": float(np.max(axial_positions)),
                "axial_span": float(np.ptp(axial_positions)),
            }
        )

    highest_fold = max(record["fold"] for record in axis_records)
    highest_fold_axes = [
        record for record in axis_records if record["fold"] == highest_fold
    ]
    principal = highest_fold_axes[0] if len(highest_fold_axes) == 1 else None
    center_distribution = _distribution(center_distances)
    center_is_unique = fixed_point_rank == 3
    summary: dict[str, Any] = {
        "atom_name": atom_name.upper(),
        "selected_atom_count": len(coordinates),
        "symmetry_multiplicity": len(transforms),
        "reference_center": [float(value) for value in center],
        "fixed_point_rank": fixed_point_rank,
        "center_is_unique": center_is_unique,
        "maximum_fixed_point_residual": center_residual,
        "axis_count": len(axis_records),
        "highest_axis_fold": highest_fold,
        "highest_fold_axis_count": len(highest_fold_axes),
        "principal_axis_unique": principal is not None,
        "principal_axis_index": (
            int(principal["axis_index"]) if principal is not None else None
        ),
        # Spherical cage radii require a unique finite-group center.  Cn
        # fixes an entire axis, so only its axis-relative values are
        # invariant and the spherical fields are intentionally unset.
        "minimum_center_distance": (
            center_distribution["minimum"] if center_is_unique else None
        ),
        "median_center_distance": (
            center_distribution["median"] if center_is_unique else None
        ),
        "maximum_center_distance": (
            center_distribution["maximum"] if center_is_unique else None
        ),
        "spherical_inner_diameter": (
            2.0 * center_distribution["minimum"] if center_is_unique else None
        ),
        "spherical_outer_diameter": (
            2.0 * center_distribution["maximum"] if center_is_unique else None
        ),
        "minimum_highest_fold_axis_clearance": min(
            record["radial_clearance"]["minimum"]
            for record in highest_fold_axes
        ),
        "maximum_highest_fold_axis_extent": max(
            record["radial_clearance"]["maximum"]
            for record in highest_fold_axes
        ),
        "central_pore_diameter": (
            principal["central_pore_diameter"]
            if principal is not None
            else None
        ),
        "central_pore_diameter_p05": (
            principal["central_pore_diameter_p05"]
            if principal is not None
            else None
        ),
        "outer_radial_diameter": (
            principal["outer_radial_diameter"]
            if principal is not None
            else None
        ),
        "outer_radial_diameter_p95": (
            principal["outer_radial_diameter_p95"]
            if principal is not None
            else None
        ),
        "principal_axis_direction": (
            principal["direction"] if principal is not None else None
        ),
        "principal_axis_fold": (
            principal["fold"] if principal is not None else None
        ),
    }
    criteria = [
        _criterion(
            metric=metric,
            observed=summary[metric],
            minimum=minimum,
            maximum=maximum,
        )
        for metric, (minimum, maximum) in bounds.items()
        if minimum is not None or maximum is not None
    ]
    failures = [
        (
            f"{criterion['metric']} is unavailable for this symmetry"
            if not criterion["available"]
            else f"{criterion['metric']}={criterion['observed']:.6g} A "
            "is outside the requested range"
        )
        for criterion in criteria
        if not criterion["passed"]
    ]
    return {
        "audit": "rfd3_mosaic.assembly_morphology",
        "schema_version": 1,
        "passed": not failures,
        "measurement_only": measurement_only,
        "diagnostic_only": measurement_only,
        "summary": summary,
        "center_distance": center_distribution,
        "axes": axis_records,
        "thresholds": threshold_payload,
        "criteria": criteria,
        "failures": failures,
        "interpretation": (
            "Morphology values are measurements only unless explicit bounds "
            "were supplied. A large central pore may be intended; Mosaic "
            "does not invent an acceptance target."
        ),
    }


__all__ = ["audit_assembly_morphology"]
