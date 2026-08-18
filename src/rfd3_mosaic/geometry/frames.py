"""Deterministic local-frame construction for interface ports."""

import numpy as np
from numpy.typing import ArrayLike

from rfd3_mosaic.geometry.se3 import FloatArray, make_transform


def _coordinates(value: ArrayLike, *, name: str) -> FloatArray:
    coordinates = np.asarray(value, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"{name} coordinates must have shape (N, 3)")
    if coordinates.shape[0] < 3:
        raise ValueError(f"{name} requires at least three atoms")
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{name} coordinates contain NaN or Inf")
    return coordinates


def _canonical_vector_sign(vector: FloatArray) -> FloatArray:
    """Resolve eigenvector sign using its largest-magnitude component."""

    signed = vector.copy()
    pivot = int(np.argmax(np.abs(signed)))
    if signed[pivot] < 0.0:
        signed *= -1.0
    return signed


def _unit_vector(
    vector: ArrayLike,
    *,
    name: str,
    tolerance: float = 1e-8,
) -> FloatArray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be one finite three-vector")
    norm = float(np.linalg.norm(value))
    if norm <= tolerance:
        raise ValueError(f"{name} has no finite non-zero direction")
    return value / norm


def anchor_interface_frame(
    *,
    origin_coordinates: ArrayLike,
    x_axis_coordinates: ArrayLike,
    xy_plane_coordinates: ArrayLike,
    degeneracy_tolerance: float = 1e-8,
) -> FloatArray:
    """Construct a local frame from explicit, ordered atom anchors.

    The origin is the centroid of ``origin_coordinates``.  The two x-axis
    anchors define the positive x direction and the three plane anchors
    define the ordered positive normal.  The x direction is projected into
    that plane before the right-handed y axis is constructed.  Consequently
    small coordinate noise cannot turn a nominally in-plane x anchor into a
    non-orthogonal frame, while degenerate declarations fail closed.
    """

    origins = np.asarray(origin_coordinates, dtype=np.float64)
    x_points = np.asarray(x_axis_coordinates, dtype=np.float64)
    plane_points = np.asarray(xy_plane_coordinates, dtype=np.float64)
    if origins.ndim != 2 or origins.shape[1:] != (3,) or not len(origins):
        raise ValueError("Anchor-frame origin coordinates must have shape (N, 3)")
    if x_points.shape != (2, 3):
        raise ValueError("Anchor-frame x-axis coordinates must have shape (2, 3)")
    if plane_points.shape != (3, 3):
        raise ValueError("Anchor-frame plane coordinates must have shape (3, 3)")
    if not (
        np.isfinite(origins).all()
        and np.isfinite(x_points).all()
        and np.isfinite(plane_points).all()
    ):
        raise ValueError("Anchor-frame coordinates contain NaN or Inf")

    plane_normal = _unit_vector(
        np.cross(
            plane_points[1] - plane_points[0],
            plane_points[2] - plane_points[0],
        ),
        name="Anchor-frame plane normal",
        tolerance=degeneracy_tolerance,
    )
    raw_x = x_points[1] - x_points[0]
    in_plane_x = raw_x - float(np.dot(raw_x, plane_normal)) * plane_normal
    x_axis = _unit_vector(
        in_plane_x,
        name="Anchor-frame x axis",
        tolerance=degeneracy_tolerance,
    )
    y_axis = _unit_vector(
        np.cross(plane_normal, x_axis),
        name="Anchor-frame y axis",
        tolerance=degeneracy_tolerance,
    )
    z_axis = _unit_vector(
        np.cross(x_axis, y_axis),
        name="Anchor-frame z axis",
        tolerance=degeneracy_tolerance,
    )
    return make_transform(
        np.column_stack((x_axis, y_axis, z_axis)),
        origins.mean(axis=0),
    )


def principal_axis_anchor_frame(
    coordinates: ArrayLike,
    *,
    anchor_coordinate: ArrayLike,
    degeneracy_tolerance: float = 1e-8,
) -> FloatArray:
    """Construct a principal-axis frame with an explicit roll anchor.

    The largest-variance direction is the positive x axis.  ``anchor`` fixes
    the otherwise ambiguous rotation about that axis: its component
    perpendicular to x defines positive y.  This is useful for elongated
    helical ports whose two minor PCA eigenvectors are nearly degenerate.
    """

    points = _coordinates(coordinates, name="Principal-axis port")
    anchor = np.asarray(anchor_coordinate, dtype=np.float64)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError("Principal-axis anchor must be one finite coordinate")
    origin = points.mean(axis=0)
    centered = points - origin
    covariance = centered.T @ centered / float(points.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    if float(eigenvalues[0]) <= degeneracy_tolerance:
        raise ValueError("Principal-axis port has no resolvable extent")
    x_axis = _canonical_vector_sign(eigenvectors[:, order[0]])
    toward_anchor = anchor - origin
    perpendicular = toward_anchor - float(np.dot(toward_anchor, x_axis)) * x_axis
    y_axis = _unit_vector(
        perpendicular,
        name="Principal-axis roll anchor",
        tolerance=degeneracy_tolerance,
    )
    z_axis = _unit_vector(
        np.cross(x_axis, y_axis),
        name="Principal-axis frame normal",
        tolerance=degeneracy_tolerance,
    )
    y_axis = _unit_vector(
        np.cross(z_axis, x_axis),
        name="Principal-axis frame y axis",
        tolerance=degeneracy_tolerance,
    )
    return make_transform(
        np.column_stack((x_axis, y_axis, z_axis)),
        origin,
    )


def reference_interface_pca_frame(
    coordinates: ArrayLike,
    *,
    partner_coordinates: ArrayLike | None = None,
    degeneracy_tolerance: float = 1e-8,
) -> FloatArray:
    """Construct a reproducible right-handed local-to-world interface frame.

    The x axis follows the largest-variance direction and z is the interface
    plane normal.  If partner coordinates are supplied, z points toward the
    partner centroid; otherwise a deterministic component-sign rule is used.
    """

    points = _coordinates(coordinates, name="Interface port")
    origin = points.mean(axis=0)
    centered = points - origin
    covariance = centered.T @ centered / float(points.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    scale = max(float(eigenvalues[0]), 1.0)
    if eigenvalues[1] <= degeneracy_tolerance * scale:
        raise ValueError(
            "Interface coordinates are collinear; PCA frame is undefined"
        )
    if abs(eigenvalues[0] - eigenvalues[1]) <= (
        degeneracy_tolerance * scale
    ):
        raise ValueError(
            "The two principal in-plane axes are degenerate; provide anchors"
        )

    x_axis = _canonical_vector_sign(eigenvectors[:, 0])
    z_axis = eigenvectors[:, 2]
    if partner_coordinates is not None:
        partner = _coordinates(
            partner_coordinates,
            name="Partner port",
        )
        toward_partner = partner.mean(axis=0) - origin
        projection = float(np.dot(z_axis, toward_partner))
        if abs(projection) <= degeneracy_tolerance:
            # A perfectly symmetric oligomeric port can place the aggregate
            # partner centroid in its PCA plane even though the complete
            # reference interface is valid.  The normal sign then carries no
            # partner-facing information, so use the same deterministic gauge
            # as a port without a partner.  The full reference transform still
            # preserves the supplied interface geometry exactly.
            z_axis = _canonical_vector_sign(z_axis)
        elif projection < 0.0:
            z_axis *= -1.0
    else:
        z_axis = _canonical_vector_sign(z_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return make_transform(rotation, origin)
