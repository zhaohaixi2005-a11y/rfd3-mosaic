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
