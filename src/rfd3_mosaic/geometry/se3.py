import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def validate_rotation_matrix(
    rotation: ArrayLike,
    *,
    atol: float = 1e-6,
) -> FloatArray:
    """Validate and return a proper 3x3 rotation matrix."""

    matrix = np.asarray(rotation, dtype=np.float64)

    if matrix.shape != (3, 3):
        raise ValueError(
            f"Rotation matrix must have shape (3, 3), got {matrix.shape}"
        )

    if not np.isfinite(matrix).all():
        raise ValueError("Rotation matrix contains NaN or Inf")

    if not np.allclose(
        matrix.T @ matrix,
        np.eye(3),
        atol=atol,
    ):
        raise ValueError("Rotation matrix is not orthogonal")

    determinant = np.linalg.det(matrix)

    if not np.isclose(determinant, 1.0, atol=atol):
        raise ValueError(
            "Rotation matrix must have determinant +1, "
            f"got {determinant}"
        )

    return matrix


def validate_transform(
    transform: ArrayLike,
    *,
    atol: float = 1e-6,
) -> FloatArray:
    """Validate and return a homogeneous 4x4 SE(3) transform."""

    matrix = np.asarray(transform, dtype=np.float64)

    if matrix.shape != (4, 4):
        raise ValueError(
            f"SE(3) transform must have shape (4, 4), got {matrix.shape}"
        )

    if not np.isfinite(matrix).all():
        raise ValueError("SE(3) transform contains NaN or Inf")

    expected_last_row = np.array(
        [0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    if not np.allclose(
        matrix[3],
        expected_last_row,
        atol=atol,
    ):
        raise ValueError(
            "SE(3) transform must end with [0, 0, 0, 1]"
        )

    validate_rotation_matrix(matrix[:3, :3], atol=atol)

    return matrix


def make_transform(
    rotation: ArrayLike,
    translation: ArrayLike,
) -> FloatArray:
    """Construct a homogeneous transform from rotation and translation."""

    rotation_matrix = validate_rotation_matrix(rotation)
    translation_vector = np.asarray(
        translation,
        dtype=np.float64,
    )

    if translation_vector.shape != (3,):
        raise ValueError(
            "Translation must have shape (3,), "
            f"got {translation_vector.shape}"
        )

    if not np.isfinite(translation_vector).all():
        raise ValueError("Translation contains NaN or Inf")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation_vector

    return transform


def invert_transform(transform: ArrayLike) -> FloatArray:
    """Invert an SE(3) transform analytically."""

    matrix = validate_transform(transform)

    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]

    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)

    return inverse


def compose_transforms(
    left: ArrayLike,
    right: ArrayLike,
) -> FloatArray:
    """Compose transforms, applying right first and then left."""

    left_matrix = validate_transform(left)
    right_matrix = validate_transform(right)

    composed = left_matrix @ right_matrix

    return validate_transform(composed)


def apply_transform(
    coordinates: ArrayLike,
    transform: ArrayLike,
) -> FloatArray:
    """Apply an SE(3) transform to coordinates shaped (..., 3)."""

    points = np.asarray(coordinates, dtype=np.float64)
    matrix = validate_transform(transform)

    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError(
            "Coordinates must have shape (..., 3), "
            f"got {points.shape}"
        )

    if not np.isfinite(points).all():
        raise ValueError("Coordinates contain NaN or Inf")

    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]

    return points @ rotation.T + translation


def axis_angle_rotation(
    axis: ArrayLike,
    angle_radians: float,
) -> FloatArray:
    """Construct a rotation matrix using Rodrigues' formula."""

    axis_vector = np.asarray(axis, dtype=np.float64)

    if axis_vector.shape != (3,):
        raise ValueError(
            f"Axis must have shape (3,), got {axis_vector.shape}"
        )

    if not np.isfinite(axis_vector).all():
        raise ValueError("Axis contains NaN or Inf")

    if not np.isfinite(angle_radians):
        raise ValueError("Angle must be finite")

    norm = np.linalg.norm(axis_vector)

    if norm <= 1e-12:
        raise ValueError("Rotation axis cannot be zero")

    x, y, z = axis_vector / norm
    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)
    one_minus_cosine = 1.0 - cosine

    rotation = np.array(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=np.float64,
    )

    return validate_rotation_matrix(rotation)