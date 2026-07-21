import unittest

import numpy as np

from rfd3_mosaic.geometry import (
    apply_transform,
    axis_angle_rotation,
    compose_transforms,
    invert_transform,
    make_transform,
    validate_rotation_matrix,
    validate_transform,
)


class SE3TestCase(unittest.TestCase):
    def test_axis_angle_produces_valid_rotation(self) -> None:
        rotation = axis_angle_rotation(
            axis=[0.0, 0.0, 1.0],
            angle_radians=np.pi / 2.0,
        )

        np.testing.assert_allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-7,
        )
        self.assertAlmostEqual(
            np.linalg.det(rotation),
            1.0,
            places=7,
        )

    def test_z_rotation_moves_x_axis_to_y_axis(self) -> None:
        rotation = axis_angle_rotation(
            axis=[0.0, 0.0, 1.0],
            angle_radians=np.pi / 2.0,
        )
        transform = make_transform(
            rotation=rotation,
            translation=[0.0, 0.0, 0.0],
        )

        transformed = apply_transform(
            coordinates=np.array([[1.0, 0.0, 0.0]]),
            transform=transform,
        )

        np.testing.assert_allclose(
            transformed,
            np.array([[0.0, 1.0, 0.0]]),
            atol=1e-7,
        )

    def test_rigid_transform_preserves_pairwise_distances(self) -> None:
        coordinates = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )

        rotation = axis_angle_rotation(
            axis=[1.0, 2.0, 3.0],
            angle_radians=0.8,
        )
        transform = make_transform(
            rotation=rotation,
            translation=[5.0, -3.0, 2.0],
        )

        transformed = apply_transform(
            coordinates,
            transform,
        )

        original_distances = np.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :],
            axis=-1,
        )
        transformed_distances = np.linalg.norm(
            transformed[:, None, :] - transformed[None, :, :],
            axis=-1,
        )

        np.testing.assert_allclose(
            transformed_distances,
            original_distances,
            atol=1e-7,
        )

    def test_inverse_transform_recovers_coordinates(self) -> None:
        coordinates = np.array(
            [
                [1.0, 2.0, 3.0],
                [-4.0, 5.0, 1.0],
            ]
        )

        rotation = axis_angle_rotation(
            axis=[0.5, 1.0, -0.5],
            angle_radians=1.2,
        )
        transform = make_transform(
            rotation=rotation,
            translation=[4.0, -2.0, 7.0],
        )

        transformed = apply_transform(
            coordinates,
            transform,
        )
        recovered = apply_transform(
            transformed,
            invert_transform(transform),
        )

        np.testing.assert_allclose(
            recovered,
            coordinates,
            atol=1e-7,
        )

    def test_transform_composition_order(self) -> None:
        rotation = axis_angle_rotation(
            axis=[0.0, 0.0, 1.0],
            angle_radians=np.pi / 2.0,
        )

        rotate = make_transform(
            rotation=rotation,
            translation=[0.0, 0.0, 0.0],
        )
        translate = make_transform(
            rotation=np.eye(3),
            translation=[1.0, 0.0, 0.0],
        )

        composed = compose_transforms(
            translate,
            rotate,
        )

        transformed = apply_transform(
            np.array([[1.0, 0.0, 0.0]]),
            composed,
        )

        np.testing.assert_allclose(
            transformed,
            np.array([[1.0, 1.0, 0.0]]),
            atol=1e-7,
        )

    def test_reflection_is_not_a_rotation(self) -> None:
        reflection = np.diag([-1.0, 1.0, 1.0])

        with self.assertRaises(ValueError):
            validate_rotation_matrix(reflection)

    def test_zero_rotation_axis_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            axis_angle_rotation(
                axis=[0.0, 0.0, 0.0],
                angle_radians=1.0,
            )

    def test_invalid_homogeneous_last_row_is_rejected(self) -> None:
        transform = np.eye(4)
        transform[3] = [0.0, 0.0, 1.0, 1.0]

        with self.assertRaises(ValueError):
            validate_transform(transform)


if __name__ == "__main__":
    unittest.main()