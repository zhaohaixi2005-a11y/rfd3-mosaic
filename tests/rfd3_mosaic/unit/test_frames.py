import unittest

import numpy as np

from rfd3_mosaic.geometry import (
    anchor_interface_frame,
    principal_axis_anchor_frame,
    reference_interface_pca_frame,
)


PORT_COORDINATES = np.array(
    [
        [-3.0, -1.0, 0.0],
        [-2.0, 1.0, 0.1],
        [0.0, -0.5, -0.1],
        [2.0, 1.0, 0.0],
        [4.0, -1.0, 0.05],
    ]
)


class InterfaceFrameTestCase(unittest.TestCase):
    def test_explicit_anchor_frame_uses_declared_directions(self) -> None:
        frame = anchor_interface_frame(
            origin_coordinates=np.array([[1.0, 2.0, 3.0]]),
            x_axis_coordinates=np.array(
                [[1.0, 2.0, 3.0], [3.0, 2.0, 3.2]]
            ),
            xy_plane_coordinates=np.array(
                [
                    [1.0, 2.0, 3.0],
                    [3.0, 2.0, 3.0],
                    [1.0, 4.0, 3.0],
                ]
            ),
        )

        np.testing.assert_allclose(frame[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(frame[:3, 0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(frame[:3, 1], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(frame[:3, 2], [0.0, 0.0, 1.0])
        self.assertAlmostEqual(np.linalg.det(frame[:3, :3]), 1.0)

    def test_explicit_anchor_frame_rejects_collinear_plane(self) -> None:
        with self.assertRaisesRegex(ValueError, "plane normal"):
            anchor_interface_frame(
                origin_coordinates=np.array([[0.0, 0.0, 0.0]]),
                x_axis_coordinates=np.array(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
                ),
                xy_plane_coordinates=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                    ]
                ),
            )

    def test_principal_axis_anchor_resolves_elongated_port_roll(self) -> None:
        points = np.array(
            [
                [-3.0, -0.2, 0.0],
                [-1.0, 0.2, 0.0],
                [1.0, -0.2, 0.0],
                [3.0, 0.2, 0.0],
            ]
        )
        frame = principal_axis_anchor_frame(
            points,
            anchor_coordinate=np.array([0.0, 5.0, 0.0]),
        )

        self.assertGreater(float(np.dot(frame[:3, 0], [1.0, 0.0, 0.0])), 0.99)
        self.assertGreater(float(np.dot(frame[:3, 1], [0.0, 1.0, 0.0])), 0.99)
        np.testing.assert_allclose(
            frame[:3, :3].T @ frame[:3, :3],
            np.eye(3),
            atol=1e-7,
        )

    def test_pca_frame_is_right_handed_and_centered(self) -> None:
        frame = reference_interface_pca_frame(PORT_COORDINATES)
        rotation = frame[:3, :3]

        np.testing.assert_allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-7,
        )
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=7)
        np.testing.assert_allclose(
            frame[:3, 3],
            PORT_COORDINATES.mean(axis=0),
            atol=1e-7,
        )

    def test_frame_is_invariant_to_atom_order(self) -> None:
        forward = reference_interface_pca_frame(PORT_COORDINATES)
        reverse = reference_interface_pca_frame(PORT_COORDINATES[::-1])

        np.testing.assert_allclose(forward, reverse, atol=1e-7)

    def test_normal_points_toward_partner(self) -> None:
        partner = PORT_COORDINATES + np.array([0.0, 0.0, 10.0])

        frame = reference_interface_pca_frame(
            PORT_COORDINATES,
            partner_coordinates=partner,
        )

        self.assertGreater(np.dot(frame[:3, 2], [0.0, 0.0, 1.0]), 0.0)

    def test_in_plane_partner_uses_deterministic_normal_gauge(self) -> None:
        planar = np.array(
            [
                [-4.0, -1.0, 0.0],
                [-2.0, 2.0, 0.0],
                [0.0, -1.5, 0.0],
                [3.0, 1.0, 0.0],
                [5.0, -0.5, 0.0],
            ]
        )
        partner = planar + np.array([10.0, 0.0, 0.0])

        forward = reference_interface_pca_frame(
            planar,
            partner_coordinates=partner,
        )
        reverse = reference_interface_pca_frame(
            planar[::-1],
            partner_coordinates=partner[::-1],
        )

        np.testing.assert_allclose(forward, reverse, atol=1e-7)
        self.assertGreaterEqual(
            forward[np.argmax(np.abs(forward[:3, 2])), 2],
            0.0,
        )

    def test_collinear_coordinates_are_rejected(self) -> None:
        coordinates = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )

        with self.assertRaises(ValueError):
            reference_interface_pca_frame(coordinates)

    def test_too_few_atoms_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reference_interface_pca_frame(
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            )


if __name__ == "__main__":
    unittest.main()
