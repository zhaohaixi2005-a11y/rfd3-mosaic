import unittest
from pathlib import Path

import numpy as np

from rfd3_mosaic.compile import (
    load_interface_seed_config,
    resolve_reference_port_frames,
)
from rfd3_mosaic.geometry import (
    apply_transform,
    build_cyclic_registry,
    validate_transform,
)
from rfd3_mosaic.structure import load_selected_atoms


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LHD101_CONFIG = (
    REPOSITORY_ROOT
    / "configs/rfd3_mosaic/single_interface/lhd101_c3.yaml"
)


class LHD101ReferenceIntegrationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_interface_seed_config(LHD101_CONFIG)
        cls.left_atoms = load_selected_atoms(
            cls.spec.fragments["left"],
            base_directory=REPOSITORY_ROOT,
        )
        cls.right_atoms = load_selected_atoms(
            cls.spec.fragments["right"],
            base_directory=REPOSITORY_ROOT,
        )

    def test_real_lhd101_fragment_selections(self) -> None:
        self.assertEqual(len(self.left_atoms), 245)
        self.assertEqual(len(self.right_atoms), 251)
        self.assertEqual(
            {atom.residue_number for atom in self.left_atoms},
            set(range(165, 195)),
        )
        self.assertEqual(
            {atom.residue_number for atom in self.right_atoms},
            set(range(211, 242)),
        )

    def test_real_port_frames_are_valid_and_deterministic(self) -> None:
        first = resolve_reference_port_frames(
            self.spec,
            base_directory=REPOSITORY_ROOT,
        )
        second = resolve_reference_port_frames(
            self.spec,
            base_directory=REPOSITORY_ROOT,
        )

        self.assertEqual(first, second)
        validate_transform(first["left_port"])
        validate_transform(first["right_port"])

    def test_port_normals_point_toward_partner_centroids(self) -> None:
        frames = resolve_reference_port_frames(
            self.spec,
            base_directory=REPOSITORY_ROOT,
        )
        left_frame = np.asarray(frames["left_port"])
        right_frame = np.asarray(frames["right_port"])
        left_to_right = right_frame[:3, 3] - left_frame[:3, 3]
        right_to_left = -left_to_right

        self.assertGreater(np.dot(left_frame[:3, 2], left_to_right), 0.0)
        self.assertGreater(np.dot(right_frame[:3, 2], right_to_left), 0.0)

    def test_c3_copy_preserves_real_fragment_distances(self) -> None:
        coordinates = np.asarray(
            [atom.coordinate for atom in self.left_atoms],
            dtype=np.float64,
        )
        registry = build_cyclic_registry(3)
        transformed = apply_transform(
            coordinates,
            registry.transform("C3:r1"),
        )

        sample = (0, 50, 100, 150, 200, 244)
        original_sample = coordinates[list(sample)]
        transformed_sample = transformed[list(sample)]
        original_distances = np.linalg.norm(
            original_sample[:, None] - original_sample[None, :],
            axis=-1,
        )
        transformed_distances = np.linalg.norm(
            transformed_sample[:, None] - transformed_sample[None, :],
            axis=-1,
        )
        np.testing.assert_allclose(
            transformed_distances,
            original_distances,
            atol=1e-7,
        )

    def test_c3_copy_preserves_complete_reference_interface(self) -> None:
        """The two motif fragments must move as one rigid interface seed."""

        left = np.asarray(
            [atom.coordinate for atom in self.left_atoms],
            dtype=np.float64,
        )
        right = np.asarray(
            [atom.coordinate for atom in self.right_atoms],
            dtype=np.float64,
        )
        original_cross_distances = np.linalg.norm(
            left[:, None, :] - right[None, :, :],
            axis=-1,
        )

        transform = build_cyclic_registry(3).transform("C3:r1")
        transformed_left = apply_transform(left, transform)
        transformed_right = apply_transform(right, transform)
        transformed_cross_distances = np.linalg.norm(
            transformed_left[:, None, :] - transformed_right[None, :, :],
            axis=-1,
        )

        np.testing.assert_allclose(
            transformed_cross_distances,
            original_cross_distances,
            atol=1e-7,
        )
        self.assertGreater(int((original_cross_distances < 4.5).sum()), 0)


if __name__ == "__main__":
    unittest.main()
