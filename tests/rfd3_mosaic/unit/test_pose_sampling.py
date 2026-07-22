import unittest
from pathlib import Path

import numpy as np

from rfd3_mosaic.compile import (
    _uniform_so3_rotation,
    build_master_group_transforms,
    load_interface_seed_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LHD101_CONFIG = (
    REPOSITORY_ROOT
    / "configs/rfd3_mosaic/single_interface/lhd101_c3.yaml"
)


class PoseSamplingTestCase(unittest.TestCase):
    def test_uniform_so3_sample_is_a_proper_rotation(self) -> None:
        rotation, quaternion = _uniform_so3_rotation(
            np.random.default_rng(17)
        )

        np.testing.assert_allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-12,
        )
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0)

    def test_pose_sampling_is_reproducible_and_seed_dependent(self) -> None:
        spec = load_interface_seed_config(LHD101_CONFIG)
        first = build_master_group_transforms(
            spec,
            base_directory=REPOSITORY_ROOT,
            random_seed=101,
        )
        repeated = build_master_group_transforms(
            spec,
            base_directory=REPOSITORY_ROOT,
            random_seed=101,
        )
        different = build_master_group_transforms(
            spec,
            base_directory=REPOSITORY_ROOT,
            random_seed=102,
        )

        np.testing.assert_allclose(
            first["primary_seed"], repeated["primary_seed"]
        )
        self.assertFalse(
            np.allclose(
                first["primary_seed"], different["primary_seed"]
            )
        )

    def test_sampled_radius_and_pose_are_recorded(self) -> None:
        spec = load_interface_seed_config(LHD101_CONFIG)
        metadata = {}
        build_master_group_transforms(
            spec,
            base_directory=REPOSITORY_ROOT,
            random_seed=103,
            sample_metadata=metadata,
        )
        sample = metadata["primary_seed"]

        self.assertEqual(sample["random_seed"], 103)
        self.assertEqual(sample["orientation_method"], "uniform_so3")
        self.assertGreaterEqual(sample["sampled_radius"], 20.0)
        self.assertLessEqual(sample["sampled_radius"], 30.0)
        self.assertEqual(len(sample["quaternion_xyzw"]), 4)

    def test_sampled_transform_preserves_internal_distances(self) -> None:
        spec = load_interface_seed_config(LHD101_CONFIG)
        transform = build_master_group_transforms(
            spec,
            base_directory=REPOSITORY_ROOT,
            random_seed=104,
        )["primary_seed"]
        points = np.asarray(
            ((0.0, 0.0, 0.0), (3.0, -2.0, 5.0)),
            dtype=np.float64,
        )
        transformed = (
            points @ transform[:3, :3].T + transform[:3, 3]
        )

        self.assertAlmostEqual(
            float(np.linalg.norm(points[1] - points[0])),
            float(np.linalg.norm(transformed[1] - transformed[0])),
        )

    def test_explicit_unit_samples_control_joint_pose(self) -> None:
        spec = load_interface_seed_config(LHD101_CONFIG)
        metadata = {}
        build_master_group_transforms(
            spec,
            base_directory=REPOSITORY_ROOT,
            random_seed=999,
            sample_metadata=metadata,
            sample_overrides={
                "primary_seed": {
                    "radius_unit": 0.75,
                    "axial_offset_unit": 0.5,
                    "so3_unit": [0.25, 0.5, 0.75],
                }
            },
        )
        sample = metadata["primary_seed"]

        self.assertAlmostEqual(sample["sampled_radius"], 27.5)
        self.assertEqual(
            sample["unit_samples"]["so3"],
            [0.25, 0.5, 0.75],
        )
        self.assertIsNotNone(sample["principal_axis_tilt_deg"])
        self.assertGreaterEqual(sample["principal_axis_tilt_deg"], 0.0)
        self.assertLessEqual(sample["principal_axis_tilt_deg"], 90.0)


if __name__ == "__main__":
    unittest.main()
