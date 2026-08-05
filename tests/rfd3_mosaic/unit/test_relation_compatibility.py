import unittest

import numpy as np

from rfd3_mosaic.geometry import build_cyclic_registry
from rfd3_mosaic.geometry.se3 import invert_transform
from rfd3_mosaic.relation_compatibility import (
    CyclicRelationSearchSpace,
    enumerate_cyclic_relation_compatibility,
)


class RelationCompatibilityTestCase(unittest.TestCase):
    def test_c3_relation_distinguishes_subgroup_from_full_group(self) -> None:
        transform = build_cyclic_registry(
            3,
            center=(4.0, -2.0, 10.0),
        ).transform("C3:r1")

        report = enumerate_cyclic_relation_compatibility(transform)
        by_id = {item.id: item for item in report.compatible_relations}

        self.assertIn("C3:k1|k2", by_id)
        self.assertIn("C6:k2|k4", by_id)
        self.assertTrue(by_id["C3:k1|k2"].relation_generates_complete_group)
        self.assertEqual(by_id["C3:k1|k2"].unobserved_cosets, 0)
        self.assertFalse(by_id["C6:k2|k4"].relation_generates_complete_group)
        self.assertEqual(by_id["C6:k2|k4"].unobserved_cosets, 1)
        self.assertEqual(
            by_id["C6:k2|k4"].observed_relation_subgroup_order,
            3,
        )

    def test_axis_line_uses_a_canonical_axial_gauge(self) -> None:
        transform = build_cyclic_registry(
            5,
            axis=(0.0, 0.0, 1.0),
            center=(4.0, -2.0, 10.0),
        ).transform("C5:r1")

        report = enumerate_cyclic_relation_compatibility(transform)
        hypothesis = next(
            item
            for item in report.compatible_relations
            if item.proposed_group == "C5"
        )

        np.testing.assert_allclose(
            hypothesis.axis,
            [0.0, 0.0, 1.0],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            hypothesis.axis_point,
            [4.0, -2.0, 0.0],
            atol=1e-7,
        )

    def test_inverse_relation_has_the_same_compatibility_classes(self) -> None:
        transform = build_cyclic_registry(3).transform("C3:r1")

        forward = enumerate_cyclic_relation_compatibility(transform)
        inverse = enumerate_cyclic_relation_compatibility(
            invert_transform(transform)
        )

        self.assertEqual(
            [item.id for item in forward.compatible_relations],
            [item.id for item in inverse.compatible_relations],
        )

    def test_small_rotation_noise_is_retained_with_nonzero_residual(self) -> None:
        angle = np.radians(120.25)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        transform = np.eye(4)
        transform[:3, :3] = np.array(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        report = enumerate_cyclic_relation_compatibility(
            transform,
            CyclicRelationSearchSpace(
                orders=(3,),
                max_angle_error_deg=1.0,
                max_closure_rotation_deg=1.0,
            ),
        )

        self.assertEqual(len(report.compatible_relations), 1)
        hypothesis = report.compatible_relations[0]
        self.assertAlmostEqual(hypothesis.angle_error_deg, 0.25, places=6)
        self.assertGreater(hypothesis.relation_compatibility_score, 0.0)

    def test_screw_translation_is_rejected_for_point_group(self) -> None:
        transform = build_cyclic_registry(3).transform("C3:r1").copy()
        transform[:3, 3] += np.array([0.0, 0.0, 0.5])

        report = enumerate_cyclic_relation_compatibility(
            transform,
            CyclicRelationSearchSpace(max_screw_translation=0.1),
        )

        self.assertEqual(report.compatible_relations, ())

    def test_identity_relation_is_rejected_as_uninformative(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not determine"):
            enumerate_cyclic_relation_compatibility(np.eye(4))

    def test_invalid_rotation_is_rejected(self) -> None:
        transform = np.eye(4)
        transform[0, 0] = 2.0

        with self.assertRaisesRegex(ValueError, "not orthogonal"):
            enumerate_cyclic_relation_compatibility(transform)


if __name__ == "__main__":
    unittest.main()
