import unittest

import numpy as np

from rfd3_mosaic.geometry import (
    apply_transform,
    build_cyclic_registry,
    build_dihedral_registry,
    build_transform_registry,
    cyclic_transform_id,
    dihedral_transform_id,
    validate_group_closure,
)
from rfd3_mosaic.schema.specs import (
    SymmetryTransformSetSpec,
    SymmetryType,
)


class CyclicSymmetryRegistryTestCase(unittest.TestCase):
    def test_c3_has_stable_transform_ids(self) -> None:
        registry = build_cyclic_registry(3)

        self.assertEqual(
            registry.transform_ids,
            ("C3:e", "C3:r1", "C3:r2"),
        )
        np.testing.assert_allclose(
            registry.transform("C3:e"),
            np.eye(4),
            atol=1e-7,
        )

    def test_transform_ids_normalize_copy_indices(self) -> None:
        self.assertEqual(cyclic_transform_id(3, -1), "C3:r2")
        self.assertEqual(cyclic_transform_id(3, 3), "C3:e")
        self.assertEqual(cyclic_transform_id(3, 4), "C3:r1")

    def test_signed_orbit_offsets_are_deterministic(self) -> None:
        registry = build_cyclic_registry(3)

        self.assertEqual(registry.transform_id_for_offset(-1), "C3:r2")
        self.assertEqual(registry.transform_id_for_offset(1), "C3:r1")
        self.assertEqual(
            registry.transform_id_for_offset(-1, source_copy_index=1),
            "C3:e",
        )

    def test_c3_c4_c5_are_closed(self) -> None:
        for order in (3, 4, 5):
            with self.subTest(order=order):
                validate_group_closure(build_cyclic_registry(order))

    def test_composition_returns_expected_group_element(self) -> None:
        registry = build_cyclic_registry(5)

        self.assertEqual(
            registry.compose_ids("C5:r2", "C5:r4"),
            "C5:r1",
        )

    def test_rotation_uses_requested_center(self) -> None:
        center = np.array([4.0, -2.0, 1.0])
        registry = build_cyclic_registry(
            4,
            axis=(0.0, 0.0, 1.0),
            center=center,
        )

        transformed_center = apply_transform(
            center,
            registry.transform("C4:r1"),
        )
        np.testing.assert_allclose(transformed_center, center, atol=1e-7)

    def test_master_copy_reconstruction_has_no_drift(self) -> None:
        registry = build_cyclic_registry(3)
        master = np.array(
            [
                [2.0, 0.0, 1.0],
                [3.0, 1.0, -1.0],
            ]
        )

        copy_one = apply_transform(master, registry.transform("C3:r1"))
        reconstructed = apply_transform(
            copy_one,
            registry.transform("C3:r2"),
        )
        np.testing.assert_allclose(reconstructed, master, atol=1e-7)

    def test_schema_spec_dispatches_to_cyclic_registry(self) -> None:
        spec = SymmetryTransformSetSpec(
            type=SymmetryType.CYCLIC,
            order=3,
            axis=(0.0, 1.0, 0.0),
            center=(1.0, 2.0, 3.0),
        )

        registry = build_transform_registry(spec)

        self.assertEqual(registry.group_name, "C3")
        self.assertEqual(registry.order, 3)

    def test_schema_spec_dispatches_to_dihedral_registry(self) -> None:
        spec = SymmetryTransformSetSpec(
            type=SymmetryType.DIHEDRAL,
            order=3,
            secondary_axis=(1.0, 0.0, 0.0),
        )

        registry = build_transform_registry(spec)

        self.assertEqual(registry.group_name, "D3")
        self.assertEqual(registry.order, 6)


class DihedralSymmetryRegistryTestCase(unittest.TestCase):
    def test_d3_has_stable_transform_ids(self) -> None:
        registry = build_dihedral_registry(3)

        self.assertEqual(
            registry.transform_ids,
            (
                "D3:e",
                "D3:r1",
                "D3:r2",
                "D3:s0",
                "D3:s1",
                "D3:s2",
            ),
        )

    def test_transform_ids_normalize_group_indices(self) -> None:
        self.assertEqual(dihedral_transform_id(3, -1), "D3:s2")
        self.assertEqual(dihedral_transform_id(3, 6), "D3:e")
        self.assertEqual(dihedral_transform_id(3, 4), "D3:s1")

    def test_d2_d3_d5_are_closed(self) -> None:
        for order in (2, 3, 5):
            with self.subTest(order=order):
                validate_group_closure(build_dihedral_registry(order))

    def test_all_rotations_fix_requested_center(self) -> None:
        center = np.array([4.0, -2.0, 1.0])
        registry = build_dihedral_registry(4, center=center)

        for transform_id in registry.transform_ids:
            with self.subTest(transform_id=transform_id):
                np.testing.assert_allclose(
                    apply_transform(center, registry.transform(transform_id)),
                    center,
                    atol=1e-7,
                )

    def test_twofold_rotation_uses_requested_secondary_axis(self) -> None:
        registry = build_dihedral_registry(
            3,
            axis=(0.0, 0.0, 1.0),
            secondary_axis=(1.0, 0.0, 0.0),
        )

        transformed = apply_transform(
            np.array([0.0, 1.0, 1.0]),
            registry.transform("D3:s0"),
        )
        np.testing.assert_allclose(
            transformed,
            np.array([0.0, -1.0, -1.0]),
            atol=1e-7,
        )

    def test_orbit_offsets_stay_inside_each_cyclic_coset(self) -> None:
        registry = build_dihedral_registry(3)

        self.assertEqual(
            registry.transform_id_for_offset(1, source_copy_index=2),
            "D3:e",
        )
        self.assertEqual(
            registry.transform_id_for_offset(1, source_copy_index=5),
            "D3:s0",
        )

    def test_twofold_relation_pairs_dihedral_cosets(self) -> None:
        registry = build_dihedral_registry(3)

        expected_targets = (
            "D3:s0",
            "D3:s1",
            "D3:s2",
            "D3:e",
            "D3:r1",
            "D3:r2",
        )
        observed_targets = tuple(
            registry.transform_id_for_relation(
                "D3:s0",
                source_copy_index=source_copy_index,
            )
            for source_copy_index in range(6)
        )
        self.assertEqual(observed_targets, expected_targets)

    def test_unknown_relation_transform_is_rejected(self) -> None:
        registry = build_dihedral_registry(3)

        with self.assertRaisesRegex(KeyError, "Unknown transform"):
            registry.transform_id_for_relation("D4:s0")

    def test_nonperpendicular_secondary_axis_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "perpendicular"):
            build_dihedral_registry(
                3,
                axis=(0.0, 0.0, 1.0),
                secondary_axis=(1.0, 0.0, 1.0),
            )


if __name__ == "__main__":
    unittest.main()
