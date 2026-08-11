import unittest

import numpy as np

from rfd3_mosaic.geometry import (
    apply_transform,
    build_cyclic_registry,
    build_dihedral_registry,
    build_polyhedral_registry,
)
from rfd3_mosaic.schema.specs import SymmetryType
from rfd3_mosaic.structure import AtomRecord
from rfd3_mosaic.validation import (
    audit_assembly_morphology,
    audit_scaffold_geometry,
)


def _ca(
    serial: int,
    coordinate: np.ndarray,
    *,
    chain: str,
) -> AtomRecord:
    return AtomRecord(
        record_type="ATOM",
        serial=serial,
        atom_name="CA",
        alternate_location="",
        residue_name="GLY",
        chain_id=chain,
        residue_number=1,
        insertion_code="",
        coordinate=tuple(float(value) for value in coordinate),
        element="C",
    )


def _orbit_atoms(registry, points: tuple[np.ndarray, ...]) -> tuple[AtomRecord, ...]:
    atoms = []
    serial = 1
    for copy_index, transform_id in enumerate(registry.transform_ids):
        chain = chr(ord("A") + copy_index)
        for point in points:
            atoms.append(
                _ca(
                    serial,
                    apply_transform(point, registry.transform(transform_id)),
                    chain=chain,
                )
            )
            serial += 1
    return tuple(atoms)


def _transforms(registry) -> tuple[np.ndarray, ...]:
    return tuple(
        registry.transform(transform_id)
        for transform_id in registry.transform_ids
    )


class AssemblyMorphologyTestCase(unittest.TestCase):
    def test_c3_reports_unambiguous_pore_and_outer_diameter(self) -> None:
        registry = build_cyclic_registry(
            3,
            center=(4.0, -2.0, 7.0),
        )
        atoms = _orbit_atoms(
            registry,
            (
                np.asarray((12.0, -2.0, 5.0)),
                np.asarray((14.0, -2.0, 9.0)),
            ),
        )

        report = audit_assembly_morphology(
            atoms,
            symmetry_transforms=_transforms(registry),
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["measurement_only"])
        summary = report["summary"]
        self.assertFalse(summary["center_is_unique"])
        self.assertEqual(summary["fixed_point_rank"], 2)
        self.assertTrue(summary["principal_axis_unique"])
        self.assertEqual(summary["principal_axis_fold"], 3)
        self.assertAlmostEqual(summary["central_pore_diameter"], 16.0)
        self.assertAlmostEqual(summary["outer_radial_diameter"], 20.0)
        np.testing.assert_allclose(
            summary["principal_axis_direction"],
            (0.0, 0.0, 1.0),
            atol=1.0e-7,
        )

    def test_explicit_pore_bound_can_gate_without_an_invented_default(
        self,
    ) -> None:
        registry = build_cyclic_registry(3)
        atoms = _orbit_atoms(
            registry,
            (np.asarray((8.0, 0.0, 0.0)),),
        )

        accepted = audit_assembly_morphology(
            atoms,
            symmetry_transforms=_transforms(registry),
            minimum_central_pore_diameter=15.0,
            maximum_central_pore_diameter=17.0,
        )
        rejected = audit_assembly_morphology(
            atoms,
            symmetry_transforms=_transforms(registry),
            maximum_central_pore_diameter=15.0,
        )

        self.assertTrue(accepted["passed"])
        self.assertFalse(accepted["measurement_only"])
        self.assertFalse(rejected["passed"])
        self.assertIn("outside the requested range", rejected["failures"][0])

    def test_d3_selects_the_unique_threefold_axis(self) -> None:
        registry = build_dihedral_registry(3)
        atoms = _orbit_atoms(
            registry,
            (np.asarray((6.0, 0.0, 2.0)),),
        )

        report = audit_assembly_morphology(
            atoms,
            symmetry_transforms=_transforms(registry),
        )

        self.assertTrue(report["passed"])
        summary = report["summary"]
        self.assertTrue(summary["center_is_unique"])
        self.assertEqual(summary["axis_count"], 4)
        self.assertEqual(summary["highest_fold_axis_count"], 1)
        self.assertEqual(summary["principal_axis_fold"], 3)
        self.assertAlmostEqual(summary["central_pore_diameter"], 12.0)

    def test_tetrahedral_group_keeps_equivalent_axes_explicit(self) -> None:
        registry = build_polyhedral_registry(SymmetryType.TETRAHEDRAL)
        atoms = _orbit_atoms(
            registry,
            (np.asarray((8.0, 2.0, 1.0)),),
        )

        report = audit_assembly_morphology(
            atoms,
            symmetry_transforms=_transforms(registry),
        )

        self.assertTrue(report["passed"])
        summary = report["summary"]
        self.assertTrue(summary["center_is_unique"])
        self.assertEqual(summary["highest_axis_fold"], 3)
        self.assertEqual(summary["highest_fold_axis_count"], 4)
        self.assertFalse(summary["principal_axis_unique"])
        self.assertIsNone(summary["central_pore_diameter"])
        self.assertIsNone(summary["outer_radial_diameter"])
        self.assertAlmostEqual(
            summary["spherical_outer_diameter"],
            2.0 * np.sqrt(69.0),
        )

        unavailable = audit_assembly_morphology(
            atoms,
            symmetry_transforms=_transforms(registry),
            minimum_central_pore_diameter=1.0,
        )
        self.assertFalse(unavailable["passed"])
        self.assertIn("unavailable", unavailable["failures"][0])

    def test_incompatible_rotation_axes_fail_closed(self) -> None:
        first = build_cyclic_registry(2)
        second = build_cyclic_registry(2, center=(1.0, 0.0, 0.0))
        transforms = (
            first.transform("C2:e"),
            first.transform("C2:r1"),
            second.transform("C2:r1"),
        )

        report = audit_assembly_morphology(
            (_ca(1, np.asarray((2.0, 0.0, 0.0)), chain="A"),),
            symmetry_transforms=transforms,
        )

        self.assertFalse(report["passed"])
        self.assertIn("do not share", report["failures"][0])

    def test_scaffold_audit_embeds_morphology_without_gating_it(self) -> None:
        registry = build_cyclic_registry(3)
        atoms = _orbit_atoms(
            registry,
            (np.asarray((8.0, 0.0, 0.0)),),
        )

        report = audit_scaffold_geometry(
            atoms,
            expected_symmetry_multiplicity=3,
            expected_symmetry_transforms=_transforms(registry),
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["schema_version"], 2)
        self.assertTrue(
            report["summary"]["assembly_morphology_available"]
        )
        self.assertAlmostEqual(
            report["summary"]["assembly_central_pore_diameter"],
            16.0,
        )
        self.assertTrue(report["morphology"]["diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
