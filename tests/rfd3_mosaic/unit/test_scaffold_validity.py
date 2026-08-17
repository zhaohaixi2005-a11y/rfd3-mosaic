from dataclasses import replace
import math
import unittest

import numpy as np

from rfd3_mosaic.structure import AtomRecord
from rfd3_mosaic.validation import audit_scaffold_geometry


def _atom(
    serial: int,
    name: str,
    residue: int,
    x: float,
    y: float = 0.0,
    z: float = 0.0,
    chain: str = "A",
) -> AtomRecord:
    return AtomRecord(
        record_type="ATOM",
        serial=serial,
        atom_name=name,
        alternate_location="",
        residue_name="GLY",
        chain_id=chain,
        residue_number=residue,
        insertion_code="",
        coordinate=(x, y, z),
        element=name[0],
    )


class ScaffoldValidityTestCase(unittest.TestCase):
    def test_compact_continuous_chain_passes(self) -> None:
        atoms = (
            _atom(1, "N", 1, 0.00),
            _atom(2, "CA", 1, 1.45),
            _atom(3, "C", 1, 2.45),
            _atom(4, "N", 2, 3.78),
            _atom(5, "CA", 2, 5.10),
            _atom(6, "C", 2, 6.20),
            _atom(7, "N", 3, 7.53),
            _atom(8, "CA", 3, 8.80),
            _atom(9, "C", 3, 9.90),
        )

        report = audit_scaffold_geometry(atoms)

        self.assertTrue(report["passed"])
        self.assertEqual(report["summary"]["chain_break_count"], 0)

    def test_chain_break_and_extended_chain_fail(self) -> None:
        atoms = (
            _atom(1, "N", 1, 0.0),
            _atom(2, "CA", 1, 1.0),
            _atom(3, "C", 1, 2.0),
            _atom(4, "N", 2, 10.0),
            _atom(5, "CA", 2, 11.0),
            _atom(6, "C", 2, 12.0),
        )

        report = audit_scaffold_geometry(
            atoms,
            max_chain_ca_rg=2.0,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["summary"]["chain_break_count"], 1)
        self.assertFalse(report["summary"]["passed_compactness"])

    def test_declared_c3_transforms_are_the_symmetry_hard_gate(
        self,
    ) -> None:
        transforms = []
        for copy_index in range(3):
            angle = 2.0 * math.pi * copy_index / 3.0
            matrix = np.eye(4)
            matrix[:3, :3] = np.asarray(
                [
                    [math.cos(angle), -math.sin(angle), 0.0],
                    [math.sin(angle), math.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            transforms.append(matrix)

        atoms = []
        serial = 1
        for chain_index, chain_id in enumerate(("A", "B", "C")):
            for residue, base in enumerate(
                (10.0, 13.78, 17.56),
                start=1,
            ):
                source_coordinates = (
                    (base, 0.0, 0.0),
                    (base + 1.45, 0.0, 0.0),
                    (base + 2.45, 0.0, 0.0),
                )
                transformed = [
                    np.asarray(coordinate)
                    @ transforms[chain_index][:3, :3].T
                    for coordinate in source_coordinates
                ]
                atoms.extend(
                    (
                        _atom(
                            serial,
                            "N",
                            residue,
                            *transformed[0],
                            chain=chain_id,
                        ),
                        _atom(
                            serial + 1,
                            "CA",
                            residue,
                            *transformed[1],
                            chain=chain_id,
                        ),
                        _atom(
                            serial + 2,
                            "C",
                            residue,
                            *transformed[2],
                            chain=chain_id,
                        ),
                    )
                )
                serial += 3

        report = audit_scaffold_geometry(
            tuple(atoms),
            expected_symmetry_multiplicity=3,
            expected_symmetry_transforms=tuple(transforms),
        )

        self.assertTrue(report["summary"]["passed_symmetry"])
        self.assertLess(
            report["summary"]["maximum_symmetry_coordinate_rmsd"],
            1e-10,
        )

        # Pure translations retain each chain's internal distance matrix but
        # do not satisfy the declared C3 transforms.
        translated = []
        for atom in atoms:
            chain_offset = 20.0 * ("ABC".index(atom.chain_id))
            translated.append(
                replace(
                    atom,
                    coordinate=(
                        atom.coordinate[0] + chain_offset,
                        atom.coordinate[1],
                        atom.coordinate[2],
                    ),
                )
            )
        translated_report = audit_scaffold_geometry(
            tuple(translated),
            expected_symmetry_multiplicity=3,
            expected_symmetry_transforms=tuple(transforms),
        )
        self.assertFalse(
            translated_report["summary"]["passed_symmetry"]
        )
        self.assertLess(
            translated_report["summary"][
                "maximum_copy_internal_distance_matrix_rmsd"
            ],
            1e-10,
        )

        broken = list(atoms)
        atom = broken[-2]
        broken[-2] = replace(
            atom,
            coordinate=(
                atom.coordinate[0] + 0.2,
                atom.coordinate[1],
                atom.coordinate[2],
            ),
        )
        broken_report = audit_scaffold_geometry(
            tuple(broken),
            expected_symmetry_multiplicity=3,
            expected_symmetry_transforms=tuple(transforms),
        )
        self.assertFalse(broken_report["summary"]["passed_symmetry"])

    def test_symmetry_audit_fails_closed_without_transforms(self) -> None:
        atoms = (
            _atom(1, "CA", 1, 0.0, chain="A"),
            _atom(2, "CA", 1, 0.0, chain="B"),
            _atom(3, "CA", 1, 0.0, chain="C"),
        )

        report = audit_scaffold_geometry(
            atoms,
            expected_symmetry_multiplicity=3,
        )

        self.assertFalse(report["summary"]["passed_symmetry"])
        self.assertIn(
            "expected symmetry transforms were not provided",
            report["symmetry"]["failures"],
        )

    def test_symmetry_audit_rejects_non_rigid_transform(self) -> None:
        atoms = (
            _atom(1, "CA", 1, 1.0, chain="A"),
            _atom(2, "CA", 1, 1.0, chain="B"),
            _atom(3, "CA", 1, 1.0, chain="C"),
        )
        transforms = [np.eye(4) for _ in range(3)]
        transforms[1][0, 0] = 0.0

        report = audit_scaffold_geometry(
            atoms,
            expected_symmetry_multiplicity=3,
            expected_symmetry_transforms=tuple(transforms),
        )

        self.assertFalse(report["summary"]["passed_symmetry"])
        self.assertTrue(
            any(
                "not a proper rigid rotation" in failure
                for failure in report["symmetry"]["failures"]
            )
        )

    def test_symmetry_audit_accepts_preexpanded_mixed_entity_orbits(
        self,
    ) -> None:
        transforms = []
        for angle in (0.0, 90.0, 180.0, 270.0):
            radians = np.radians(angle)
            transform = np.eye(4)
            transform[:2, :2] = (
                (np.cos(radians), -np.sin(radians)),
                (np.sin(radians), np.cos(radians)),
            )
            transforms.append(transform)

        atoms = []
        layout = []
        serial = 1
        declarations = (
            (0, (0, 2), np.asarray((5.0, 0.0, 0.0))),
            (1, (0, 1, 2, 3), np.asarray((10.0, 0.0, 0.0))),
        )
        chain_index = 0
        for entity_id, transform_indices, source in declarations:
            for transform_index in transform_indices:
                chain_id = chr(ord("A") + chain_index)
                coordinate = (
                    source @ transforms[transform_index][:3, :3].T
                    + transforms[transform_index][:3, 3]
                )
                atoms.append(
                    _atom(
                        serial,
                        "CA",
                        1,
                        *coordinate,
                        chain=chain_id,
                    )
                )
                layout.append(
                    {
                        "entity_id": entity_id,
                        "transform_index": transform_index,
                        "is_asu": transform_index == 0,
                    }
                )
                serial += 1
                chain_index += 1

        report = audit_scaffold_geometry(
            tuple(atoms),
            expected_symmetry_multiplicity=4,
            expected_symmetry_transforms=tuple(transforms),
            expected_symmetry_chain_layout=tuple(layout),
        )

        self.assertTrue(report["summary"]["passed_symmetry"])
        self.assertEqual(
            len(report["symmetry"]["transform_comparisons"]),
            4,
        )


if __name__ == "__main__":
    unittest.main()
