import unittest

from rfd3_mosaic.rfd3_prevalidate import (
    _expected_multiplicity,
    _validate_report,
)


class RFD3PrevalidationLogicTestCase(unittest.TestCase):
    def test_reads_cyclic_multiplicity(self) -> None:
        self.assertEqual(_expected_multiplicity("C3"), 3)
        self.assertEqual(_expected_multiplicity("c12"), 12)

    def test_reads_dihedral_multiplicity(self) -> None:
        self.assertEqual(_expected_multiplicity("D2"), 4)
        self.assertEqual(_expected_multiplicity("d5"), 10)

    def test_rejects_unsupported_symmetry(self) -> None:
        with self.assertRaisesRegex(ValueError, "Cn/Dn"):
            _expected_multiplicity("T3")

    def test_accepts_consistent_constructed_atom_array_report(self) -> None:
        report = {
            "expected_multiplicity": 3,
            "expected_asu_chain_count": 1,
            "chain_count": 3,
            "symmetry_transform_ids": [0, 1, 2],
            "residues_per_chain": {"A": 120, "B": 120, "C": 120},
            "motif_atom_count": 100,
            "fixed_coordinate_atom_count": 100,
            "fixed_sequence_atom_count": 100,
            "asu_atom_count": 500,
            "symmetry_ids": ["C3"],
            "expected_symmetry_id": "C3",
        }
        self.assertEqual(_validate_report(report), [])

    def test_reports_incomplete_symmetry_expansion(self) -> None:
        report = {
            "expected_multiplicity": 3,
            "expected_asu_chain_count": 1,
            "chain_count": 2,
            "symmetry_transform_ids": [0, 1],
            "residues_per_chain": {"A": 120, "B": 119},
            "motif_atom_count": 0,
            "fixed_coordinate_atom_count": 0,
            "fixed_sequence_atom_count": 0,
            "asu_atom_count": 0,
            "symmetry_ids": ["C2"],
            "expected_symmetry_id": "C3",
        }
        self.assertGreaterEqual(len(_validate_report(report)), 6)

    def test_accepts_two_chain_asu_repeated_by_symmetry(self) -> None:
        report = {
            "expected_multiplicity": 3,
            "expected_asu_chain_count": 2,
            "chain_count": 6,
            "symmetry_transform_ids": [0, 1, 2],
            "residues_per_chain": {
                "A": 31,
                "B": 30,
                "C": 31,
                "D": 30,
                "E": 31,
                "F": 30,
            },
            "motif_atom_count": 100,
            "fixed_coordinate_atom_count": 100,
            "fixed_sequence_atom_count": 100,
            "asu_atom_count": 500,
            "symmetry_ids": ["C3"],
            "expected_symmetry_id": "C3",
        }
        self.assertEqual(_validate_report(report), [])

    def test_rejects_partially_fixed_interface_seed(self) -> None:
        report = {
            "expected_multiplicity": 3,
            "expected_asu_chain_count": 1,
            "chain_count": 3,
            "symmetry_transform_ids": [0, 1, 2],
            "residues_per_chain": {"A": 120, "B": 120, "C": 120},
            "motif_atom_count": 100,
            "fixed_coordinate_atom_count": 60,
            "fixed_sequence_atom_count": 100,
            "asu_atom_count": 500,
            "symmetry_ids": ["C3"],
            "expected_symmetry_id": "C3",
        }
        failures = _validate_report(report)
        self.assertTrue(
            any("not every motif atom has fixed coordinates" in failure
                for failure in failures)
        )


if __name__ == "__main__":
    unittest.main()
