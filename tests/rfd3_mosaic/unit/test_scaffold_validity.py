import unittest

from rfd3_mosaic.structure import AtomRecord
from rfd3_mosaic.validation import audit_scaffold_geometry


def _atom(
    serial: int,
    name: str,
    residue: int,
    x: float,
    y: float = 0.0,
    z: float = 0.0,
) -> AtomRecord:
    return AtomRecord(
        record_type="ATOM",
        serial=serial,
        atom_name=name,
        alternate_location="",
        residue_name="GLY",
        chain_id="A",
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


if __name__ == "__main__":
    unittest.main()
