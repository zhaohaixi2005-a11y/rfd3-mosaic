import unittest

from rfd3_mosaic.structure import AtomRecord
from rfd3_mosaic.validation import (
    FragmentPlacement,
    audit_two_fragment_seed,
    infer_fragment_placements,
)


def atom(
    chain: str,
    residue: int,
    coordinate: tuple[float, float, float],
) -> AtomRecord:
    return AtomRecord(
        record_type="ATOM",
        serial=1,
        atom_name="CA",
        alternate_location="",
        residue_name="ALA",
        chain_id=chain,
        residue_number=residue,
        insertion_code="",
        coordinate=coordinate,
        element="C",
    )


class SeedIntegrityTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.references = {
            "left": (
                atom("L", 10, (0.0, 0.0, 0.0)),
                atom("L", 11, (0.0, 1.0, 0.0)),
                atom("L", 12, (0.0, 0.0, 1.0)),
            ),
            "right": (
                atom("R", 20, (3.0, 0.0, 0.0)),
                atom("R", 21, (3.0, 1.0, 0.0)),
                atom("R", 22, (3.0, 0.0, 1.0)),
            ),
        }
        self.placements = {
            "left": FragmentPlacement(
                "left", "X", {10: 10, 11: 11, 12: 12}
            ),
            "right": FragmentPlacement(
                "right", "X", {20: 1, 21: 2, 22: 3}
            ),
        }

    def _cyclic_output(self, perturb: float = 0.0):
        output = []
        chains = ("A", "B", "C")
        translations = (0.0, 100.0, 200.0)
        for index, chain in enumerate(chains):
            current = translations[index]
            previous = translations[(index - 1) % 3]
            for offset, residue in enumerate((10, 11, 12)):
                coordinate = self.references["left"][offset].coordinate
                output.append(
                    atom(
                        chain,
                        residue,
                        (
                            coordinate[0] + current,
                            coordinate[1],
                            coordinate[2],
                        ),
                    )
                )
            for offset, residue in enumerate((1, 2, 3)):
                coordinate = self.references["right"][offset].coordinate
                extra = perturb if chain == "B" else 0.0
                output.append(
                    atom(
                        chain,
                        residue,
                        (
                            coordinate[0] + previous + extra,
                            coordinate[1],
                            coordinate[2],
                        ),
                    )
                )
        return tuple(output)

    def test_infers_fragment_ranges_from_adapter_and_result_mappings(self):
        mapping = {
            "atom_mappings": [
                {
                    "source": {
                        "fragment_id": "right",
                        "residue_number": 20,
                    },
                    "compiled": {"chain_id": "B", "label_seq_id": 1},
                },
                {
                    "source": {
                        "fragment_id": "left",
                        "residue_number": 10,
                    },
                    "compiled": {"chain_id": "C", "label_seq_id": 1},
                },
            ]
        }
        placements = infer_fragment_placements(
            mapping, {"B1": "A1", "C1": "A10"}
        )

        self.assertEqual(
            placements["right"].output_residues_by_source, {20: 1}
        )
        self.assertEqual(
            placements["left"].output_residues_by_source, {10: 10}
        )

    def test_finds_cross_chain_cyclic_seed_pairs(self):
        report = audit_two_fragment_seed(
            output_atoms=self._cyclic_output(),
            references=self.references,
            placements=self.placements,
            left_fragment_id="left",
            right_fragment_id="right",
        )

        self.assertTrue(report["passed"])
        self.assertEqual(
            {
                (pair["left_chain"], pair["right_chain"])
                for pair in report["seed_pairs"]
            },
            {("A", "B"), ("B", "C"), ("C", "A")},
        )
        self.assertLess(report["summary"]["maximum_ca_rmsd"], 1e-6)

    def test_rejects_a_displaced_interface_fragment(self):
        report = audit_two_fragment_seed(
            output_atoms=self._cyclic_output(perturb=3.0),
            references=self.references,
            placements=self.placements,
            left_fragment_id="left",
            right_fragment_id="right",
        )

        self.assertFalse(report["passed"])
        failed = next(
            pair
            for pair in report["seed_pairs"]
            if pair["right_chain"] == "B"
        )
        self.assertIn("ca_rmsd", failed["failed_checks"])


if __name__ == "__main__":
    unittest.main()
