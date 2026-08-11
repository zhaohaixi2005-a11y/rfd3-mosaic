import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rfd3_mosaic.seed_stabilizer import resolve_seed_stabilizer


def _atom_line(
    serial: int,
    atom_name: str,
    chain: str,
    residue: int,
    coordinate: np.ndarray,
) -> str:
    x, y, z = coordinate
    return (
        f"ATOM  {serial:5d} {atom_name:^4s} ALA {chain}{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}"
        f"          {atom_name[0]:>2s}\n"
    )


def _write_regular_seed(path: Path, order: int) -> None:
    base = []
    for residue in range(1, 4):
        for atom_index, atom_name in enumerate(("N", "CA", "C", "O")):
            base.append(
                (
                    residue,
                    atom_name,
                    np.asarray(
                        (
                            2.5 + residue,
                            0.35 * atom_index,
                            1.2 * residue + 0.15 * atom_index,
                        ),
                        dtype=np.float64,
                    ),
                )
            )
    lines = []
    serial = 1
    for copy_index in range(order):
        angle = 2.0 * math.pi * copy_index / order
        rotation = np.asarray(
            (
                (math.cos(angle), -math.sin(angle), 0.0),
                (math.sin(angle), math.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        chain = chr(ord("A") + copy_index)
        for residue, atom_name, coordinate in base:
            lines.append(
                _atom_line(
                    serial,
                    atom_name,
                    chain,
                    residue,
                    rotation @ coordinate,
                )
            )
            serial += 1
    path.write_text("".join(lines) + "END\n", encoding="utf-8")


class SeedStabilizerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_binary_c2_seed_resolves_tetrahedral_six_orbit(self) -> None:
        structure = self.root / "c2_seed.pdb"
        _write_regular_seed(structure, 2)

        evidence = resolve_seed_stabilizer(
            source=structure,
            interface_id="binary_seed",
            participants=("A", "B"),
            selectors={"A": "A/1-3/*", "B": "B/1-3/*"},
            symmetry="T",
            orbit_size=6,
        )

        self.assertEqual(len(evidence.stabilizer_transform_ids), 2)
        self.assertEqual(len(evidence.coset_representative_ids), 6)
        self.assertLess(evidence.maximum_fit_rmsd, 1e-5)

    def test_three_way_c3_seed_resolves_tetrahedral_four_orbit(self) -> None:
        structure = self.root / "c3_seed.pdb"
        _write_regular_seed(structure, 3)

        evidence = resolve_seed_stabilizer(
            source=structure,
            interface_id="three_way_seed",
            participants=("A", "B", "C"),
            selectors={
                "A": "A/1-3/*",
                "B": "B/1-3/*",
                "C": "C/1-3/*",
            },
            symmetry="T",
            orbit_size=4,
        )

        self.assertEqual(len(evidence.stabilizer_transform_ids), 3)
        self.assertEqual(len(evidence.coset_representative_ids), 4)
        # The synthetic PDB is serialized to three decimal places, so the
        # recovered common center carries a small coordinate-quantization
        # residual even though the underlying C3 construction is exact.
        self.assertLess(evidence.common_center_residual, 1e-4)

    def test_parallel_translation_is_not_accepted_as_c2(self) -> None:
        structure = self.root / "translated_seed.pdb"
        _write_regular_seed(structure, 1)
        text = structure.read_text(encoding="utf-8")
        second = []
        for line in text.splitlines():
            if not line.startswith("ATOM"):
                continue
            coordinate = np.asarray(
                (
                    float(line[30:38]) + 4.0,
                    float(line[38:46]),
                    float(line[46:54]),
                )
            )
            second.append(
                _atom_line(
                    int(line[6:11]) + 100,
                    line[12:16].strip(),
                    "B",
                    int(line[22:26]),
                    coordinate,
                )
            )
        structure.write_text(
            text.replace("END\n", "") + "".join(second) + "END\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "closed rotation group"):
            resolve_seed_stabilizer(
                source=structure,
                interface_id="translated",
                participants=("A", "B"),
                selectors={"A": "A/1-3/*", "B": "B/1-3/*"},
                symmetry="T",
                orbit_size=6,
            )


if __name__ == "__main__":
    unittest.main()
