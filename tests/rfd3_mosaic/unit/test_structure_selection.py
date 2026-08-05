import tempfile
import unittest
from pathlib import Path

from rfd3_mosaic.structure import (
    parse_atom_selection,
    read_mmcif_atoms,
    read_pdb_atoms,
    select_atoms,
)


PDB_TEXT = """\
ATOM      1  N   ALA A 165      10.000  11.000  12.000  1.00 20.00           N
ATOM      2  CA  ALA A 165      11.000  11.000  12.000  1.00 20.00           C
ATOM      3  CB AALA A 165      11.000  12.000  12.000  0.50 20.00           C
ATOM      4  CB BALA A 165      21.000  22.000  22.000  0.50 20.00           C
ATOM      5  N   GLY A 166      12.000  11.000  12.000  1.00 20.00           N
ATOM      6  CA  GLY A 166      13.000  11.000  12.000  1.00 20.00           C
ATOM      7  CA  GLY B 211      30.000  31.000  32.000  1.00 20.00           C
END
"""


class StructureSelectionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "input.pdb"
        self.path.write_text(PDB_TEXT, encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_pdb_reader_resolves_alternate_locations(self) -> None:
        atoms = read_pdb_atoms(self.path)

        self.assertEqual(len(atoms), 6)
        cb = next(atom for atom in atoms if atom.atom_name == "CB")
        self.assertEqual(cb.alternate_location, "A")

    def test_lhd101_selection_syntax_is_parsed(self) -> None:
        selection = parse_atom_selection("A/165-194/*")

        self.assertEqual(selection.chain_id, "A")
        self.assertEqual(selection.residue_start, 165)
        self.assertEqual(selection.residue_end, 194)
        self.assertIsNone(selection.atom_names)

    def test_selection_resolves_chain_residue_and_atom_names(self) -> None:
        atoms = read_pdb_atoms(self.path)

        selected = select_atoms(atoms, "A/165-166/CA")

        self.assertEqual([atom.serial for atom in selected], [2, 6])

    def test_backbone_shortcut_is_supported(self) -> None:
        selected = select_atoms(read_pdb_atoms(self.path), "A/165/backbone")

        self.assertEqual(
            {atom.atom_name for atom in selected},
            {"N", "CA"},
        )

    def test_empty_selection_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_atoms(read_pdb_atoms(self.path), "Z/1-10/*")

    def test_reverse_residue_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_atom_selection("A/194-165/*")

    def test_reads_rfd3_style_mmcif_atom_site_loop(self) -> None:
        cif_path = Path(self.temporary_directory.name) / "result.cif"
        cif_path.write_text(
            """\
data_result
#
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
ATOM C CA ALA A 1 7 X 1.0 2.0 3.0 1
#
""",
            encoding="utf-8",
        )

        atoms = read_mmcif_atoms(cif_path)

        self.assertEqual(len(atoms), 1)
        self.assertEqual(atoms[0].chain_id, "X")
        self.assertEqual(atoms[0].residue_number, 7)
        self.assertEqual(atoms[0].coordinate, (1.0, 2.0, 3.0))

        label_atoms = read_mmcif_atoms(
            cif_path,
            identifier_namespace="label",
        )

        self.assertEqual(label_atoms[0].chain_id, "A")
        self.assertEqual(label_atoms[0].residue_number, 1)

    def test_rejects_unknown_mmcif_identifier_namespace(self) -> None:
        with self.assertRaisesRegex(ValueError, "identifier_namespace"):
            read_mmcif_atoms(
                self.path,
                identifier_namespace="rfd3",
            )


if __name__ == "__main__":
    unittest.main()
