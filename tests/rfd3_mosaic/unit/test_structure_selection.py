import gzip
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rfd3_mosaic.structure import (
    parse_atom_selection,
    read_mmcif_atoms,
    read_pdb_atoms,
    read_structure_atoms,
    select_atom_subset,
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

    def test_heavy_selection_uses_element_before_atom_name_heuristic(self) -> None:
        atom = read_pdb_atoms(self.path)[0]
        mercury = replace(atom, atom_name="HG", element="HG")
        carbon = replace(atom, atom_name="H1", element="C")
        hydrogen = replace(atom, atom_name="H1", element="H")
        deuterium = replace(atom, atom_name="D1", element="D")
        inferred_hydrogen = replace(atom, atom_name="1HA", element="")
        self.assertEqual(
            select_atom_subset(
                (mercury, carbon, hydrogen, deuterium, inferred_hydrogen), "heavy"
            ),
            (mercury, carbon),
        )

    def test_pdb_reader_never_merges_atoms_from_later_models(self) -> None:
        lines = PDB_TEXT.splitlines()
        self.path.write_text(
            "MODEL        7\n"
            + lines[2]
            + "\nENDMDL\nMODEL        8\n"
            + lines[2][:16]
            + " "
            + lines[2][17:30]
            + "  99.000"
            + lines[2][38:]
            + "\n"
            + lines[0]
            + "\nENDMDL\n"
        )
        atoms = read_pdb_atoms(self.path)
        self.assertEqual(len(atoms), 1)
        self.assertEqual(atoms[0].atom_name, "CB")
        self.assertEqual(atoms[0].coordinate, (11.0, 12.0, 12.0))

    def test_pdb_reader_retains_single_non_a_conformer(self) -> None:
        self.path.write_text(PDB_TEXT.splitlines()[3] + "\nEND\n")
        atoms = read_pdb_atoms(self.path)
        self.assertEqual(len(atoms), 1)
        self.assertEqual(atoms[0].alternate_location, "B")

    def test_compressed_pdb_matches_plain_input(self) -> None:
        compressed = self.path.with_suffix(".PDB.GZ")
        with gzip.open(compressed, "wt") as handle:
            handle.write(PDB_TEXT)
        self.assertEqual(read_structure_atoms(compressed), read_pdb_atoms(self.path))

    def test_nonfinite_pdb_coordinates_fail_before_compilation(self) -> None:
        self.path.write_text(PDB_TEXT.replace("  10.000", "     nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            read_pdb_atoms(self.path)

    def _write_cif(self, rows: str) -> Path:
        path = self.path.with_suffix(".cif")
        path.write_text(
            "data_test\nloop_\n"
            + "\n".join(
                "  _atom_site." + name
                for name in (
                    "group_PDB",
                    "label_atom_id",
                    "label_alt_id",
                    "label_comp_id",
                    "label_asym_id",
                    "label_seq_id",
                    "Cartn_x",
                    "Cartn_y",
                    "Cartn_z",
                    "pdbx_PDB_model_num",
                )
            )
            + "\n"
            + rows
            + "\n#\n"
        )
        return path

    def test_cif_first_model_is_not_assumed_to_be_number_one(self) -> None:
        path = self._write_cif("ATOM CA . ALA A 1 1 2 3 7\nATOM N . ALA A 1 99 99 99 8")
        atoms = read_mmcif_atoms(path)
        self.assertEqual(
            [(a.atom_name, a.coordinate) for a in atoms], [("CA", (1.0, 2.0, 3.0))]
        )

    def test_cif_conformers_match_pdb_policy_without_mixing_residue(self) -> None:
        path = self._write_cif(
            "ATOM N . ALA A 1 0 0 0 1\n"
            "ATOM CA B ALA A 1 99 0 0 1\n"
            "ATOM CA A ALA A 1 1 0 0 1\n"
            "ATOM CB B ALA A 1 99 1 0 1\n"
            "ATOM CA B GLY A 2 2 0 0 1"
        )
        atoms = read_mmcif_atoms(path)
        self.assertEqual(
            [(a.atom_name, a.residue_number) for a in atoms],
            [("N", 1), ("CA", 1), ("CA", 2)],
        )
        self.assertEqual(atoms[1].coordinate, (1.0, 0.0, 0.0))

    def test_cif_wrapped_rows_quotes_comments_and_apostrophes(self) -> None:
        path = self._write_cif(
            "HETATM O5' . 'LIG' A 1\n"
            "1 2 3 1 # inline comment\n"
            'HETATM "C5\'" . LIG A 1 4 5 6 1'
        )
        atoms = read_mmcif_atoms(path)
        self.assertEqual([a.atom_name for a in atoms], ["O5'", "C5'"])
        self.assertEqual(atoms[1].coordinate, (4.0, 5.0, 6.0))

    def test_cif_rejects_incomplete_rows(self) -> None:
        path = self._write_cif("ATOM CA . ALA A 1 1 2")
        with self.assertRaisesRegex(ValueError, "Incomplete mmCIF atom row"):
            read_mmcif_atoms(path)

    def test_cif_rejects_nonfinite_coordinates(self) -> None:
        path = self._write_cif("ATOM CA . ALA A 1 nan 2 3 1")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            read_mmcif_atoms(path)

    def test_cif_does_not_hide_duplicate_atom_identity(self) -> None:
        row = "ATOM CA . ALA A 1 1 2 3 1"
        path = self._write_cif(row + "\n" + row)
        with self.assertRaisesRegex(ValueError, "Duplicate atom identity"):
            read_mmcif_atoms(path)

    def test_cif_does_not_merge_ambiguous_residue_names(self) -> None:
        path = self._write_cif(
            "ATOM CA . ALA A 1 1 2 3 1\nATOM N . GLY A 1 4 5 6 1"
        )
        with self.assertRaisesRegex(ValueError, "Ambiguous residue identity"):
            read_mmcif_atoms(path)

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

    def test_missing_residue_cannot_be_silently_bridged_in_fragment(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing residues.*167"):
            select_atoms(read_pdb_atoms(self.path), "A/165-167/*")

    def test_atom_specific_selection_cannot_silently_drop_a_residue(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing residues.*166"):
            select_atoms(read_pdb_atoms(self.path), "A/165-166/CB")

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
