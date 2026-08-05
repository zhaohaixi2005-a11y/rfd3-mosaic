import json
from pathlib import Path
import tempfile
import unittest

from rfd3_mosaic.rfd3_central_motif_audit import audit_central_motif


MMCIF_HEADER = """\
data_structure
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
"""


class CentralMotifAuditTestCase(unittest.TestCase):
    def test_uses_rfd3_label_numbering_for_fixed_selector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.cif"
            source.write_text(
                MMCIF_HEADER
                + "ATOM C CA ALA B 1 211 B 1.0 0.0 0.0 1\n#\n",
                encoding="utf-8",
            )
            probe = root / "rfd3_input.json"
            probe.write_text(
                json.dumps(
                    {
                        "probe": {
                            "input": source.name,
                            "extra": {
                                "probe_fixed_selector": "B1-1",
                                "symmetry_multiplicity": 2,
                                "registry_transform_order": [
                                    "C2:e",
                                    "C2:r1",
                                ],
                                "registry_transform_matrices": {
                                    "C2:e": [
                                        [1.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ],
                                    "C2:r1": [
                                        [1.0, 0.0, 0.0, 10.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ],
                                },
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            result_json = root / "result.json"
            result_json.write_text(
                json.dumps({"diffused_index_map": {"B1": "A2"}}),
                encoding="utf-8",
            )
            result_structure = root / "result.cif"
            result_structure.write_text(
                MMCIF_HEADER
                + "ATOM C CA ALA A 2 2 A 1.0 0.0 0.0 1\n"
                + "ATOM C CA ALA B 2 2 B 11.0 0.0 0.0 1\n#\n",
                encoding="utf-8",
            )

            report = audit_central_motif(
                probe_input=probe,
                result_json=result_json,
                result_structure=result_structure,
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["summary"]["matched_heavy_atoms"], 2)
            self.assertEqual(report["summary"]["joint_orbit_rmsd"], 0.0)

    def test_audits_multiple_fixed_selectors_as_one_complete_orbit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.cif"
            source.write_text(
                MMCIF_HEADER
                + "ATOM C CA ALA B 1 12 B 1.0 0.0 0.0 1\n"
                + "ATOM C CA LEU C 1 26 C 2.0 0.0 0.0 1\n#\n",
                encoding="utf-8",
            )
            probe = root / "rfd3_input.json"
            probe.write_text(
                json.dumps(
                    {
                        "probe": {
                            "input": source.name,
                            "select_fixed_atoms": {
                                "B1-1": "ALL",
                                "C1-1": "ALL",
                            },
                            "extra": {
                                "symmetry_multiplicity": 2,
                                "registry_transform_order": [
                                    "C2:e",
                                    "C2:r1",
                                ],
                                "registry_transform_matrices": {
                                    "C2:e": [
                                        [1.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ],
                                    "C2:r1": [
                                        [1.0, 0.0, 0.0, 10.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ],
                                },
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            result_json = root / "result.json"
            result_json.write_text(
                json.dumps(
                    {"diffused_index_map": {"B1": "A2", "C1": "A3"}}
                ),
                encoding="utf-8",
            )
            result_structure = root / "result.cif"
            result_structure.write_text(
                MMCIF_HEADER
                + "ATOM C CA ALA A 2 2 A 1.0 0.0 0.0 1\n"
                + "ATOM C CA LEU A 3 3 A 2.0 0.0 0.0 1\n"
                + "ATOM C CA ALA B 2 2 B 11.0 0.0 0.0 1\n"
                + "ATOM C CA LEU B 3 3 B 12.0 0.0 0.0 1\n#\n",
                encoding="utf-8",
            )

            report = audit_central_motif(
                probe_input=probe,
                result_json=result_json,
                result_structure=result_structure,
            )

            self.assertTrue(report["passed"])
            self.assertEqual(
                report["inputs"]["fixed_selectors"],
                ["B1-1", "C1-1"],
            )
            self.assertEqual(report["summary"]["matched_heavy_atoms"], 4)
            self.assertEqual(
                report["summary"]["joint_coordinate_maximum_error"],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
