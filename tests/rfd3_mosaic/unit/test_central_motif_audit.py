import json
from pathlib import Path
import tempfile
import unittest

from rfd3_mosaic.rfd3_central_motif_audit import audit_central_motif
from rfd3_mosaic.rfd3_constraint_orbit_audit import (
    audit_constraint_orbit,
)


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
                + "ATOM C CA ALA A 2 2 A 8.0 0.0 0.0 1\n"
                + "ATOM C CA ALA B 2 2 B 18.0 0.0 0.0 1\n#\n",
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
            self.assertNotIn(
                "joint_coordinate_maximum_error", report["summary"]
            )

            generic = audit_constraint_orbit(
                compiled_input=probe,
                result_json=result_json,
                result_structure=result_structure,
            )
            self.assertTrue(generic["passed"])
            self.assertEqual(
                generic["audit"],
                "rfd3_mosaic.fixed_constraint_orbit",
            )
            self.assertEqual(
                generic["inputs"]["compiled_input"],
                str(probe.resolve()),
            )
            self.assertNotIn("probe_input", generic["inputs"])
            self.assertEqual(generic["summary"], report["summary"])
            self.assertEqual(generic["thresholds"], report["thresholds"])

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

    def test_independent_components_allow_independent_rigid_gauges(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.cif"
            source.write_text(
                MMCIF_HEADER
                + "ATOM C CA ALA B 1 1 B 1.0 0.0 0.0 1\n"
                + "ATOM C CA LEU C 1 1 C 2.0 0.0 0.0 1\n#\n",
                encoding="utf-8",
            )
            identity = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            copy = [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            payload = {
                "probe": {
                    "input": source.name,
                    "select_fixed_atoms": {
                        "B1-1": "ALL",
                        "C1-1": "ALL",
                    },
                    "extra": {
                        "symmetry_multiplicity": 2,
                        "registry_transform_order": ["C2:e", "C2:r1"],
                        "registry_transform_matrices": {
                            "C2:e": identity,
                            "C2:r1": copy,
                        },
                        "motif_constraint_orbits": [
                            {
                                "constraint_orbit_id": "component_b",
                                "coupling_group_id": "component_b",
                                "source_components": ["B1"],
                            },
                            {
                                "constraint_orbit_id": "component_c",
                                "coupling_group_id": "component_c",
                                "source_components": ["C1"],
                            },
                        ],
                    },
                }
            }
            probe = root / "rfd3_input.json"
            probe.write_text(json.dumps(payload), encoding="utf-8")
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
                + "ATOM C CA ALA A 2 2 A 6.0 0.0 0.0 1\n"
                + "ATOM C CA LEU A 3 3 A -1.0 0.0 0.0 1\n"
                + "ATOM C CA ALA B 2 2 B 16.0 0.0 0.0 1\n"
                + "ATOM C CA LEU B 3 3 B 9.0 0.0 0.0 1\n#\n",
                encoding="utf-8",
            )

            independent = audit_constraint_orbit(
                compiled_input=probe,
                result_json=result_json,
                result_structure=result_structure,
            )

            self.assertTrue(independent["passed"])
            self.assertEqual(
                independent["summary"]["constraint_component_count"],
                2,
            )
            self.assertEqual(
                [
                    item["joint_orbit_rmsd"]
                    for item in independent["summary"][
                        "constraint_components"
                    ]
                ],
                [0.0, 0.0],
            )

            payload["probe"]["extra"].pop("motif_constraint_orbits")
            probe.write_text(json.dumps(payload), encoding="utf-8")
            jointly_locked = audit_constraint_orbit(
                compiled_input=probe,
                result_json=result_json,
                result_structure=result_structure,
            )
            self.assertFalse(jointly_locked["passed"])

    def test_mobile_component_preserves_each_copy_not_initial_orbit_pose(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.cif"
            source.write_text(
                MMCIF_HEADER
                + "ATOM C CA ALA B 1 1 B 1.0 0.0 0.0 1\n"
                + "ATOM C CA LEU B 2 2 B 2.0 0.0 0.0 1\n#\n",
                encoding="utf-8",
            )
            identity = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            copy = [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            orbit = {
                "constraint_orbit_id": "mobile_b",
                "coupling_group_id": "mobile_b",
                "source_components": ["B1", "B2"],
                "mobility_mode": "orbit_rigid",
            }
            payload = {
                "probe": {
                    "input": source.name,
                    "select_fixed_atoms": {"B1-2": "ALL"},
                    "extra": {
                        "symmetry_multiplicity": 2,
                        "registry_transform_order": ["C2:e", "C2:r1"],
                        "registry_transform_matrices": {
                            "C2:e": identity,
                            "C2:r1": copy,
                        },
                        "motif_constraint_orbits": [orbit],
                    },
                }
            }
            probe = root / "rfd3_input.json"
            probe.write_text(json.dumps(payload), encoding="utf-8")
            result_json = root / "result.json"
            result_json.write_text(
                json.dumps(
                    {"diffused_index_map": {"B1": "A2", "B2": "A3"}}
                ),
                encoding="utf-8",
            )
            result_structure = root / "result.cif"
            result_structure.write_text(
                MMCIF_HEADER
                + "ATOM C CA ALA A 2 2 A 6.0 0.0 0.0 1\n"
                + "ATOM C CA LEU A 3 3 A 7.0 0.0 0.0 1\n"
                + "ATOM C CA ALA B 2 2 B 26.0 0.0 0.0 1\n"
                + "ATOM C CA LEU B 3 3 B 27.0 0.0 0.0 1\n#\n",
                encoding="utf-8",
            )

            mobile = audit_constraint_orbit(
                compiled_input=probe,
                result_json=result_json,
                result_structure=result_structure,
            )
            self.assertTrue(mobile["passed"])
            component = mobile["summary"]["constraint_components"][0]
            self.assertGreater(component["joint_orbit_rmsd"], 0.5)
            self.assertEqual(component["maximum_per_copy_internal_rmsd"], 0.0)
            self.assertEqual(
                component["geometry_contract"],
                "per_copy_rigid_with_bounded_orbit_pose",
            )

            orbit["mobility_mode"] = "fixed"
            probe.write_text(json.dumps(payload), encoding="utf-8")
            static = audit_constraint_orbit(
                compiled_input=probe,
                result_json=result_json,
                result_structure=result_structure,
            )
            self.assertFalse(static["passed"])


if __name__ == "__main__":
    unittest.main()
