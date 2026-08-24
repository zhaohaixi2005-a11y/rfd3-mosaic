import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rfd3_mosaic.rfd3_central_motif_audit import audit_central_motif
from rfd3_mosaic.rfd3_constraint_orbit_audit import (
    _chain_ids_in_encounter_order,
    _pairwise_distance_matrix_rmsd,
    audit_constraint_orbit,
)
from rfd3_mosaic.structure import AtomRecord

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
    def test_blockwise_distance_matrix_rmsd_matches_dense_formula(self) -> None:
        generator = np.random.default_rng(19)
        expected = generator.normal(size=(37, 3))
        observed = expected + generator.normal(scale=0.08, size=(37, 3))
        expected_distances = np.linalg.norm(
            expected[:, None, :] - expected[None, :, :],
            axis=-1,
        )
        observed_distances = np.linalg.norm(
            observed[:, None, :] - observed[None, :, :],
            axis=-1,
        )
        dense = float(
            np.sqrt(np.mean((observed_distances - expected_distances) ** 2))
        )

        blockwise = _pairwise_distance_matrix_rmsd(
            expected,
            observed,
            block_size=7,
        )

        self.assertAlmostEqual(blockwise, dense, places=12)

    def test_high_order_output_chains_keep_materialization_order(self) -> None:
        """Punctuation chain IDs must not reorder high-order group actions."""

        chain_ids = [chr(ord("A") + index) for index in range(27)]
        # RFD3's legacy mmCIF writer materializes the backslash identifier as
        # a blank label; this is the pair that a lexical sort used to swap.
        chain_ids.append(" ")
        atoms = tuple(
            AtomRecord(
                record_type="ATOM",
                serial=index + 1,
                atom_name="CA",
                alternate_location="",
                residue_name="ALA",
                chain_id=chain_id,
                residue_number=1,
                insertion_code="",
                coordinate=(float(index), 0.0, 0.0),
                element="C",
            )
            for index, chain_id in enumerate(chain_ids)
        )

        self.assertEqual(
            _chain_ids_in_encounter_order(atoms),
            chain_ids,
        )
        self.assertEqual(chain_ids[-2:], ["[", " "])

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

    def test_audits_cross_seam_group_with_two_asu_output_chains(
        self,
    ) -> None:
        """Runtime group members, not chain count, define physical copies."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.cif"
            source.write_text(
                MMCIF_HEADER
                + "ATOM C CA ALA A 1 1 A 1.0 0.0 0.0 1\n"
                + "ATOM N N ALA A 1 1 A 1.0 1.0 0.0 1\n"
                + "ATOM C CA LEU F 1 1 F 3.0 0.0 1.0 1\n"
                + "ATOM N N LEU F 1 1 F 3.0 1.0 1.0 1\n#\n",
                encoding="utf-8",
            )
            identity = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            action_1 = [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            action_2 = [
                [1.0, 0.0, 0.0, 20.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            orbit_id = "cross_seam_component"
            group_ids = [f"cross_seam[{index}]" for index in range(3)]
            groups = []
            for index, right_action in enumerate((1, 2, 0)):
                groups.append(
                    {
                        "constraint_orbit_id": orbit_id,
                        "group_id": group_ids[index],
                        "members": [
                            {
                                "src_components": ["A1"],
                                "sym_transform_id": index,
                            },
                            {
                                "src_components": ["F1"],
                                "sym_transform_id": right_action,
                            },
                        ],
                    }
                )
            probe = root / "rfd3_input.json"
            probe.write_text(
                json.dumps(
                    {
                        "probe": {
                            "input": source.name,
                            "select_fixed_atoms": {
                                "A1-1": "ALL",
                                "F1-1": "ALL",
                            },
                            "extra": {
                                "symmetry_multiplicity": 3,
                                "asu_chain_count": 2,
                                "registry_transform_order": [
                                    "C3:e",
                                    "C3:r1",
                                    "C3:r2",
                                ],
                                "registry_transform_matrices": {
                                    "C3:e": identity,
                                    "C3:r1": action_1,
                                    "C3:r2": action_2,
                                },
                                "motif_constraint_orbits": [
                                    {
                                        "constraint_orbit_id": orbit_id,
                                        "coupling_group_id": "fixed_pair",
                                        "source_components": ["A1", "F1"],
                                        "group_ids": group_ids,
                                        "mobility_mode": "fixed",
                                    }
                                ],
                                "motif_constraint_groups": groups,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            result_json = root / "result.json"
            result_json.write_text(
                json.dumps(
                    {"diffused_index_map": {"A1": "E2", "F1": "F3"}}
                ),
                encoding="utf-8",
            )
            result_structure = root / "result.cif"
            result_structure.write_text(
                MMCIF_HEADER
                # action 0, ASU slots 0/1
                + "ATOM C CA ALA A 2 2 A 1.0 5.0 2.0 1\n"
                + "ATOM N N ALA A 2 2 A 1.0 6.0 2.0 1\n"
                + "ATOM C CA LEU B 3 3 B 3.0 5.0 3.0 1\n"
                + "ATOM N N LEU B 3 3 B 3.0 6.0 3.0 1\n"
                # action 1, ASU slots 0/1
                + "ATOM C CA ALA C 2 2 C 11.0 5.0 2.0 1\n"
                + "ATOM N N ALA C 2 2 C 11.0 6.0 2.0 1\n"
                + "ATOM C CA LEU D 3 3 D 13.0 5.0 3.0 1\n"
                + "ATOM N N LEU D 3 3 D 13.0 6.0 3.0 1\n"
                # action 2, ASU slots 0/1
                + "ATOM C CA ALA E 2 2 E 21.0 5.0 2.0 1\n"
                + "ATOM N N ALA E 2 2 E 21.0 6.0 2.0 1\n"
                + "ATOM C CA LEU F 3 3 F 23.0 5.0 3.0 1\n"
                + "ATOM N N LEU F 3 3 F 23.0 6.0 3.0 1\n#\n",
                encoding="utf-8",
            )

            report = audit_constraint_orbit(
                compiled_input=probe,
                result_json=result_json,
                result_structure=result_structure,
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["summary"]["asu_chain_count"], 2)
            self.assertEqual(
                report["summary"]["output_chains"],
                ["A", "B", "C", "D", "E", "F"],
            )
            self.assertEqual(report["summary"]["matched_heavy_atoms"], 12)
            self.assertLess(report["summary"]["joint_orbit_rmsd"], 1e-6)

    def test_quotient_uses_direct_runtime_fixed_target_contract(self) -> None:
        """A physical quotient must not re-transform a materialized source."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.cif"
            source.write_text(
                MMCIF_HEADER
                + "ATOM C CA ALA A 1 1 A 1.0 0.0 0.0 1\n#\n",
                encoding="utf-8",
            )
            identity = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            action = [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            orbit_id = "quotient_orbit"
            probe = root / "rfd3_input.json"
            probe.write_text(
                json.dumps(
                    {
                        "probe": {
                            "input": source.name,
                            "select_fixed_atoms": {"A1-1": "ALL"},
                            "extra": {
                                "symmetry_multiplicity": 2,
                                "symmetry_action_kind": "stabilizer_quotient",
                                "registry_transform_order": ["C4:e", "C4:r1"],
                                "registry_transform_matrices": {
                                    "C4:e": identity,
                                    "C4:r1": action,
                                },
                                "motif_constraint_orbits": [
                                    {
                                        "constraint_orbit_id": orbit_id,
                                        "coupling_group_id": "fixed_seed",
                                        "source_components": ["A1"],
                                        "group_ids": ["q[0]", "q[1]"],
                                        "mobility_mode": "fixed",
                                    }
                                ],
                                "motif_constraint_groups": [
                                    {
                                        "constraint_orbit_id": orbit_id,
                                        "group_id": "q[0]",
                                        "members": [
                                            {
                                                "src_components": ["A1"],
                                                "sym_transform_id": 0,
                                            }
                                        ],
                                    },
                                    {
                                        "constraint_orbit_id": orbit_id,
                                        "group_id": "q[1]",
                                        "members": [
                                            {
                                                "src_components": ["A1"],
                                                "sym_transform_id": 1,
                                            }
                                        ],
                                    },
                                ],
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            result_json = root / "result.json"
            result_json.write_text(
                json.dumps(
                    {
                        "diffused_index_map": {"A1": "A1"},
                        "constraint_runtime_diagnostics": {
                            "schema_version": 2,
                            "state": "finalized",
                            "final_fixed_target_rmsd": 0.0,
                            "final_fixed_target_maximum_error": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            result_structure = root / "result.cif"
            result_structure.write_text(
                MMCIF_HEADER
                + "ATOM C CA ALA A 1 1 A 1.0 0.0 0.0 1\n"
                + "ATOM C CA ALA B 1 1 B 21.0 0.0 0.0 1\n#\n",
                encoding="utf-8",
            )

            report = audit_constraint_orbit(
                compiled_input=probe,
                result_json=result_json,
                result_structure=result_structure,
            )

            self.assertTrue(report["passed"])
            component = report["summary"]["constraint_components"][0]
            self.assertEqual(
                component["acceptance_reference"],
                "runtime_fixed_target",
            )
            self.assertEqual(component["joint_orbit_rmsd"], 0.0)
            self.assertGreater(
                component["legacy_reference_joint_orbit_rmsd"],
                1.0,
            )

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
