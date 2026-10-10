import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rfd3_mosaic.rfd3_central_motif_audit import audit_central_motif
from rfd3_mosaic.rfd3_constraint_orbit_audit import (
    _chain_ids_in_encounter_order,
    _pairwise_distance_matrix_rmsd,
    _parse_selector,
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
    def test_enumerated_fixed_residues_preserve_gaps_and_audit_selected_atoms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source.cif").write_text(MMCIF_HEADER + "\n".join(
                f"ATOM C CA ALA A {i} {i} A {float(i)} 0.0 0.0 1"
                for i in (1, 2, 3)
            ) + "\n#\n")
            example = {
                "input": "source.cif", "select_fixed_atoms": {"A1,A3": "ALL"},
                "extra": {
                    "symmetry_multiplicity": 2,
                    "registry_transform_order": ["C2:e", "C2:r1"],
                    "registry_transform_matrices": {
                        "C2:e": np.eye(4).tolist(),
                        "C2:r1": np.diag([-1.0, -1.0, 1.0, 1.0]).tolist(),
                    },
                },
            }
            (root / "input.json").write_text(json.dumps({"case": example}))
            (root / "result.json").write_text(json.dumps({
                "diffused_index_map": {f"A{i}": f"A{i}" for i in (1, 2, 3)}
            }))
            lines = [
                f"ATOM C CA ALA {chain} {i} {i} {chain} {sign * (20 if i == 2 else i)} 0.0 0.0 1"
                for chain, sign in (("A", 1), ("B", -1)) for i in (1, 2, 3)
            ]
            structure = root / "result.cif"
            structure.write_text(MMCIF_HEADER + "\n".join(lines) + "\n#\n")
            arguments = {"compiled_input": root / "input.json", "result_json": root / "result.json"}
            report = audit_constraint_orbit(**arguments)
            self.assertTrue(report["passed"])
            self.assertEqual(report["summary"]["matched_heavy_atoms"], 4)
            # The unselected middle residue may move; a selected residue may not.
            lines[-1] = "ATOM C CA ALA B 3 3 B -3.0 8.0 0.0 1"
            structure.write_text(MMCIF_HEADER + "\n".join(lines) + "\n#\n")
            self.assertFalse(audit_constraint_orbit(**arguments)["passed"])
            # A truncated source must not reduce the denominator and pass.
            source = root / "source.cif"
            source.write_text(source.read_text().replace(
                "ATOM C CA ALA A 1 1 A 1.0 0.0 0.0 1\n", ""
            ))
            with self.assertRaisesRegex(ValueError, "missing from source"):
                audit_constraint_orbit(**arguments)

    def test_fixed_selector_rejects_malformed_or_reversed_components(self):
        for selector in ("", "A3-1", "A1-", "A", "A1+3"):
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                _parse_selector(selector)

    def test_preexpanded_mobile_interface_uses_physical_cross_chain_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coordinates = {"A1": (1, 0), "A3": (3, -1), "B1": (-1, 0), "B3": (-3, 1)}

            def write_structure(path, points):
                path.write_text(MMCIF_HEADER + "\n".join(
                    f"ATOM {element} {name} ALA {key[0]} {key[1:]} {key[1:]} {key[0]} {x + dx} {y + dy} {dz} 1"
                    for key, (x, y) in points.items()
                    for name, element, dx, dy, dz in (
                        ("CA", "C", 0.0, 0.0, 0.0),
                        ("N", "N", 0.2, 0.3, 0.4),
                    )
                ) + "\n#\n")

            write_structure(root / "source.cif", coordinates)
            groups = [
                {"group_id": "g0", "constraint_orbit_id": "interface", "members": [
                    {"src_components": ["A1"], "sym_transform_id": 0},
                    {"src_components": ["B3"], "sym_transform_id": 1},
                ]},
                {"group_id": "g1", "constraint_orbit_id": "interface", "members": [
                    {"src_components": ["B1"], "sym_transform_id": 1},
                    {"src_components": ["A3"], "sym_transform_id": 0},
                ]},
            ]
            orbit = {"constraint_orbit_id": "interface", "source_components": ["A1", "A3"],
                     "group_ids": ["g0", "g1"], "mobility_mode": "orbit_rigid",
                     "max_translation": 3.0, "max_rotation_deg": 10.0}
            example = {"input": "source.cif", "select_fixed_atoms": {"A1,A3": "ALL", "B1,B3": "ALL"},
                       "extra": {"symmetry_multiplicity": 2,
                                 "preexpanded_chain_layout": [{"is_asu": True}, {"is_asu": False}],
                                 "registry_transform_order": ["e", "r"],
                                 "registry_transform_matrices": {"e": np.eye(4).tolist(), "r": np.diag([-1., -1., 1., 1.]).tolist()},
                                 "motif_constraint_groups": groups, "motif_constraint_orbits": [orbit]}}
            compiled = root / "input.json"
            compiled.write_text(json.dumps({"case": example}))
            result = root / "result.json"
            result.write_text(json.dumps({"diffused_index_map": {key: key for key in coordinates}}))
            moved = {key: (x + (2 if key in {"A1", "B3"} else -2), y)
                     for key, (x, y) in coordinates.items()}
            write_structure(root / "result.cif", moved)
            arguments = {"compiled_input": compiled, "result_json": result}
            report = audit_constraint_orbit(**arguments)
            self.assertTrue(report["passed"])
            self.assertEqual(report["summary"]["matched_heavy_atoms"], 8)
            self.assertEqual(len(report["summary"]["per_copy_internal_rmsd"]), 2)
            self.assertGreater(report["summary"]["joint_orbit_rmsd"], 0.5)
            orbit["mobility_mode"] = "fixed"
            compiled.write_text(json.dumps({"case": example}))
            self.assertFalse(audit_constraint_orbit(**arguments)["passed"])
            orbit["mobility_mode"] = "orbit_rigid"
            compiled.write_text(json.dumps({"case": example}))
            moved["A1"] = (9, 0)
            write_structure(root / "result.cif", moved)
            self.assertFalse(audit_constraint_orbit(**arguments)["passed"])

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
                + "ATOM N N ALA B 1 1 B 1.0 1.0 0.0 1\n"
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
                "max_translation": 20.0,
                "max_rotation_deg": 10.0,
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
                + "ATOM N N ALA A 2 2 A 6.0 1.0 0.0 1\n"
                + "ATOM C CA LEU A 3 3 A 7.0 0.0 0.0 1\n"
                + "ATOM C CA ALA B 2 2 B 26.0 0.0 0.0 1\n"
                + "ATOM N N ALA B 2 2 B 26.0 1.0 0.0 1\n"
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
            self.assertLess(component["maximum_per_copy_internal_rmsd"], 1e-12)
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


class IndependentMobilePoseBoundsTestCase(unittest.TestCase):
    """Final CIF evidence must not inherit a stale runtime pose claim."""

    def _case(self, root, *, translation=1.0, rotation=0.0, collinear=False):
        points = {}
        for chain, sign in (("A", 1), ("B", -1)):
            for residue, atom, xyz in (
                (1, "CA", (10.0, 0.0, 0.0)),
                (1, "N", (9.0, 1.0, 0.0)),
                (3, "CA", (12.0, 2.0, 0.0)),
                (3, "N", (13.0, 1.0, 1.0)),
            ):
                points[(chain, residue, atom)] = np.asarray(
                    (sign * xyz[0], sign * xyz[1], xyz[2])
                    if not collinear else (sign * xyz[0], 0.0, 0.0)
                )
        groups = (("A1", "B3"), ("B1", "A3"))

        def write(path, coordinates):
            path.write_text(MMCIF_HEADER + "\n".join(
                f"ATOM {atom[0]} {atom} ALA {chain} {residue} {residue} "
                f"{chain} {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f} 1"
                for (chain, residue, atom), xyz in coordinates.items()
            ) + "\n#\n")

        write(root / "source.cif", points)
        moved = dict(points)
        angle = np.radians(rotation)
        matrix = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        for index, residues in enumerate(groups):
            keys = [key for key in points if f"{key[0]}{key[1]}" in residues]
            center = np.asarray([points[key] for key in keys]).mean(axis=0)
            shift = np.asarray(((1 - 2 * index) * translation, 0.0, 0.0))
            for key in keys:
                moved[key] = (points[key] - center) @ matrix.T + center + shift
        write(root / "result.cif", moved)
        orbit = {
            "constraint_orbit_id": "joint", "coupling_group_id": "seed",
            "source_components": ["A1", "A3"], "group_ids": ["g0", "g1"],
            "mobility_mode": "orbit_rigid", "mobility_subspace": "bounded_se3",
            "max_translation": 3.0, "max_rotation_deg": 10.0,
        }
        example = {
            "input": "source.cif",
            "select_fixed_atoms": {"A1,A3": "ALL", "B1,B3": "ALL"},
            "extra": {
                "symmetry_multiplicity": 2,
                "preexpanded_chain_layout": [{"is_asu": True}, {"is_asu": False}],
                "registry_transform_order": ["C2:e", "C2:r1"],
                "registry_transform_matrices": {
                    "C2:e": np.eye(4).tolist(),
                    "C2:r1": np.diag([-1., -1., 1., 1.]).tolist(),
                },
                "motif_constraint_orbits": [orbit],
                "motif_constraint_groups": [
                    {"group_id": f"g{index}", "constraint_orbit_id": "joint",
                     "members": [{"src_components": [residue],
                                  "sym_transform_id": int(residue[0] == "B")}
                                 for residue in residues]}
                    for index, residues in enumerate(groups)
                ],
            },
        }
        compiled = root / "input.json"
        compiled.write_text(json.dumps({"case": example}))
        result = root / "result.json"
        result.write_text(json.dumps({
            "diffused_index_map": {f"{c}{r}": f"{c}{r}" for c, r, _ in points},
            "motif_mobility_diagnostics": {"orbits": [{
                "translation_norms": [0.0], "rotation_degrees": [0.0],
            }]},
        }))
        return {"compiled_input": compiled, "result_json": result}, example

    def test_complete_joint_translation_exceeding_bounds_rejects_stale_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, _ = self._case(root, translation=1.0)
            self.assertTrue(audit_constraint_orbit(**arguments)["passed"])
            original_metadata = arguments["result_json"].read_bytes()
            arguments, _ = self._case(root, translation=8.0)
            self.assertEqual(arguments["result_json"].read_bytes(), original_metadata)
            report = audit_constraint_orbit(**arguments)
            self.assertFalse(report["passed"])
            component = report["summary"]["constraint_components"][0]
            self.assertLess(component["maximum_per_copy_internal_rmsd"], 1e-10)
            measured = component["independent_mobile_pose_bounds"]
            self.assertFalse(measured["directional_subspace_measured"])
            self.assertEqual(len(measured["copies"]), 2)
            for copy in measured["copies"]:
                self.assertAlmostEqual(copy["translation_norm_angstrom"], 8.0)
                self.assertFalse(copy["passed"])

    def test_rotation_about_joint_centroid_is_not_origin_translation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for angle, passed in ((9.0, True), (30.0, False)):
                with self.subTest(angle=angle):
                    arguments, _ = self._case(root, translation=0.0, rotation=angle)
                    report = audit_constraint_orbit(**arguments)
                    self.assertEqual(report["passed"], passed)
                    measured = report["summary"]["constraint_components"][0][
                        "independent_mobile_pose_bounds"
                    ]
                    for copy in measured["copies"]:
                        self.assertLess(copy["translation_norm_angstrom"], 1e-8)
                        self.assertAlmostEqual(copy["rotation_degrees"], angle, places=7)

    def test_explicit_serialization_tolerances_at_both_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            for translation, rotation, passed in (
                (3.0, 10.0, True), (3.0019, 10.019, True),
                (3.0021, 10.0, False), (3.0, 10.021, False),
            ):
                with self.subTest(translation=translation, rotation=rotation):
                    arguments, _ = self._case(
                        Path(temporary), translation=translation, rotation=rotation
                    )
                    self.assertEqual(audit_constraint_orbit(**arguments)["passed"], passed)

    def test_missing_invalid_or_nonfinite_bounds_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            for key, value in (
                ("max_translation", "missing"), ("max_rotation_deg", "missing"),
                ("max_translation", float("nan")), ("max_rotation_deg", float("inf")),
                ("max_translation", True), ("max_translation", -1.0),
                ("max_rotation_deg", 181.0), ("max_rotation_deg", None),
            ):
                with self.subTest(key=key, value=value):
                    arguments, example = self._case(Path(temporary))
                    orbit = example["extra"]["motif_constraint_orbits"][0]
                    if value == "missing":
                        orbit.pop(key)
                    else:
                        orbit[key] = value
                    arguments["compiled_input"].write_text(json.dumps({"case": example}))
                    self.assertFalse(audit_constraint_orbit(**arguments)["passed"])

    def test_explicit_null_rotation_bound_freezes_radial_rotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            for angle, passed in ((0.0, True), (1.0, False)):
                with self.subTest(angle=angle):
                    arguments, example = self._case(Path(temporary), rotation=angle)
                    orbit = example["extra"]["motif_constraint_orbits"][0]
                    orbit.update(mobility_subspace="radial", max_rotation_deg=None)
                    arguments["compiled_input"].write_text(json.dumps({"case": example}))
                    self.assertEqual(audit_constraint_orbit(**arguments)["passed"], passed)

    def test_collinear_joint_cannot_certify_rotation_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments, _ = self._case(Path(temporary), collinear=True)
            report = audit_constraint_orbit(**arguments)
            self.assertFalse(report["passed"])
            measured = report["summary"]["constraint_components"][0][
                "independent_mobile_pose_bounds"
            ]
            self.assertTrue(all("underdetermined" in c["failure"] for c in measured["copies"]))

    def test_fullnoise_native_source_centering_and_partial_declared_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for partial in (False, True):
                with self.subTest(partial=partial):
                    arguments, example = self._case(root, translation=3.0)
                    example["symmetry"] = {"id": "C2"}
                    if partial:
                        example["partial_t"] = 2.0
                        example["extra"]["generated_coordinate_initialization"] = (
                            "complete_scaffold_partial_diffusion"
                        )
                    else:
                        # The eight source atoms have z-COM = 0.25. Native
                        # full-noise input loading removes it before diffusion.
                        lines = (root / "result.cif").read_text().splitlines()
                        for index, line in enumerate(lines):
                            if line.startswith("ATOM"):
                                fields = line.split()
                                fields[10] = str(float(fields[10]) - 0.25)
                                lines[index] = " ".join(fields)
                        (root / "result.cif").write_text("\n".join(lines) + "\n")
                    arguments["compiled_input"].write_text(json.dumps({"case": example}))
                    report = audit_constraint_orbit(**arguments)
                    self.assertTrue(report["passed"])
                    measured = report["summary"]["constraint_components"][0][
                        "independent_mobile_pose_bounds"
                    ]
                    for copy in measured["copies"]:
                        self.assertAlmostEqual(copy["translation_norm_angstrom"], 3.0)
                    self.assertEqual(
                        measured["reference_frame"]["source_protein_center_subtracted"],
                        [0.0, 0.0, 0.0 if partial else 0.25],
                    )

    def test_unreconstructed_origin_strategy_is_not_certified(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments, example = self._case(Path(temporary))
            example.update(symmetry={"id": "C2"}, infer_ori_strategy="hotspots")
            arguments["compiled_input"].write_text(json.dumps({"case": example}))
            report = audit_constraint_orbit(**arguments)
            self.assertFalse(report["passed"])
            measured = report["summary"]["constraint_components"][0][
                "independent_mobile_pose_bounds"
            ]
            self.assertIn("unavailable", measured["failure"])

    def test_malformed_mobile_declaration_cannot_fall_back_to_fixed_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for malformed in (None, "A1,A3", {}, []):
                with self.subTest(malformed=malformed):
                    arguments, example = self._case(root, translation=0.0)
                    lines = (root / "result.cif").read_text().splitlines()
                    for index, line in enumerate(lines):
                        if line.startswith("ATOM"):
                            fields = line.split()
                            fields[10] = str(float(fields[10]) + 100.0)
                            lines[index] = " ".join(fields)
                    (root / "result.cif").write_text("\n".join(lines) + "\n")
                    self.assertFalse(audit_constraint_orbit(**arguments)["passed"])
                    orbit = example["extra"]["motif_constraint_orbits"][0]
                    if malformed is None:
                        orbit.pop("source_components")
                    else:
                        orbit["source_components"] = malformed
                    arguments["compiled_input"].write_text(json.dumps({"case": example}))
                    with self.assertRaisesRegex(ValueError, "metadata is malformed|no source atoms"):
                        audit_constraint_orbit(**arguments)


if __name__ == "__main__":
    unittest.main()
