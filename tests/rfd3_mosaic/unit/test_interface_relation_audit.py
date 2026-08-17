import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rfd3_mosaic.rfd3_interface_relation_audit import (
    audit_interface_relations,
)


def _atom_line(
    serial: int,
    chain: str,
    residue: int,
    coordinate: tuple[float, float, float],
) -> str:
    x, y, z = coordinate
    return (
        f"ATOM  {serial:5d}  CA  ALA {chain:1s}{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}"
        "           C\n"
    )


class InterfaceRelationAuditTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "presymmetrized_input.pdb"
        left = np.asarray(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        )
        right = left + np.asarray([4.0, 0.0, 0.0])
        self.left = left
        self.right = right
        self.source.write_text(
            "".join(
                [
                    _atom_line(index, "A", index, tuple(coordinate))
                    for index, coordinate in enumerate(left, start=1)
                ]
                + [
                    _atom_line(index + 3, "B", index, tuple(coordinate))
                    for index, coordinate in enumerate(right, start=1)
                ]
            )
            + "END\n",
            encoding="utf-8",
        )
        self.result_json = self.root / "result_0_model_0.json"
        self.result_json.write_text(
            json.dumps(
                {
                    "diffused_index_map": {
                        **{f"A{i}": f"B{i}" for i in range(1, 4)},
                        **{f"B{i}": f"B{i + 3}" for i in range(1, 4)},
                    }
                }
            ),
            encoding="utf-8",
        )
        self.result_structure = self.root / "result_0_model_0.pdb"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _transform(coordinates: np.ndarray, copy_index: int) -> np.ndarray:
        if copy_index == 0:
            return coordinates.copy()
        return coordinates @ np.diag([-1.0, -1.0, 1.0])

    def _write_result(
        self,
        *,
        right_shift: float = 0.0,
        generated_coordinates: dict[
            str,
            tuple[float, float, float]
            | dict[int, tuple[float, float, float]],
        ]
        | None = None,
    ) -> None:
        lines = []
        serial = 1
        for copy_index, chain in enumerate(("A", "B")):
            left = self._transform(self.left, copy_index)
            right = self._transform(self.right, copy_index)
            right = right + np.asarray([right_shift, 0.0, 0.0])
            for residue, coordinate in enumerate(left, start=1):
                lines.append(
                    _atom_line(serial, chain, residue, tuple(coordinate))
                )
                serial += 1
            for residue, coordinate in enumerate(right, start=4):
                lines.append(
                    _atom_line(serial, chain, residue, tuple(coordinate))
                )
                serial += 1
            if generated_coordinates and chain in generated_coordinates:
                generated = generated_coordinates[chain]
                residue_coordinates = (
                    generated.items()
                    if isinstance(generated, dict)
                    else ((10, generated),)
                )
                for residue, coordinate in residue_coordinates:
                    lines.append(
                        _atom_line(
                            serial,
                            chain,
                            residue,
                            coordinate,
                        )
                    )
                    serial += 1
        self.result_structure.write_text(
            "".join(lines) + "END\n", encoding="utf-8"
        )

    def _compiled_input(
        self,
        geometry: dict,
        *,
        satisfaction_stage: str = "input",
        target_copy_offset: int = 0,
    ) -> Path:
        identity = np.eye(4).tolist()
        half_turn = np.diag([-1.0, -1.0, 1.0, 1.0]).tolist()
        plan = [
            {
                "edge_instance_id": f"edge@orbit[{copy_index}]",
                "source_interface_id": "edge",
                "required": True,
                "satisfaction_stage": satisfaction_stage,
                "source_copy_index": copy_index,
                "target_copy_index": (
                    copy_index + target_copy_offset
                ) % 2,
                "left_source_components": ["A1-3"],
                "right_source_components": ["B1-3"],
                "target_geometry": geometry,
            }
            for copy_index in range(2)
        ]
        path = self.root / "rfd3_input.json"
        path.write_text(
            json.dumps(
                {
                    "example": {
                        "input": str(self.source),
                        "extra": {
                            "symmetry_multiplicity": 2,
                            "registry_transform_order": ["C2:e", "C2:r1"],
                            "registry_transform_matrices": {
                                "C2:e": identity,
                                "C2:r1": half_turn,
                            },
                            "assembly_interface_relations": plan,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_preserve_input_relation_passes_for_one_joint_pose(self) -> None:
        self._write_result()
        report = audit_interface_relations(
            compiled_input=self._compiled_input(
                {
                    "mode": "reference_transform",
                    "from_reference_seed": True,
                    "translation_tolerance": 0.1,
                    "rotation_tolerance_deg": 1.0,
                }
            ),
            result_json=self.result_json,
            result_structure=self.result_structure,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["summary"]["edge_instance_count"], 2)
        self.assertTrue(
            all(edge["satisfied"] for edge in report["interfaces"])
        )
        self.assertTrue(
            all(
                edge["translation_error"] < 1e-6
                for edge in report["interfaces"]
            )
        )

    def test_atomic_hyperedge_groups_member_edges_by_orbit_action(self) -> None:
        """Pairwise runtime members remain one physical interface object."""

        self._write_result()
        compiled = self._compiled_input(
            {
                "mode": "reference_transform",
                "from_reference_seed": True,
                "translation_tolerance": 0.1,
                "rotation_tolerance_deg": 1.0,
            }
        )
        payload = json.loads(compiled.read_text(encoding="utf-8"))
        members = []
        for edge in payload["example"]["extra"][
            "assembly_interface_relations"
        ]:
            for member_index in (1, 2):
                member = dict(edge)
                member["edge_instance_id"] = (
                    f"three_way__member_{member_index:02d}"
                    f"@orbit[{edge['source_copy_index']}]"
                )
                member["source_interface_id"] = (
                    f"three_way__member_{member_index:02d}"
                )
                member["hyperedge_id"] = "three_way"
                member["orbit_id"] = "motif_orbit"
                member["action_copy_index"] = edge["source_copy_index"]
                members.append(member)
        payload["example"]["extra"][
            "assembly_interface_relations"
        ] = members
        compiled.write_text(json.dumps(payload), encoding="utf-8")

        report = audit_interface_relations(
            compiled_input=compiled,
            result_json=self.result_json,
            result_structure=self.result_structure,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["summary"]["interface_count"], 2)
        self.assertEqual(report["summary"]["interface_hyperedge_count"], 1)
        hyperedge = report["interface_hyperedges"][0]
        self.assertEqual(hyperedge["hyperedge_id"], "three_way")
        self.assertEqual(hyperedge["physical_instance_count"], 2)
        self.assertEqual(hyperedge["member_edge_instance_count"], 4)
        self.assertEqual(hyperedge["members_per_physical_instance"], [2])
        self.assertTrue(hyperedge["satisfied"])
        self.assertEqual(
            {
                (instance["orbit_id"], instance["action_copy_index"])
                for instance in hyperedge["physical_instances"]
            },
            {("motif_orbit", 0), ("motif_orbit", 1)},
        )

    def test_preserve_input_relation_detects_component_drift(self) -> None:
        self._write_result(right_shift=1.0)
        report = audit_interface_relations(
            compiled_input=self._compiled_input(
                {
                    "mode": "reference_transform",
                    "from_reference_seed": True,
                    "translation_tolerance": 0.1,
                    "rotation_tolerance_deg": 1.0,
                }
            ),
            result_json=self.result_json,
            result_structure=self.result_structure,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            len(report["summary"]["failed_required_edge_instances"]), 2
        )
        self.assertTrue(
            all(
                edge["translation_error"] > 0.9
                for edge in report["interfaces"]
            )
        )

    def test_preserve_input_relation_can_require_contacts(self) -> None:
        self._write_result()
        report = audit_interface_relations(
            compiled_input=self._compiled_input(
                {
                    "mode": "reference_transform",
                    "from_reference_seed": True,
                    "translation_tolerance": 0.1,
                    "rotation_tolerance_deg": 1.0,
                    "minimum_heavy_atom_contacts": 100,
                    "contact_cutoff": 4.1,
                }
            ),
            result_json=self.result_json,
            result_structure=self.result_structure,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            all(
                not edge["contacts_satisfied"]
                for edge in report["interfaces"]
            )
        )

    def test_contact_relation_checks_distance_and_contacts(self) -> None:
        self._write_result()
        report = audit_interface_relations(
            compiled_input=self._compiled_input(
                {
                    "mode": "geometric_constraints",
                    "distance": {
                        "type": "com",
                        "target": 4.0,
                        "tolerance": 0.1,
                    },
                    "contacts": {
                        "min_heavy_atom_contacts": 1,
                        "cutoff": 4.1,
                    },
                }
            ),
            result_json=self.result_json,
            result_structure=self.result_structure,
        )

        self.assertTrue(report["passed"])
        self.assertTrue(
            all(edge["contacts_satisfied"] for edge in report["interfaces"])
        )
        self.assertTrue(
            all(edge["distance_satisfied"] for edge in report["interfaces"])
        )

    def test_output_contact_relation_audits_generated_chain_atoms(self) -> None:
        # The declared ports are fixed motif atoms.  A design-interface edge
        # must instead judge the generated scaffold on the concrete chains
        # joined by the symmetry-expanded relation.
        self._write_result(
            generated_coordinates={
                "A": (10.0, 0.0, 0.0),
                "B": (14.0, 0.0, 0.0),
            }
        )
        report = audit_interface_relations(
            compiled_input=self._compiled_input(
                {
                    "mode": "geometric_constraints",
                    "contacts": {
                        "min_heavy_atom_contacts": 1,
                        "cutoff": 4.1,
                    },
                },
                satisfaction_stage="output",
                target_copy_offset=1,
            ),
            result_json=self.result_json,
            result_structure=self.result_structure,
        )

        self.assertTrue(report["passed"])
        self.assertTrue(
            all(
                edge["evaluation_scope"] == "generated_chain_atoms"
                for edge in report["interfaces"]
            )
        )
        self.assertTrue(
            all(edge["evaluated_left_atoms"] == 1 for edge in report["interfaces"])
        )
        self.assertTrue(
            all(edge["contacts_satisfied"] for edge in report["interfaces"])
        )

    def test_output_contact_auto_derives_residue_quality_targets(self) -> None:
        self._write_result(
            generated_coordinates={
                "A": (10.0, 0.0, 0.0),
                "B": (14.0, 0.0, 0.0),
            }
        )
        report = audit_interface_relations(
            compiled_input=self._compiled_input(
                {
                    "mode": "geometric_constraints",
                    "contacts": {
                        "min_heavy_atom_contacts": 0,
                        "cutoff": 8.0,
                    },
                    "coverage": {"mode": "auto"},
                },
                satisfaction_stage="output",
                target_copy_offset=1,
            ),
            result_json=self.result_json,
            result_structure=self.result_structure,
        )

        self.assertTrue(report["passed"])
        self.assertTrue(
            report["summary"]["output_packing_quality_satisfied"]
        )
        self.assertGreaterEqual(
            report["summary"][
                "minimum_reciprocal_contact_residue_pairs"
            ],
            1,
        )
        for edge in report["interfaces"]:
            self.assertEqual(edge["minimum_contact_residues_per_side"], 1)
            self.assertEqual(
                edge["minimum_contiguous_contact_residues_per_side"],
                1,
            )
            self.assertTrue(edge["contact_residue_coverage_satisfied"])
            self.assertTrue(edge["contact_continuity_satisfied"])
            self.assertGreaterEqual(
                edge["reciprocal_contact_residue_pair_count"],
                1,
            )
            self.assertGreaterEqual(edge["reciprocal_contact_density"], 1.0)
            self.assertGreater(edge["heavy_atom_burial_proxy"], 0.0)
            self.assertTrue(edge["output_packing_quality_satisfied"])

    def test_output_contact_auto_continuity_respects_generated_runs(
        self,
    ) -> None:
        residue_numbers = tuple(
            residue
            for start in range(10, 90, 10)
            for residue in (start, start + 1)
        )
        self._write_result(
            generated_coordinates={
                "A": {
                    residue: (10.0, float(index * 10), 0.0)
                    for index, residue in enumerate(residue_numbers)
                },
                "B": {
                    residue: (14.0, float(index * 10), 0.0)
                    for index, residue in enumerate(residue_numbers)
                },
            }
        )
        report = audit_interface_relations(
            compiled_input=self._compiled_input(
                {
                    "mode": "geometric_constraints",
                    "contacts": {
                        "min_heavy_atom_contacts": 0,
                        "cutoff": 8.0,
                    },
                    "coverage": {"mode": "auto"},
                },
                satisfaction_stage="output",
                target_copy_offset=1,
            ),
            result_json=self.result_json,
            result_structure=self.result_structure,
        )

        self.assertTrue(report["passed"])
        for edge in report["interfaces"]:
            self.assertEqual(edge["minimum_contact_residues_per_side"], 4)
            self.assertEqual(
                edge["minimum_contiguous_contact_residues_per_side"],
                2,
            )
            self.assertEqual(edge["available_contiguous_residues_left"], 2)
            self.assertEqual(edge["available_contiguous_residues_right"], 2)
            self.assertTrue(edge["contact_continuity_satisfied"])
            self.assertEqual(edge["contact_island_count_left"], 8)
            self.assertEqual(edge["contact_island_count_right"], 8)
            self.assertTrue(edge["output_packing_quality_satisfied"])

    def test_output_contact_explicit_continuity_is_not_relaxed(self) -> None:
        residue_numbers = tuple(
            residue
            for start in range(10, 90, 10)
            for residue in (start, start + 1)
        )
        self._write_result(
            generated_coordinates={
                "A": {
                    residue: (10.0, float(index * 10), 0.0)
                    for index, residue in enumerate(residue_numbers)
                },
                "B": {
                    residue: (14.0, float(index * 10), 0.0)
                    for index, residue in enumerate(residue_numbers)
                },
            }
        )
        report = audit_interface_relations(
            compiled_input=self._compiled_input(
                {
                    "mode": "geometric_constraints",
                    "contacts": {
                        "min_heavy_atom_contacts": 0,
                        "cutoff": 8.0,
                    },
                    "coverage": {
                        "mode": "explicit",
                        "minimum_contact_residues_per_side": 4,
                        "minimum_contiguous_contact_residues_per_side": 3,
                    },
                },
                satisfaction_stage="output",
                target_copy_offset=1,
            ),
            result_json=self.result_json,
            result_structure=self.result_structure,
        )

        self.assertFalse(report["passed"])
        for edge in report["interfaces"]:
            self.assertEqual(
                edge["minimum_contiguous_contact_residues_per_side"],
                3,
            )
            self.assertEqual(edge["available_contiguous_residues_left"], 2)
            self.assertEqual(edge["available_contiguous_residues_right"], 2)
            self.assertFalse(edge["contact_continuity_satisfied"])

    def test_output_contact_relation_fails_without_generated_contacts(self) -> None:
        self._write_result()
        report = audit_interface_relations(
            compiled_input=self._compiled_input(
                {
                    "mode": "geometric_constraints",
                    "contacts": {
                        "min_heavy_atom_contacts": 1,
                        "cutoff": 4.1,
                    },
                },
                satisfaction_stage="output",
                target_copy_offset=1,
            ),
            result_json=self.result_json,
            result_structure=self.result_structure,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            all(edge["evaluated_left_atoms"] == 0 for edge in report["interfaces"])
        )

    def test_cross_seam_port_uses_declared_runtime_provenance(self) -> None:
        """An original B port may compile as equivalent F@next-action atoms."""

        theta = 2.0 * np.pi / 3.0
        rotation_one = np.asarray(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotation_two = rotation_one @ rotation_one
        rotations = (np.eye(3), rotation_one, rotation_two)
        left = np.asarray(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        )
        original_right = left + np.asarray([4.0, 0.0, 0.0])
        runtime_right = original_right @ rotation_one
        self.source.write_text(
            "".join(
                _atom_line(serial, chain, residue, tuple(coordinate))
                for serial, (chain, residue, coordinate) in enumerate(
                    [
                        *(('A', index, coordinate) for index, coordinate in enumerate(left, 1)),
                        *((
                            'B', index, coordinate
                        ) for index, coordinate in enumerate(original_right, 1)),
                        *(('F', index, coordinate) for index, coordinate in enumerate(runtime_right, 1)),
                    ],
                    start=1,
                )
            )
            + "END\n",
            encoding="utf-8",
        )
        self.result_json.write_text(
            json.dumps(
                {
                    "diffused_index_map": {
                        **{f"A{i}": f"A{i}" for i in range(1, 4)},
                        **{f"F{i}": f"B{i + 3}" for i in range(1, 4)},
                    }
                }
            ),
            encoding="utf-8",
        )
        lines: list[str] = []
        serial = 1
        output_chains = ("A", "B", "C", "D", "E", "F")
        for action_index, rotation in enumerate(rotations):
            left_chain = output_chains[action_index * 2]
            right_chain = output_chains[action_index * 2 + 1]
            for residue, coordinate in enumerate(left @ rotation.T, 1):
                lines.append(
                    _atom_line(serial, left_chain, residue, tuple(coordinate))
                )
                serial += 1
            for residue, coordinate in enumerate(
                runtime_right @ rotation.T, 4
            ):
                lines.append(
                    _atom_line(serial, right_chain, residue, tuple(coordinate))
                )
                serial += 1
        self.result_structure.write_text(
            "".join(lines) + "END\n", encoding="utf-8"
        )

        matrices = {}
        order = []
        for index, rotation in enumerate(rotations):
            transform_id = "C3:e" if index == 0 else f"C3:r{index}"
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrices[transform_id] = matrix.tolist()
            order.append(transform_id)
        groups = [
            {
                "group_id": f"fixed@cross_seam[{index}]",
                "constraint_orbit_id": "cross_seam",
                "members": [
                    {
                        "src_components": ["A1", "A2", "A3"],
                        "sym_transform_id": index,
                    },
                    {
                        "src_components": ["F1", "F2", "F3"],
                        "sym_transform_id": (index + 1) % 3,
                    },
                ],
            }
            for index in range(3)
        ]
        relations = [
            {
                "edge_instance_id": f"edge@orbit[{index}]",
                "source_interface_id": "edge",
                "required": True,
                "satisfaction_stage": "input",
                "source_copy_index": index,
                "target_copy_index": index,
                "left_source_components": ["A1-3"],
                "right_source_components": ["B1-3"],
                "target_geometry": {
                    "mode": "reference_transform",
                    "from_reference_seed": True,
                    "translation_tolerance": 0.01,
                    "rotation_tolerance_deg": 0.1,
                },
            }
            for index in range(3)
        ]
        compiled_input = self.root / "cross_seam_rfd3_input.json"
        compiled_input.write_text(
            json.dumps(
                {
                    "example": {
                        "input": str(self.source),
                        "extra": {
                            "symmetry_multiplicity": 3,
                            "registry_transform_order": order,
                            "registry_transform_matrices": matrices,
                            "motif_constraint_groups": groups,
                            "assembly_interface_relations": relations,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        report = audit_interface_relations(
            compiled_input=compiled_input,
            result_json=self.result_json,
            result_structure=self.result_structure,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["summary"]["asu_chain_count"], 2)
        self.assertEqual(
            report["summary"]["satisfied_required_edge_instance_count"], 3
        )
        for edge in report["interfaces"]:
            self.assertTrue(edge["runtime_provenance_remapped"])
            self.assertEqual(edge["left_runtime_remapped_heavy_atoms"], 0)
            self.assertEqual(edge["right_runtime_remapped_heavy_atoms"], 3)
            self.assertEqual(edge["matched_heavy_atoms"], 6)
            self.assertLess(edge["translation_error"], 0.002)


if __name__ == "__main__":
    unittest.main()
