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
        generated_coordinates: dict[str, tuple[float, float, float]]
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
                lines.append(
                    _atom_line(
                        serial,
                        chain,
                        10,
                        generated_coordinates[chain],
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
        for edge in report["interfaces"]:
            self.assertEqual(edge["minimum_contact_residues_per_side"], 1)
            self.assertEqual(
                edge["minimum_contiguous_contact_residues_per_side"],
                1,
            )
            self.assertTrue(edge["contact_residue_coverage_satisfied"])
            self.assertTrue(edge["contact_continuity_satisfied"])

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


if __name__ == "__main__":
    unittest.main()
