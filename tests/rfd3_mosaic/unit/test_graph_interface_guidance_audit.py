import json
import tempfile
import unittest
from pathlib import Path

from rfd3_mosaic.rfd3_graph_interface_guidance_audit import (
    audit_graph_interface_guidance,
)


class GraphInterfaceGuidanceAuditTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.compiled = self.root / "input.json"
        self.result = self.root / "result.json"
        self.compiled.write_text(
            json.dumps(
                {
                    "example": {
                        "extra": {
                            "assembly_interface_relations": [
                                {
                                    "edge_instance_id": "edge@0",
                                    "source_interface_id": "edge",
                                    "required": True,
                                    "satisfaction_stage": "output",
                                    "target_geometry": {
                                        "mode": "geometric_constraints"
                                    },
                                },
                                {
                                    "edge_instance_id": "preserved@0",
                                    "required": True,
                                    "satisfaction_stage": "input",
                                    "target_geometry": {
                                        "mode": "reference_transform"
                                    },
                                },
                            ]
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_result(self, *, edge_id: str = "edge@0") -> None:
        self.result.write_text(
            json.dumps(
                {
                    "graph_interface_guidance_diagnostics": {
                        "schema_version": 3,
                        "runtime_active": True,
                        "edge_count": 1,
                        "edge_ids": [edge_id],
                        "source_interface_ids": ["edge"],
                        "applied_steps": 1,
                        "steps": [
                            {
                                "applied": True,
                                "window_weight": 1.0,
                                "energy": 2.0,
                                "attraction": 2.0,
                                "coverage": 0.5,
                                "continuity": 0.25,
                                "clash": 0.0,
                                "distance": 0.0,
                                "maximum_token_step": 0.25,
                                "mean_token_step": 0.2,
                                "minimum_distances": [7.5],
                                "covered_left_residues": [3],
                                "covered_right_residues": [3],
                                "target_residues_per_side": [3],
                                "target_contiguous_residues_per_side": [2],
                                "contiguous_left_residues": [3],
                                "contiguous_right_residues": [3],
                                "per_edge_total": [2.75],
                            }
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )

    def test_accepts_complete_runtime_evidence(self) -> None:
        self._write_result()
        report = audit_graph_interface_guidance(
            compiled_input=self.compiled,
            result_json=self.result,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["summary"]["applied_steps"], 1)
        self.assertEqual(
            report["summary"]["diagnostics_schema_version"],
            3,
        )

    def test_accepts_legacy_v1_runtime_evidence(self) -> None:
        self._write_result()
        payload = json.loads(self.result.read_text(encoding="utf-8"))
        diagnostics = payload["graph_interface_guidance_diagnostics"]
        diagnostics.pop("schema_version")
        diagnostics.pop("source_interface_ids")
        for step in diagnostics["steps"]:
            for key in (
                "coverage",
                "continuity",
                "mean_token_step",
                "covered_left_residues",
                "covered_right_residues",
                "target_residues_per_side",
                "target_contiguous_residues_per_side",
                "contiguous_left_residues",
                "contiguous_right_residues",
                "per_edge_total",
            ):
                step.pop(key)
        self.result.write_text(json.dumps(payload), encoding="utf-8")

        report = audit_graph_interface_guidance(
            compiled_input=self.compiled,
            result_json=self.result,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(
            report["summary"]["diagnostics_schema_version"],
            1,
        )

    def test_rejects_wrong_edge_identity(self) -> None:
        self._write_result(edge_id="wrong@0")
        report = audit_graph_interface_guidance(
            compiled_input=self.compiled,
            result_json=self.result,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["summary"]["identifier_contract_valid"])

    def test_rejects_missing_runtime_diagnostics(self) -> None:
        self.result.write_text("{}", encoding="utf-8")
        report = audit_graph_interface_guidance(
            compiled_input=self.compiled,
            result_json=self.result,
        )

        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
