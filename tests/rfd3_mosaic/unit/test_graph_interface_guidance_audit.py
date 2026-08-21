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
                        "schema_version": 8,
                        "runtime_active": True,
                        "edge_count": 1,
                        "edge_ids": [edge_id],
                        "source_interface_ids": ["edge"],
                        "capacity_preflight": [
                            {
                                "edge_id": "edge@0",
                                "source_interface_id": "edge",
                                "requested_residues_per_side": 3,
                                "requested_contiguous_residues_per_side": 2,
                                "available_residues_left": 8,
                                "available_residues_right": 8,
                                "available_contiguous_residues_left": 8,
                                "available_contiguous_residues_right": 8,
                            }
                        ],
                        "applied_steps": 1,
                        "final_polish_steps": 1,
                        "final_proxy_targets_satisfied": True,
                        "final_proxy": {
                            "energy": 1.5,
                            "attraction": 0.1,
                            "coverage": 0.2,
                            "continuity": 0.3,
                            "orientation": 0.4,
                            "shape": 0.5,
                            "backbone": 0.6,
                            "interface_balance": 0.7,
                            "clash": 0.0,
                            "distance": 0.0,
                            "minimum_distances": [7.25],
                            "mean_selected_distances": [8.0],
                            "covered_left_residues": [3],
                            "covered_right_residues": [3],
                            "target_residues_per_side": [3],
                            "contiguous_left_residues": [3],
                            "contiguous_right_residues": [3],
                            "target_contiguous_residues_per_side": [2],
                            "per_edge_orientation": [0.4],
                            "per_edge_shape": [0.5],
                            "per_edge_backbone": [0.6],
                            "per_edge_total": [1.5],
                            "per_source_total": [1.5],
                        },
                        "steps": [
                            {
                                "applied": True,
                                "patch_locked": True,
                                "patch_assignments": {
                                    "edge@0": {
                                        "left_token_ids": [1, 2, 3],
                                        "right_token_ids": [7, 8, 9],
                                    }
                                },
                                "window_weight": 1.0,
                                "adaptive_phase": "expand",
                                "time_scheduled_target_ca_distance": 9.0,
                                "scheduled_target_ca_distance": 9.0,
                                "energy": 2.0,
                                "attraction": 2.0,
                                "coverage": 0.5,
                                "continuity": 0.25,
                                "orientation": 0.1,
                                "shape": 0.2,
                                "backbone": 0.0,
                                "interface_balance": 0.0,
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
                                "per_edge_orientation": [0.1],
                                "per_edge_shape": [0.2],
                                "per_edge_backbone": [0.0],
                                "per_edge_total": [2.75],
                                "per_source_total": [2.75],
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
            8,
        )
        self.assertEqual(
            report["summary"]["final_packing_metrics"]["orientation"],
            0.4,
        )
        self.assertEqual(
            report["summary"]["final_packing_metrics"][
                "minimum_edge_distance"
            ],
            7.25,
        )
        self.assertEqual(
            report["summary"]["final_metrics_source"],
            "post_finalize_state",
        )
        self.assertTrue(
            report["summary"]["patch_identity_contract_valid"]
        )
        self.assertTrue(
            report["summary"]["adaptive_phase_contract_valid"]
        )
        self.assertTrue(
            report["summary"]["capacity_preflight_contract_valid"]
        )
        self.assertEqual(
            report["summary"]["adaptive_phase_counts"]["expand"],
            1,
        )

    def test_v8_rejects_missing_adaptive_phase_evidence(self) -> None:
        self._write_result()
        payload = json.loads(self.result.read_text(encoding="utf-8"))
        payload["graph_interface_guidance_diagnostics"]["steps"][0].pop(
            "adaptive_phase"
        )
        self.result.write_text(json.dumps(payload), encoding="utf-8")

        report = audit_graph_interface_guidance(
            compiled_input=self.compiled,
            result_json=self.result,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(
            report["summary"]["adaptive_phase_contract_valid"]
        )

    def test_v8_rejects_missing_capacity_preflight(self) -> None:
        self._write_result()
        payload = json.loads(self.result.read_text(encoding="utf-8"))
        payload["graph_interface_guidance_diagnostics"].pop(
            "capacity_preflight"
        )
        self.result.write_text(json.dumps(payload), encoding="utf-8")

        report = audit_graph_interface_guidance(
            compiled_input=self.compiled,
            result_json=self.result,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(
            report["summary"]["capacity_preflight_contract_valid"]
        )

    def test_v9_requires_contact_prior_runtime_evidence(self) -> None:
        self._write_result()
        payload = json.loads(self.result.read_text(encoding="utf-8"))
        diagnostics = payload["graph_interface_guidance_diagnostics"]
        diagnostics["schema_version"] = 9
        diagnostics["config"] = {
            "contact_prior_weight": 0.1,
            "contact_prior_guide_scale": 2.0,
            "contact_prior_decay_power": 2.0,
            "contact_prior_r_0": 8.0,
            "contact_prior_d_0": 2.0,
        }
        diagnostics["steps"][0].update(
            {
                "contact_prior": -1.0,
                "contact_prior_schedule_scale": 1.5,
                "effective_contact_prior_weight": 0.15,
                "per_edge_contact_prior": [-1.0],
            }
        )
        diagnostics["final_proxy"].update(
            {
                "contact_prior": -0.8,
                "per_edge_contact_prior": [-0.8],
            }
        )
        self.result.write_text(json.dumps(payload), encoding="utf-8")

        report = audit_graph_interface_guidance(
            compiled_input=self.compiled,
            result_json=self.result,
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["summary"]["contact_prior_contract_valid"])
        self.assertEqual(
            report["summary"]["final_packing_metrics"]["contact_prior"],
            -0.8,
        )

        diagnostics["steps"][0].pop("per_edge_contact_prior")
        self.result.write_text(json.dumps(payload), encoding="utf-8")
        rejected = audit_graph_interface_guidance(
            compiled_input=self.compiled,
            result_json=self.result,
        )
        self.assertFalse(rejected["passed"])

    def test_v7_rejects_patch_hopping_after_lock(self) -> None:
        self._write_result()
        payload = json.loads(self.result.read_text(encoding="utf-8"))
        diagnostics = payload["graph_interface_guidance_diagnostics"]
        second = dict(diagnostics["steps"][0])
        second["patch_assignments"] = {
            "edge@0": {
                "left_token_ids": [2, 3, 4],
                "right_token_ids": [8, 9, 10],
            }
        }
        diagnostics["steps"].append(second)
        diagnostics["applied_steps"] = 2
        self.result.write_text(json.dumps(payload), encoding="utf-8")

        report = audit_graph_interface_guidance(
            compiled_input=self.compiled,
            result_json=self.result,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(
            report["summary"]["patch_identity_contract_valid"]
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

    def test_v5_rejects_missing_orientation_evidence(self) -> None:
        self._write_result()
        payload = json.loads(self.result.read_text(encoding="utf-8"))
        payload["graph_interface_guidance_diagnostics"]["steps"][0].pop(
            "per_edge_orientation"
        )
        self.result.write_text(json.dumps(payload), encoding="utf-8")

        report = audit_graph_interface_guidance(
            compiled_input=self.compiled,
            result_json=self.result,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["summary"]["packing_evidence_steps"], 0)

    def test_v5_rejects_unsatisfied_final_proxy(self) -> None:
        self._write_result()
        payload = json.loads(self.result.read_text(encoding="utf-8"))
        diagnostics = payload["graph_interface_guidance_diagnostics"]
        diagnostics["final_proxy_targets_satisfied"] = False
        diagnostics["final_proxy"]["covered_left_residues"] = [1]
        self.result.write_text(json.dumps(payload), encoding="utf-8")

        report = audit_graph_interface_guidance(
            compiled_input=self.compiled,
            result_json=self.result,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(
            report["summary"]["final_proxy_targets_satisfied"]
        )
        self.assertTrue(report["summary"]["final_proxy_contract_valid"])
        self.assertFalse(
            report["summary"]["final_result_contract_valid"]
        )

    def test_v5_rejects_missing_final_proxy_evidence(self) -> None:
        self._write_result()
        payload = json.loads(self.result.read_text(encoding="utf-8"))
        diagnostics = payload["graph_interface_guidance_diagnostics"]
        diagnostics.pop("final_proxy")
        self.result.write_text(json.dumps(payload), encoding="utf-8")

        report = audit_graph_interface_guidance(
            compiled_input=self.compiled,
            result_json=self.result,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["summary"]["final_proxy_contract_valid"])

    def test_rejects_missing_runtime_diagnostics(self) -> None:
        self.result.write_text("{}", encoding="utf-8")
        report = audit_graph_interface_guidance(
            compiled_input=self.compiled,
            result_json=self.result,
        )

        self.assertFalse(report["passed"])

    def test_accepts_explicit_c2_automatic_scaffold_packing(self) -> None:
        edge_id = (
            "automatic_symmetric_scaffold_interface@chain_0_chain_1"
        )
        source_id = "automatic_symmetric_scaffold_interface"
        self.compiled.write_text(
            json.dumps(
                {
                    "example": {
                        "symmetry": {"id": "C2"},
                        "extra": {
                            "assembly_interface_relations": [],
                            "automatic_symmetric_scaffold_packing": {
                                "mode": "symmetric_generated",
                                "neighbour_policy": "cyclic_adjacent",
                            },
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        self._write_result(edge_id=edge_id)
        payload = json.loads(self.result.read_text(encoding="utf-8"))
        diagnostics = payload["graph_interface_guidance_diagnostics"]
        diagnostics["source_interface_ids"] = [source_id]
        diagnostics["capacity_preflight"][0]["edge_id"] = edge_id
        diagnostics["capacity_preflight"][0]["source_interface_id"] = (
            source_id
        )
        diagnostics["steps"][0]["patch_assignments"] = {
            edge_id: {
                "left_token_ids": [1, 2, 3],
                "right_token_ids": [7, 8, 9],
            }
        }
        self.result.write_text(json.dumps(payload), encoding="utf-8")

        report = audit_graph_interface_guidance(
            compiled_input=self.compiled,
            result_json=self.result,
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["summary"]["identifier_contract_valid"])


if __name__ == "__main__":
    unittest.main()
