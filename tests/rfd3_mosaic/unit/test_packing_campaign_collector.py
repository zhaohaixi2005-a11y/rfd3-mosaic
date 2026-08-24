import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PackingCampaignCollectorTestCase(unittest.TestCase):
    def test_reports_contract_and_advisory_metrics_without_acceptance_verdict(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "runs"
            run = run_root / "campaign" / "example" / "1"
            audit = run / "audits" / "design_000"
            audit.mkdir(parents=True)
            index = run_root / ".rfd3-mosaic" / "jobs" / "1.json"
            index.parent.mkdir(parents=True)
            index.write_text(
                json.dumps({"run_directory": str(run)}),
                encoding="utf-8",
            )
            (audit / "graph_interface_guidance_audit.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "summary": {
                            "quality_targets_satisfied": False,
                            "final_packing_metrics": {
                                "shape": 0.25,
                                "minimum_edge_distance": 3.6,
                                "covered_left_residues": [8, 6],
                                "covered_right_residues": [7, 5],
                                "contiguous_left_residues": [5, 4],
                                "contiguous_right_residues": [4, 3],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (audit / "assembly_interface_relation_audit.json").write_text(
                json.dumps(
                    {
                        "passed": False,
                        "summary": {
                            "satisfied_required_edge_instance_count": 1,
                            "required_edge_instance_count": 2,
                        },
                        "interfaces": [
                            {
                                "contact_residue_count_left": 9,
                                "contact_residue_count_right": 8,
                                "maximum_contiguous_contact_residues_left": 4,
                                "maximum_contiguous_contact_residues_right": 3,
                                "hard_clashes_below_2_0A": 0,
                            },
                            {
                                "contact_residue_count_left": 6,
                                "contact_residue_count_right": 5,
                                "maximum_contiguous_contact_residues_left": 3,
                                "maximum_contiguous_contact_residues_right": 2,
                                "hard_clashes_below_2_0A": 1,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (audit / "scaffold_validity_audit.json").write_text(
                json.dumps({"passed": True}),
                encoding="utf-8",
            )
            (audit / "screening_advice.json").write_text(
                json.dumps(
                    {
                        "contract_status": "met",
                        "recommendation": "review_advisory_metrics",
                        "contract_flags": [],
                        "advisory_flags": [
                            {"code": "advisory.interface.packing"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (audit / "component_mobility_audit.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "summary": {
                            "applied_proposal_count": 3,
                            "components": [
                                {
                                    "maximum_translation_observed": 1.25,
                                    "maximum_rotation_deg_observed": 4.5,
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "campaign_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "run_root": str(run_root),
                        "git_revision": "test-revision",
                        "requested_output_count": 1,
                        "records": [
                            {"mode": "guided", "seed": 1, "job_id": "1"}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            script = (
                Path(__file__).resolve().parents[3]
                / "scripts"
                / "rfd3_mosaic"
                / "collect_packing_campaign.py"
            )
            subprocess.run(
                [sys.executable, str(script), str(manifest)],
                check=True,
                capture_output=True,
                text=True,
            )

            summary = json.loads(
                (root / "packing_campaign_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            result = summary["records"][0]["results"][0]
            self.assertEqual(summary["schema_version"], 3)
            self.assertEqual(summary["generated_output_count"], 1)
            self.assertEqual(summary["runtime_contract_met_output_count"], 1)
            self.assertEqual(
                summary[
                    "interface_guidance_runtime_contract_met_output_count"
                ],
                1,
            )
            self.assertEqual(summary["overall_contract_met_output_count"], 1)
            self.assertEqual(
                summary["packing_targets_satisfied_output_count"], 0
            )
            self.assertNotIn("accepted_output_count", summary)
            self.assertEqual(result["runtime_ca_contact_residues_per_side"], 5)
            self.assertEqual(
                result["runtime_ca_contiguous_residues_per_side"], 3
            )
            self.assertEqual(result["posthoc_contact_residues_per_side"], 5)
            self.assertEqual(
                result["posthoc_contiguous_residues_per_side"], 2
            )
            self.assertEqual(result["maximum_translation_observed"], 1.25)
            self.assertEqual(result["maximum_rotation_deg_observed"], 4.5)
            self.assertEqual(result["mobility_applied_proposal_count"], 3)
            markdown = (
                root / "packing_campaign_summary.md"
            ).read_text(encoding="utf-8")
            self.assertNotIn("scientifically accepted", markdown)
            self.assertIn("advisory packing targets satisfied", markdown)
            self.assertIn("runtime CA contiguous", markdown)
            self.assertIn("interface-guidance runtime contracts met", markdown)
            self.assertIn("move (A/deg; commits)", markdown)


if __name__ == "__main__":
    unittest.main()
