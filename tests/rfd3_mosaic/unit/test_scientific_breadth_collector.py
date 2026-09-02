import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
COLLECTOR = (
    REPOSITORY_ROOT
    / "scripts"
    / "rfd3_mosaic"
    / "collect_scientific_breadth_campaign.py"
)


class ScientificBreadthCollectorTestCase(unittest.TestCase):
    def test_collects_indexed_run_outputs_and_writes_transfer_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "runs"
            run = run_root / "2026-09-02" / "campaign" / "case" / "run"
            mirror = run / "generated_structures_cif"
            audits = run / "audits" / "design-000"
            mirror.mkdir(parents=True)
            audits.mkdir(parents=True)
            (mirror / "design-000.cif.gz").write_bytes(b"coordinates")
            (audits / "scaffold_validity_audit.json").write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )
            (audits / "screening_advice.json").write_text(
                json.dumps(
                    {
                        "contract_status": "met",
                        "recommendation": "recommended_for_next_stage",
                    }
                ),
                encoding="utf-8",
            )
            index_root = run_root / ".rfd3-mosaic" / "jobs"
            index_root.mkdir(parents=True)
            (index_root / "123.json").write_text(
                json.dumps(
                    {
                        "state": "completed",
                        "run_directory": str(run),
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "campaign_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "campaign": "breadth-test",
                        "source_revision": "abc123",
                        "run_root": str(run_root),
                        "records": [
                            {
                                "case": "case-a",
                                "job_id": "123",
                                "requested_designs": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [sys.executable, str(COLLECTOR), str(manifest)],
                cwd=REPOSITORY_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            summary = json.loads(
                (root / "collection_summary.json").read_text(encoding="utf-8")
            )
            transfer = (root / "transfer_paths.txt").read_text(encoding="utf-8")

        self.assertEqual(summary["observed_output_count"], 1)
        self.assertEqual(summary["audit_directory_count"], 1)
        self.assertEqual(summary["audit_passed_output_count"], 1)
        self.assertEqual(summary["contract_met_output_count"], 1)
        self.assertEqual(summary["recommended_output_count"], 1)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["records"][0]["state"], "completed")
        self.assertIn(str(run), transfer)
        self.assertIn(str(manifest.parent), transfer)


if __name__ == "__main__":
    unittest.main()
