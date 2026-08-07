import json
from pathlib import Path
import tempfile
import unittest

import yaml

from rfd3_mosaic.run_reporting import (
    RunReference,
    collect_run_status,
    format_status_text,
    resolve_run_reference,
    write_report,
)


class RunReportingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _completed_run(self, job_id: str = "12345") -> Path:
        run = self.root / "campaign" / "design" / job_id
        run.mkdir(parents=True)
        constraint = run / "constraint_orbit_audit.json"
        scaffold = run / "scaffold_validity_audit.json"
        self._write_json(
            constraint,
            {"passed": True, "summary": {"joint_orbit_rmsd": 1e-5}},
        )
        self._write_json(
            scaffold,
            {"passed": True, "summary": {"ca_clash_count": 0}},
        )
        self._write_json(
            run / "rfd3_prevalidation.json",
            {"status": "passed"},
        )
        structure = run / "result_model_0.cif.gz"
        structure.write_bytes(b"structure")
        self._write_json(
            run / "experiment_summary.json",
            {
                "status": "completed",
                "experiment": "design",
                "reports": [str(constraint), str(scaffold)],
                "result_json": str(run / "result_model_0.json"),
            },
        )
        return run

    def test_completed_run_requires_every_declared_audit(self) -> None:
        run = self._completed_run()

        status = collect_run_status(
            RunReference(job_id="12345", run_directory=run),
            include_scheduler=False,
        )

        self.assertEqual(status["state"], "completed")
        self.assertTrue(status["passed"])
        self.assertEqual(len(status["audits"]), 3)
        self.assertEqual(len(status["artifacts"]["structures"]), 1)
        self.assertIn("verdict:    PASSED", format_status_text(status))

        self._write_json(
            run / "scaffold_validity_audit.json",
            {"passed": False, "summary": {"ca_clash_count": 3}},
        )
        failed = collect_run_status(
            RunReference(job_id="12345", run_directory=run),
            include_scheduler=False,
        )
        self.assertFalse(failed["passed"])

    def test_failed_worker_reports_failure_without_scheduler(self) -> None:
        run = self.root / "campaign" / "design" / "999"
        run.mkdir(parents=True)
        self._write_json(
            run / "experiment_summary.json",
            {
                "status": "failed",
                "experiment": "design",
                "error_type": "ValueError",
                "error": "semantic gate failed",
            },
        )

        status = collect_run_status(
            RunReference(job_id="999", run_directory=run),
            include_scheduler=False,
        )

        self.assertEqual(status["state"], "failed")
        self.assertFalse(status["passed"])
        self.assertIn("semantic gate failed", format_status_text(status))

    def test_numeric_job_id_resolves_completed_run(self) -> None:
        run = self._completed_run("24680")

        reference = resolve_run_reference("24680", root=self.root)

        self.assertEqual(reference.job_id, "24680")
        self.assertEqual(reference.run_directory, run.resolve())

    def test_pending_submission_receipt_resolves_without_run_directory(self) -> None:
        submission = self.root / "runs" / "campaign" / "_submissions" / "x"
        submission.mkdir(parents=True)
        self._write_json(
            submission / "submission.json",
            {"job_id": "777", "sbatch_output": "Submitted batch job 777"},
        )
        (submission / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "pending-design",
                    "output": {
                        "root": str(self.root / "runs"),
                        "campaign": "campaign",
                    },
                }
            ),
            encoding="utf-8",
        )

        reference = resolve_run_reference(submission)
        status = collect_run_status(reference, include_scheduler=False)

        self.assertEqual(reference.job_id, "777")
        self.assertIsNone(reference.run_directory)
        self.assertEqual(status["state"], "submitted")
        self.assertIsNone(status["passed"])

    def test_report_writes_html_and_canonical_json(self) -> None:
        run = self._completed_run("13579")
        status = collect_run_status(
            RunReference(job_id="13579", run_directory=run),
            include_scheduler=False,
        )
        status["experiment"] = "design <unsafe>"

        output = write_report(status)

        self.assertEqual(output, run / "mosaic_report.html")
        html = output.read_text(encoding="utf-8")
        self.assertIn("design &lt;unsafe&gt;", html)
        self.assertIn("joint_orbit_rmsd", html)
        payload = json.loads(
            output.with_suffix(".json").read_text(encoding="utf-8")
        )
        self.assertTrue(payload["passed"])

    def test_copied_run_resolves_absolute_report_paths_by_name(self) -> None:
        run = self._completed_run("86420")
        summary_path = run / "experiment_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["reports"] = [
            "/unavailable/original/constraint_orbit_audit.json",
            "/unavailable/original/scaffold_validity_audit.json",
        ]
        self._write_json(summary_path, summary)

        status = collect_run_status(
            RunReference(job_id="86420", run_directory=run),
            include_scheduler=False,
        )

        self.assertTrue(status["passed"])
        self.assertTrue(
            all(Path(item["path"]).parent == run for item in status["audits"])
        )


if __name__ == "__main__":
    unittest.main()
