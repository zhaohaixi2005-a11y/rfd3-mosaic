import json
import tempfile
import unittest
from pathlib import Path

from rfd3_mosaic.run_index import (
    list_run_records,
    read_run_record,
    rebuild_run_index,
    record_submission,
    update_run_state,
)
from rfd3_mosaic.run_reporting import resolve_run_reference


class RunIndexTestCase(unittest.TestCase):
    def test_local_executor_identity_is_indexed(self) -> None:
        run = self.root / "local-run"
        update_run_state(
            root=self.root,
            job_id="local-20260818T120000Z-42",
            state="completed",
            experiment="local-test",
            campaign="local",
            run_directory=run,
        )
        path = record_submission(
            root=self.root,
            job_id="local-20260818T120000Z-42",
            experiment="local-test",
            campaign="local",
            run_directory=run,
            submission_directory=self.root / "submission",
            executor="local",
        )

        self.assertTrue(path.is_file())
        record = read_run_record(self.root, "local-20260818T120000Z-42")
        self.assertEqual(record["executor"], "local")
        self.assertEqual(record["state"], "completed")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.submission = self.root / "campaign" / "_submissions" / "stamp"
        self.submission.mkdir(parents=True)
        self.run = self.root / "campaign" / "design" / "5734001"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_submission_and_worker_states_share_one_durable_record(self) -> None:
        path = record_submission(
            root=self.root,
            job_id="5734001",
            experiment="design",
            campaign="campaign",
            run_directory=self.run,
            submission_directory=self.submission,
            executor="slurm",
        )

        submitted = read_run_record(self.root, "5734001")
        self.assertEqual(submitted["state"], "submitted")
        self.assertEqual(path.parent.name, "jobs")

        self.run.mkdir(parents=True)
        update_run_state(
            root=self.root,
            job_id="5734001",
            state="running",
            experiment="design",
            campaign="campaign",
            run_directory=self.run,
        )
        update_run_state(
            root=self.root,
            job_id="5734001",
            state="completed",
            experiment="design",
            campaign="campaign",
            run_directory=self.run,
        )

        completed = read_run_record(self.root, "5734001")
        self.assertEqual(completed["state"], "completed")
        self.assertIsNone(completed["error"])
        self.assertEqual(list_run_records(self.root), [completed])

    def test_numeric_status_resolution_prefers_constant_time_index(self) -> None:
        record_submission(
            root=self.root,
            job_id="5734001",
            experiment="design",
            campaign="campaign",
            run_directory=self.run,
            submission_directory=self.submission,
            executor="slurm",
        )
        self.run.mkdir(parents=True)

        reference = resolve_run_reference("5734001", root=self.root)

        self.assertEqual(reference.run_directory, self.run.resolve())
        self.assertEqual(
            reference.submission_directory,
            self.submission.resolve(),
        )

    def test_worker_can_create_index_for_legacy_submission(self) -> None:
        self.run.mkdir(parents=True)

        path = update_run_state(
            root=self.root,
            job_id="5734001",
            state="failed",
            experiment="legacy",
            campaign="campaign",
            run_directory=self.run,
            error="runtime failed",
        )

        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["state"], "failed")
        self.assertEqual(payload["error"], "runtime failed")
        self.assertEqual(payload["executor"], "unknown")

    def test_rebuild_imports_historical_worker_summaries(self) -> None:
        historical = self.root / "old-campaign" / "old-design" / "12345"
        historical.mkdir(parents=True)
        (historical / "experiment_summary.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "experiment": "old-design",
                }
            ),
            encoding="utf-8",
        )
        (historical / "resolved_config.yaml").write_text(
            "name: old-design\noutput:\n  campaign: old-campaign\n",
            encoding="utf-8",
        )

        report = rebuild_run_index(self.root)

        self.assertEqual(report["indexed"], 1)
        self.assertEqual(report["failed"], 0)
        record = read_run_record(self.root, "12345")
        self.assertEqual(record["state"], "completed")
        self.assertEqual(record["campaign"], "old-campaign")
        self.assertEqual(record["run_directory"], str(historical.resolve()))

    def test_rebuild_reports_but_does_not_hide_malformed_history(self) -> None:
        malformed = self.root / "campaign" / "broken" / "54321"
        malformed.mkdir(parents=True)
        (malformed / "experiment_summary.json").write_text(
            "not json",
            encoding="utf-8",
        )

        report = rebuild_run_index(self.root)

        self.assertEqual(report["indexed"], 0)
        self.assertEqual(report["failed"], 1)
        self.assertIn("experiment_summary.json", report["failures"][0]["path"])


if __name__ == "__main__":
    unittest.main()
