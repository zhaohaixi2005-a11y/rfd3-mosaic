import json
import tempfile
import unittest
from pathlib import Path

from rfd3_mosaic.run_index import (
    read_run_record,
    record_submission,
    update_run_state,
)
from rfd3_mosaic.run_layout import dated_run_directory
from rfd3_mosaic.run_reorganization import (
    apply_date_reorganization,
    plan_date_reorganization,
)


class RunReorganizationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _indexed_run(self, job_id: str, state: str) -> tuple[Path, Path]:
        source = self.root / "old-campaign" / "experiment" / job_id
        source.mkdir(parents=True)
        (source / "experiment_summary.json").write_text(
            json.dumps({"status": state}),
            encoding="utf-8",
        )
        submission = (
            self.root / "old-campaign" / "_submissions" / job_id / "stamp"
        )
        submission.mkdir(parents=True)
        (submission / "submission.json").write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "expected_run_directory": str(source),
                    "run_root": str(source.parent),
                }
            ),
            encoding="utf-8",
        )
        update_run_state(
            root=self.root,
            job_id=job_id,
            state=state,
            experiment="clear-experiment",
            campaign="old-campaign",
            run_directory=source,
            observed_at="2026-08-19T09:15:00+00:00",
        )
        record_submission(
            root=self.root,
            job_id=job_id,
            experiment="clear-experiment",
            campaign="old-campaign",
            run_directory=source,
            submission_directory=submission,
            executor="local",
        )
        return source, submission

    def test_plan_and_apply_move_real_run_and_update_references(self) -> None:
        source, submission = self._indexed_run("5754107", "completed")

        plan = plan_date_reorganization(self.root)

        self.assertEqual(plan["ready_count"], 1)
        target = dated_run_directory(
            self.root,
            run_day="2026-08-19",
            experiment="clear-experiment",
            job_id="5754107",
        )
        self.assertEqual(Path(plan["entries"][0]["target"]), target)
        self.assertTrue(source.is_dir())

        result = apply_date_reorganization(self.root)

        self.assertEqual(result["completed_count"], 1)
        self.assertFalse(source.exists())
        self.assertTrue(target.is_dir())
        indexed = read_run_record(self.root, "5754107")
        self.assertEqual(Path(indexed["run_directory"]), target)
        moved_submission = Path(indexed["submission_directory"])
        self.assertFalse(submission.exists())
        receipt = json.loads(
            (moved_submission / "submission.json").read_text(encoding="utf-8")
        )
        self.assertEqual(Path(receipt["expected_run_directory"]), target)
        self.assertTrue(Path(result["manifest"]).is_file())

    def test_nonterminal_run_is_never_moved(self) -> None:
        source = self.root / "campaign" / "experiment" / "5754999"
        source.mkdir(parents=True)
        index = self.root / ".rfd3-mosaic" / "jobs"
        index.mkdir(parents=True)
        (index / "5754999.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "job_id": "5754999",
                    "experiment": "running-design",
                    "campaign": "campaign",
                    "executor": "slurm",
                    "state": "submitted",
                    "created_at": "2026-08-19T10:00:00+00:00",
                    "updated_at": "2026-08-19T10:00:00+00:00",
                    "run_directory": str(source),
                    "submission_directory": None,
                }
            ),
            encoding="utf-8",
        )

        result = apply_date_reorganization(self.root)

        self.assertEqual(result["completed_count"], 0)
        self.assertTrue(source.is_dir())
        self.assertIn("not terminal", result["entries"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
