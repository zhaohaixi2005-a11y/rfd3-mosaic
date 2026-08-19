import json
import tempfile
import unittest
from pathlib import Path

from rfd3_mosaic.run_catalog import write_run_catalog
from rfd3_mosaic.run_index import update_run_state


class RunCatalogTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _record(self, job_id: str, state: str, revision: str) -> Path:
        run = self.root / "campaign" / f"design-{job_id}" / job_id
        run.mkdir(parents=True)
        (run / "runtime_provenance.json").write_text(
            json.dumps({"repository": {"commit": revision}}),
            encoding="utf-8",
        )
        (run / f"rfd3_input_design-{job_id}_0_model_0.cif.gz").write_bytes(
            b"structure"
        )
        update_run_state(
            root=self.root,
            job_id=job_id,
            state=state,
            experiment=f"design-{job_id}",
            campaign="campaign",
            run_directory=run,
            observed_at="2026-08-19T10:30:00+00:00",
        )
        return run

    def test_catalog_builds_version_state_structure_and_retained_views(self) -> None:
        retained_run = self._record("1001", "completed", "a" * 40)
        self._record("1002", "failed", "b" * 40)

        payload = write_run_catalog(
            self.root,
            retained_job_ids=("1001",),
        )

        catalog = Path(payload["catalog_directory"])
        current = self.root / "_catalog" / "CURRENT"
        self.assertTrue(current.is_symlink())
        self.assertEqual(current.resolve(), catalog)
        self.assertEqual((self.root / "RUN_CATALOG").resolve(), catalog)
        self.assertEqual(payload["run_count"], 2)
        self.assertEqual(payload["structure_count"], 2)
        self.assertEqual(
            (
                catalog
                / "retained"
                / f"1001__design-1001__completed__{'a' * 12}"
            ).resolve(),
            retained_run.resolve(),
        )
        self.assertTrue(
            (
                catalog
                / "by-version"
                / ("a" * 12)
                / f"1001__design-1001__completed__{'a' * 12}"
            ).is_symlink()
        )
        self.assertTrue(
            (
                catalog
                / "by-state"
                / "failed"
                / f"1002__design-1002__failed__{'b' * 12}"
            ).is_symlink()
        )
        self.assertTrue(
            (
                catalog
                / "by-date"
                / "2026-08-19"
                / f"1001__design-1001__completed__{'a' * 12}"
            ).is_symlink()
        )
        day_summary = catalog / "by-date" / "2026-08-19" / "RUNS.md"
        self.assertIn("`1001`", day_summary.read_text(encoding="utf-8"))
        self.assertEqual(len(list((catalog / "structures").iterdir())), 2)
        self.assertIn("`1001`", (catalog / "CATALOG.md").read_text())

        refreshed = write_run_catalog(self.root)
        refreshed_catalog = Path(refreshed["catalog_directory"])
        self.assertIn("1001", refreshed["retained_job_ids"])
        self.assertEqual(
            len(list((refreshed_catalog / "retained").glob("1001__*"))),
            1,
        )

    def test_explicit_existing_output_fails_closed(self) -> None:
        output = self.root / "existing"
        output.mkdir()

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            write_run_catalog(self.root, output_directory=output)


if __name__ == "__main__":
    unittest.main()
