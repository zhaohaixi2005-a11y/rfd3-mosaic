from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

from rfd3_mosaic.provenance.software import (
    collect_repository_provenance,
    collect_runtime_provenance,
    load_compatibility_manifest,
)


class SoftwareProvenanceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _git(self, *arguments: str) -> None:
        subprocess.run(
            ["git", "-C", str(self.root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_repository_provenance_records_commit_and_dirty_diff(self) -> None:
        self._git("init")
        self._git("config", "user.email", "test@example.invalid")
        self._git("config", "user.name", "Mosaic Test")
        tracked = self.root / "tracked.txt"
        tracked.write_text("first\n", encoding="utf-8")
        self._git("add", "tracked.txt")
        self._git("commit", "-m", "initial")

        clean = collect_repository_provenance(self.root)
        self.assertEqual(len(clean["commit"]), 40)
        self.assertFalse(clean["tracked_dirty"])
        self.assertIsNone(clean["working_tree_diff_sha256"])

        tracked.write_text("changed\n", encoding="utf-8")
        dirty = collect_repository_provenance(self.root)
        self.assertTrue(dirty["tracked_dirty"])
        self.assertEqual(len(dirty["working_tree_diff_sha256"]), 64)

    def test_compatibility_manifest_is_fingerprinted(self) -> None:
        manifest = self.root / "foundry.yaml"
        manifest.write_text(
            yaml.safe_dump({"schema_version": 1, "engine_id": "test"}),
            encoding="utf-8",
        )

        record = load_compatibility_manifest(manifest)

        self.assertEqual(record["manifest"]["engine_id"], "test")
        self.assertEqual(len(record["sha256"]), 64)

    def test_runtime_provenance_marks_missing_checkpoint(self) -> None:
        checkpoint = self.root / "missing.ckpt"

        record = collect_runtime_provenance(
            self.root,
            checkpoint=checkpoint,
            checkpoint_sha256="a" * 64,
        )

        self.assertFalse(record["checkpoint"]["exists"])
        self.assertEqual(record["checkpoint"]["declared_sha256"], "a" * 64)
        self.assertIn("python", record)
        self.assertIn("platform", record)


if __name__ == "__main__":
    unittest.main()
