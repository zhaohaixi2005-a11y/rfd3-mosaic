import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from rfd3_mosaic.provenance.software import verify_file_identities
from rfd3_mosaic.provenance.source_snapshot import (
    create_source_snapshot,
    verify_source_snapshot_tree,
)


class SourceSnapshotTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        subprocess.run(
            ["git", "init", "-q", str(self.repository)],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repository),
                "config",
                "user.email",
                "test@example.invalid",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repository),
                "config",
                "user.name",
                "Mosaic Test",
            ],
            check=True,
        )
        package = self.repository / "src" / "test_package"
        package.mkdir(parents=True)
        self.module = package / "module.py"
        self.module.write_text("value = 1\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(self.repository), "add", "."],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repository),
                "commit",
                "-qm",
                "initial",
            ],
            check=True,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create_and_extract(self):
        archive = self.root / "source_snapshot.tar.gz"
        identity = create_source_snapshot(
            self.repository,
            archive,
            roots=("src/test_package",),
        )
        extracted = self.root / "extracted"
        extracted.mkdir()
        with tarfile.open(archive, mode="r:gz") as handle:
            handle.extractall(extracted)
        return identity, extracted

    def test_snapshot_freezes_and_verifies_runtime_source(self) -> None:
        identity, extracted = self._create_and_extract()

        verify_file_identities([identity["archive"]])
        manifest = verify_source_snapshot_tree(
            extracted,
            expected_manifest_sha256=identity["manifest_sha256"],
        )

        self.assertEqual(identity["file_count"], 1)
        self.assertEqual(manifest["files"][0]["path"], "src/test_package/module.py")

    def test_snapshot_tree_rejects_extracted_source_mutation(self) -> None:
        identity, extracted = self._create_and_extract()
        extracted_module = extracted / "src" / "test_package" / "module.py"
        extracted_module.write_text("value = 2\n", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "SHA256 changed"):
            verify_source_snapshot_tree(
                extracted,
                expected_manifest_sha256=identity["manifest_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
