import gzip
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from rfd3_mosaic.structure_archive import (
    GeneratedCifMirror,
    create_generated_cif_archive,
    materialize_plain_cif,
)


class StructureArchiveTestCase(unittest.TestCase):
    def test_materialize_plain_cif_is_atomic_and_preserves_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "design_00000_model_0.cif.gz"
            with gzip.open(source, "wt") as handle:
                handle.write("data_design_0\n")

            output = materialize_plain_cif(source, root / "plain")

            self.assertEqual(output.name, "design_00000_model_0.cif")
            self.assertEqual(output.read_text(), "data_design_0\n")
            self.assertFalse(output.with_suffix(".cif.tmp").exists())

    def test_incremental_mirror_adds_each_completed_structure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "generated_structures_cif"
            mirror = GeneratedCifMirror(
                root,
                destination,
                poll_interval_seconds=0.01,
            ).start()
            try:
                for index in range(2):
                    source = root / f"design_{index:05d}_model_0.cif.gz"
                    with gzip.open(source, "wt") as handle:
                        handle.write(f"data_design_{index}\n")
                    mirror.scan(tolerate_incomplete_gzip=False)
                    self.assertTrue(
                        (
                            destination
                            / f"design_{index:05d}_model_0.cif"
                        ).is_file()
                    )
            finally:
                produced = mirror.stop(inference_succeeded=True)

            self.assertEqual(len(produced), 2)
            manifest = json.loads(
                (destination / "manifest.json").read_text()
            )
            self.assertEqual(manifest["produced_designs"], 2)

    def test_incremental_mirror_ignores_incomplete_gzip_until_finished(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "generated_structures_cif"
            source = root / "design_00000_model_0.cif.gz"
            source.write_bytes(b"\x1f\x8b")
            mirror = GeneratedCifMirror(root, destination)

            self.assertEqual(
                mirror.scan(tolerate_incomplete_gzip=True),
                (),
            )
            self.assertFalse(
                (destination / "design_00000_model_0.cif").exists()
            )

            with gzip.open(source, "wt") as handle:
                handle.write("data_design_0\n")
            produced = mirror.scan(tolerate_incomplete_gzip=False)

            self.assertEqual(len(produced), 1)
            self.assertEqual(produced[0].read_text(), "data_design_0\n")

    def test_archive_contains_only_plain_cif_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = []
            for index in range(3):
                stem = f"design_{index:05d}_model_0"
                result = root / f"{stem}.json"
                result.write_text(json.dumps({"index": index}))
                with gzip.open(root / f"{stem}.cif.gz", "wt") as handle:
                    handle.write(f"data_design_{index}\n")
                results.append(result)

            output = root / "generated_structures_cif.zip"
            manifest = create_generated_cif_archive(
                results,
                output,
                requested_designs=3,
            )

            with zipfile.ZipFile(output) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [f"design_{index:05d}_model_0.cif" for index in range(3)],
                )
                self.assertEqual(
                    archive.read("design_00001_model_0.cif"),
                    b"data_design_1\n",
                )
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["member_count"], 3)
            self.assertTrue(Path(str(manifest["manifest"])).is_file())


if __name__ == "__main__":
    unittest.main()
