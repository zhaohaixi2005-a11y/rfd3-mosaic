import gzip
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from rfd3_mosaic.structure_archive import create_generated_cif_archive


class StructureArchiveTestCase(unittest.TestCase):
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
