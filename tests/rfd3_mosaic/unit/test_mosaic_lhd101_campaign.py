from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT / "scripts/rfd3_mosaic/submit_mosaic_lhd101_c3_1000.py"


class MosaicLHD101CampaignTestCase(unittest.TestCase):
    def test_full_campaign_is_exactly_sharded_and_seeded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "campaign"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--mode",
                    "full",
                    "--output-dir",
                    str(output),
                    "--run-root",
                    str(Path(directory) / "runs"),
                    "--total-designs",
                    "23",
                    "--designs-per-job",
                    "10",
                    "--seed-start",
                    "1200",
                ],
                cwd=PROJECT,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(
                (output / "campaign_manifest.json").read_text(encoding="utf-8")
            )
            frozen = [
                yaml.safe_load(Path(record["design"]).read_text(encoding="utf-8"))
                for record in manifest["records"]
            ]

        self.assertIn("designs: 23 across 3 shard(s)", completed.stdout)
        self.assertEqual(manifest["total_designs"], 23)
        self.assertEqual(manifest["shard_count"], 3)
        self.assertEqual(
            [record["requested_designs"] for record in manifest["records"]],
            [10, 10, 3],
        )
        self.assertEqual(
            [record["seed"] for record in manifest["records"]],
            [1200, 1201, 1202],
        )
        self.assertEqual(
            [design["sampling"]["designs"] for design in frozen],
            [10, 10, 3],
        )
        self.assertEqual(
            [design["sampling"]["initial_pose"]["seed"] for design in frozen],
            [1200, 1201, 1202],
        )
        self.assertEqual(len({design["name"] for design in frozen}), 3)

    def test_pilot_always_freezes_one_design(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--mode",
                    "pilot",
                    "--output-dir",
                    str(output),
                    "--run-root",
                    str(Path(directory) / "runs"),
                ],
                cwd=PROJECT,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(
                (output / "campaign_manifest.json").read_text(encoding="utf-8")
            )
            design = yaml.safe_load(
                Path(manifest["records"][0]["design"]).read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["total_designs"], 1)
        self.assertEqual(manifest["shard_count"], 1)
        self.assertEqual(design["sampling"]["designs"], 1)


if __name__ == "__main__":
    unittest.main()
