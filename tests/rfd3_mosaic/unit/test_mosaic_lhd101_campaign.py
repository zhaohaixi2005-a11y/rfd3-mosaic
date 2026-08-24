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
    def test_submission_defers_full_runtime_preflight_to_compute_node(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('"--defer-runtime-preflight"', source)

    def test_full_comparison_defaults_to_one_pose_per_design(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "campaign"
            subprocess.run(
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
                    "3",
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

        self.assertEqual(manifest["total_designs"], 3)
        self.assertEqual(manifest["designs_per_job"], 1)
        self.assertEqual(manifest["compiled_pose_count"], 3)
        self.assertEqual(
            manifest["pose_semantics"],
            "one_independently_seeded_feasible_pose_per_design; one RFD3 "
            "model load per shard",
        )
        self.assertEqual(
            [record["pose_seed"] for record in manifest["records"]],
            [1200, 1201, 1202],
        )

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
        # A shard is now a model-loading unit, not a pose-sharing unit.
        # Every requested design receives an independently seeded feasible
        # pose while the three shards still load RFD3 only three times.
        self.assertEqual(manifest["compiled_pose_count"], 23)
        self.assertEqual(
            manifest["pose_semantics"],
            "one_independently_seeded_feasible_pose_per_design; one RFD3 "
            "model load per shard",
        )
        self.assertEqual(
            [record["requested_designs"] for record in manifest["records"]],
            [10, 10, 3],
        )
        self.assertEqual(
            [record["seed"] for record in manifest["records"]],
            [1200, 1210, 1220],
        )
        self.assertEqual(
            [record["design_start"] for record in manifest["records"]],
            [0, 10, 20],
        )
        derived_pose_seeds = [
            int(record["pose_seed"]) + offset
            for record in manifest["records"]
            for offset in range(int(record["requested_designs"]))
        ]
        derived_diffusion_seeds = [
            int(record["diffusion_seed"]) + offset
            for record in manifest["records"]
            for offset in range(int(record["requested_designs"]))
        ]
        self.assertEqual(derived_pose_seeds, list(range(1200, 1223)))
        self.assertEqual(derived_diffusion_seeds, list(range(1200, 1223)))
        self.assertEqual(
            [design["sampling"]["designs"] for design in frozen],
            [10, 10, 3],
        )
        self.assertEqual(
            [design["sampling"]["initial_pose"]["seed"] for design in frozen],
            [1200, 1210, 1220],
        )
        self.assertTrue(
            all(design["sampling"]["scaffold_packing"] == "off" for design in frozen)
        )
        self.assertTrue(
            all(
                design["sampling"]["scaffold_core_quality"]["required"] is False
                for design in frozen
            )
        )
        self.assertTrue(
            all(
                "inter_chain_weight" not in design["guidance"]
                and "inter_chain_excess_penalty" not in design["guidance"]
                for design in frozen
            )
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

    def test_pilot_can_freeze_independent_screened_pose_seeds(self) -> None:
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
                    "--seed-start",
                    "20000",
                    "--pose-seeds",
                    "10063",
                    "10039",
                    "10048",
                    "10027",
                ],
                cwd=PROJECT,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(
                (output / "campaign_manifest.json").read_text(encoding="utf-8")
            )
            designs = [
                yaml.safe_load(Path(record["design"]).read_text(encoding="utf-8"))
                for record in manifest["records"]
            ]

        self.assertEqual(manifest["total_designs"], 4)
        self.assertEqual(manifest["shard_count"], 4)
        self.assertEqual(
            [record["pose_seed"] for record in manifest["records"]],
            [10063, 10039, 10048, 10027],
        )
        self.assertEqual(
            [record["diffusion_seed"] for record in manifest["records"]],
            [20000, 20001, 20002, 20003],
        )
        self.assertEqual(
            [design["sampling"]["initial_pose"]["seed"] for design in designs],
            [10063, 10039, 10048, 10027],
        )
        self.assertEqual(
            [design["sampling"]["seed"] for design in designs],
            [20000, 20001, 20002, 20003],
        )

    def test_pilot_expands_pose_by_diffusion_seed_matrix(self) -> None:
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
                    "--seed-start",
                    "30000",
                    "--pose-seeds",
                    "10063",
                    "10039",
                    "--diffusion-seeds-per-pose",
                    "3",
                ],
                cwd=PROJECT,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(
                (output / "campaign_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["total_designs"], 6)
        self.assertEqual(manifest["shard_count"], 6)
        self.assertEqual(manifest["diffusion_seeds_per_pose"], 3)
        self.assertEqual(
            [record["pose_seed"] for record in manifest["records"]],
            [10063, 10063, 10063, 10039, 10039, 10039],
        )
        self.assertEqual(
            [record["diffusion_seed"] for record in manifest["records"]],
            [30000, 30001, 30002, 30003, 30004, 30005],
        )


if __name__ == "__main__":
    unittest.main()
