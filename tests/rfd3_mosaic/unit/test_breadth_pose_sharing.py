from __future__ import annotations

import importlib.util
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

PROJECT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT / "scripts/rfd3_mosaic/submit_scientific_breadth_campaign.py"


class BreadthPoseSharingTest(unittest.TestCase):
    def test_lhd_matrix_motion_declarations_follow_public_schema(self):
        from copy import deepcopy

        from rfd3_mosaic.schema import UserDesignSpec

        module = runpy.run_path(
            str(PROJECT / "scripts/rfd3_mosaic/prepare_lhd101_benchmark.py")
        )
        input_path = (
            PROJECT / "examples/rfd3_mosaic/lhd101_c3/inputs/7mwr_interface.pdb"
        )
        total = 0
        for _, payload, modes, count, _, _ in module["families"](input_path, 50):
            for mode in modes:
                candidate = deepcopy(payload)
                module["apply_motion"](candidate, mode)
                design = UserDesignSpec.model_validate(candidate)
                self.assertEqual(design.sampling.designs, 50)
                if design.components:
                    self.assertTrue(
                        all(
                            component.pose.mode == "bounded_mobile"
                            for component in design.components.values()
                        )
                    )
                total += count
        self.assertEqual(total, 22)

    def test_motion_pairs_and_recovery_shards_share_frozen_geometry(self):
        spec = importlib.util.spec_from_file_location("breadth_pose_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.yaml"
            profile.write_text("schema_version: 1\n")
            output = root / "campaign"
            arguments = [
                str(SCRIPT),
                "--output-dir",
                str(output),
                "--run-root",
                str(root / "runs"),
                "--profile-small",
                str(profile),
                "--profile-large",
                str(profile),
                "--site-label",
                "test",
                "--designs",
                "50",
                "--designs-per-job",
                "25",
                "--case",
                "c3-supplied-interface-locked",
                "--case",
                "c3-supplied-interface-guided",
            ]
            with (
                patch.object(sys, "argv", arguments),
                patch.object(Path, "cwd", return_value=PROJECT),
            ):
                module.main()
            manifest = json.loads((output / "campaign_manifest.json").read_text())
            records = manifest["records"]
            self.assertEqual(len(records), 4)
            configs = [
                yaml.safe_load(Path(item["config"]).read_text()) for item in records
            ]
            poses = [item["sampling"]["initial_pose"] for item in configs]
            self.assertTrue(all(pose == poses[0] for pose in poses))
            self.assertEqual(poses[0]["orientation"]["method"], "fixed")
            self.assertEqual(
                poses[0]["radius"]["minimum"], poses[0]["radius"]["maximum"]
            )
            seeds = [item["sampling"]["seed"] for item in configs]
            self.assertEqual(seeds[0], seeds[2])
            self.assertEqual(seeds[1], seeds[3])
            self.assertEqual(seeds[1] - seeds[0], 25)
            self.assertEqual(manifest["requested_designs"], 100)


if __name__ == "__main__":
    unittest.main()
