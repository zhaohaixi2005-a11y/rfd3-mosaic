from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT / "scripts/rfd3_mosaic/submit_packing_replicates.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("submit_packing_replicates", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackingCampaignTestCase(unittest.TestCase):
    def test_locked_and_guided_use_paired_independent_pose_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "campaign"
            script = _load_script()
            script._run = lambda command: subprocess.CompletedProcess(
                command,
                0,
                stdout="validation stubbed by unit test\n",
            )
            script._revision = lambda project: "test-revision"
            arguments = [
                str(SCRIPT),
                "--output-dir",
                str(output),
                "--run-root",
                str(Path(directory) / "runs"),
                "--seed",
                "71000",
                "--designs-per-job",
                "2",
            ]
            previous = sys.argv
            try:
                sys.argv = arguments
                script.main()
            finally:
                sys.argv = previous
            manifest = json.loads(
                (output / "campaign_manifest.json").read_text(encoding="utf-8")
            )
            designs = [
                yaml.safe_load(Path(record["design"]).read_text(encoding="utf-8"))
                for record in manifest["records"]
            ]

        self.assertEqual(len(designs), 2)
        self.assertEqual(manifest["requested_output_count"], 4)
        self.assertEqual(
            [record["pose_seed_start"] for record in manifest["records"]],
            [1_071_000, 1_071_000],
        )
        self.assertEqual(
            [design["sampling"]["initial_pose"]["seed"] for design in designs],
            [1_071_000, 1_071_000],
        )
        self.assertTrue(
            all(
                design["sampling"]["initial_pose"]["orientation"]["method"]
                == "uniform_so3"
                for design in designs
            )
        )
        self.assertTrue(
            all(design["sampling"]["replicates_per_pose"] == 1 for design in designs)
        )

    def test_can_defer_complete_preflight_to_allocated_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "campaign"
            script = _load_script()
            commands: list[list[str]] = []

            def run(command: list[str]) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                job_id = 81000 + len(commands)
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=f"Submitted batch job {job_id}\n",
                )

            script._run = run
            script._revision = lambda project: "test-revision"
            arguments = [
                str(SCRIPT),
                "--output-dir",
                str(output),
                "--run-root",
                str(Path(directory) / "runs"),
                "--mode",
                "locked",
                "--seed",
                "83000",
                "--designs-per-job",
                "2",
                "--defer-runtime-preflight",
                "--submit",
            ]
            previous = sys.argv
            try:
                sys.argv = arguments
                script.main()
            finally:
                sys.argv = previous
            manifest = json.loads(
                (output / "campaign_manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(commands[0][3], "plan")
        self.assertEqual(commands[1][3], "run")
        self.assertIn("--defer-runtime-preflight", commands[1])
        self.assertEqual(
            manifest["records"][0]["submission_preflight"],
            "complete_on_allocated_worker",
        )
        self.assertTrue(manifest["records"][0]["submitted"])


if __name__ == "__main__":
    unittest.main()
