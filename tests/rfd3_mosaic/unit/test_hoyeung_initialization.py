"""LHD101 C3 initializer integration tests (no model weights or GPU)."""

import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rfd3_mosaic.cli import main
from rfd3_mosaic.hoyeung_initialization import draw_geometry, prepare_hoyeung_poses, read_lhd101
from rfd3_mosaic.schema import load_user_design

EXAMPLE = Path(__file__).resolve().parents[3] / "examples/rfd3_mosaic/simple_interface_seed.yaml"
SEED = EXAMPLE.parent / "lhd101_c3/inputs/7mwr_interface.pdb"


class HoYeungInitializationTest(unittest.TestCase):
    def test_exact_draw_contract(self):
        atoms = read_lhd101(SEED)
        zero = draw_geometry(atoms, random.Random(0), 70, 100)
        eight = draw_geometry(atoms, random.Random(8), 70, 100)
        self.assertEqual((zero["rotation_deg"], zero["x_translation_angstrom"],
                          zero["B_to_A_orbit_offset"], zero["linker_length"]),
                         ([17, 35, -160], 24, 1, 100))
        self.assertEqual((eight["B_to_A_orbit_offset"], eight["linker_length"]), (-1, 76))
        self.assertEqual(zero["coordinates"].shape, (3, 496, 3))
        self.assertTrue(np.isfinite(zero["coordinates"]).all())

    def test_entry_point_makes_one_fresh_trial_without_shortlist(self):
        with tempfile.TemporaryDirectory() as d:
            result = Path(d) / "poses"
            main(["prepare-poses", str(EXAMPLE), "--output-dir", str(result), "--count", "10"])
            report = json.loads((result / "pose_tasks.json").read_text())
            self.assertEqual(len(report["attempts"]), 10)
            self.assertEqual(len({row["coordinates_sha256"] for row in report["attempts"]}), 10)
            for row in report["attempts"]:
                design = load_user_design(Path(row["task"]))
                self.assertEqual(design.sampling.designs, 1)
                self.assertEqual(design.generation[0].orbit_offset, row["B_to_A_orbit_offset"])
                self.assertEqual(design.generation[0].length.minimum, row["linker_length"])
                self.assertEqual(design.generation[0].length.maximum, row["linker_length"])

    def test_failed_draw_stays_in_ledger_without_replacement(self):
        with tempfile.TemporaryDirectory() as d, patch(
            "rfd3_mosaic.hoyeung_initialization.draw_geometry", side_effect=ValueError("test")
        ) as draw:
            report = prepare_hoyeung_poses(EXAMPLE, Path(d) / "out", count=4, seed=3)
            self.assertEqual(draw.call_count, 4)
            self.assertEqual([a["pose_seed"] for a in report["attempts"]], [3, 4, 5, 6])
            self.assertTrue(all(a["status"] == "failed" for a in report["attempts"]))

    def test_modified_input_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seed.pdb"
            path.write_bytes(SEED.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                read_lhd101(path)


if __name__ == "__main__":
    unittest.main()
