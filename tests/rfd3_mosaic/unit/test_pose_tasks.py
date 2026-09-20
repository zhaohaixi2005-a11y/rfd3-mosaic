import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from rfd3_mosaic.pose_optimizer import _fixed_xyz_rotation, _linker_crowding_fraction
from rfd3_mosaic.pose_tasks import (
    _rotation_xyz_degrees,
    pose_distance_spectrum,
    prepare_pose_tasks,
    spectrum_distance,
)
from rfd3_mosaic.sampling_plan import (
    compile_sampling_plan,
    design_sampling_assignments,
    pose_plan_is_stochastic,
)
from rfd3_mosaic.schema import load_user_design


class PoseTaskTestCase(unittest.TestCase):
    def test_joint_routing_detects_crossing_without_using_a_symmetry_axis(self):
        routes = (
            {
                "from_coordinate": [-5, 0, 0],
                "to_coordinate": [5, 0, 0],
                "from_fragment_instance_id": "a",
                "to_fragment_instance_id": "b",
            },
            {
                "from_coordinate": [0, -5, 0],
                "to_coordinate": [0, 5, 0],
                "from_fragment_instance_id": "c",
                "to_fragment_instance_id": "d",
            },
        )
        self.assertEqual(_linker_crowding_fraction(routes), 1.0)
        routes[1]["from_coordinate"][2] = 5
        routes[1]["to_coordinate"][2] = 5
        self.assertEqual(_linker_crowding_fraction(routes), 0.0)

    def test_distance_ignores_global_motion_and_equivalent_copy_labels(self):
        xyz = np.array(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [8.0, 0.0, 1.0], [9.0, 2.0, 1.0]]
        )
        groups = ["a", "a", "b", "b"]
        baseline = pose_distance_spectrum(xyz, groups)
        rotated = xyz @ _fixed_xyz_rotation((12.0, 41.0, 71.0)).T + [9.0, -4.0, 1.0]
        reordered = rotated[[2, 3, 0, 1]]
        self.assertAlmostEqual(
            spectrum_distance(baseline, pose_distance_spectrum(reordered, groups)), 0.0
        )
        xyz[2:] += [5.0, 0.0, 0.0]
        self.assertGreater(
            spectrum_distance(baseline, pose_distance_spectrum(xyz, groups)), 1.0
        )

    def test_rotation_freeze_including_gimbal_lock(self):
        for angles in [(12, 34, -76), (11, 90, 27), (11, -90, 27), (0, 0, 0)]:
            rotation = _fixed_xyz_rotation(angles)
            np.testing.assert_allclose(
                _fixed_xyz_rotation(_rotation_xyz_degrees(rotation)),
                rotation,
                atol=1e-8,
            )

    def _input(self, root, symmetry="C3", motion="locked"):
        # Two residues form one asymmetric rigid seed, well away from its copies.
        pdb = root / "seed.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
            "ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00 20.00           C\nEND\n"
        )
        config = root / "design.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "name": "pose-test",
                    "input": str(pdb),
                    "symmetry": symmetry,
                    "task": "preserve_supplied_geometry",
                    "output": {
                        "root": str(root / "results"),
                        "campaign": "shared-results",
                    },
                    "preferences": {"component_motion": motion},
                    "generation": [
                        {
                            "kind": "terminal",
                            "anchor": "A1-2",
                            "terminus": "c",
                            "length": 20,
                        }
                    ],
                    "constraints": [{"kind": "fixed_xyz", "selector": "A1-2"}],
                    "sampling": {
                        "designs": 1000,
                        "initial_pose": {
                            "radius": {"minimum": 15.0, "maximum": 30.0},
                            "axial_offset": {"minimum": 8.0, "maximum": 8.0},
                            "orientation": {"method": "uniform_so3"},
                            "seed": 0,
                        },
                    },
                }
            )
        )
        return config

    def test_real_compilation_freezes_distinct_tasks_for_cyclic_and_dihedral(self):
        for symmetry in ("C3", "D3"):
            for motion in ("locked", "free"):
                with (
                    self.subTest(symmetry=symmetry, motion=motion),
                    tempfile.TemporaryDirectory() as raw,
                ):
                    root = Path(raw)
                    config = self._input(root, symmetry, motion)
                    result = prepare_pose_tasks(
                        config, root / "tasks", count=2, candidates=4, seed=10
                    )
                    self.assertEqual(result["selected_tasks"], 2)
                    for task in result["tasks"]:
                        declared = load_user_design(task["task"])
                        self.assertEqual(declared.output.root, (root / "results").resolve())
                        self.assertEqual(
                            declared.preferences.component_motion.value, motion
                        )
                        plan = compile_sampling_plan(declared)
                        self.assertFalse(pose_plan_is_stochastic(plan))
                        self.assertEqual(len(design_sampling_assignments(plan)), 1000)
                        self.assertEqual(
                            {
                                item.pose_index
                                for item in design_sampling_assignments(plan)
                            },
                            {0},
                        )
                    manifest = json.loads(
                        (root / "tasks" / "pose_tasks.json").read_text()
                    )
                    self.assertGreaterEqual(
                        manifest["tasks"][1]["minimum_selected_separation_angstrom"],
                        1.0,
                    )

    def test_insufficient_diversity_returns_fewer_without_relaxing_threshold(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = self._input(root)
            result = prepare_pose_tasks(
                config, root / "tasks", count=3, candidates=3, minimum_separation=1000.0
            )
            self.assertEqual(result["selected_tasks"], 1)
            with self.assertRaises(FileExistsError):
                prepare_pose_tasks(config, root / "tasks")

    def test_multiple_public_coupling_groups_freeze_to_correct_compiler_groups(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = self._input(root)
            payload = yaml.safe_load(config.read_text())
            pdb = root / "seed.pdb"
            pdb.write_text(
                pdb.read_text().replace("END\n", "")
                + "ATOM      3  CA  ALA B   1      10.000   0.000   0.000  1.00 20.00           C\n"
                "ATOM      4  CA  ALA B   2      13.800   0.000   0.000  1.00 20.00           C\nEND\n"
            )
            payload["constraints"][0]["coupling_group"] = "alpha"
            payload["constraints"].append(
                {"kind": "fixed_xyz", "selector": "B1-2", "coupling_group": "beta"}
            )
            payload["generation"].append(
                {"kind": "terminal", "anchor": "B1-2", "terminus": "c", "length": 20}
            )
            initial = payload["sampling"].pop("initial_pose")
            payload["sampling"]["initial_poses"] = {
                "alpha": initial,
                "beta": {
                    **initial,
                    "radius": {"minimum": 40, "maximum": 50},
                },
            }
            config.write_text(yaml.safe_dump(payload))
            result = prepare_pose_tasks(config, root / "tasks", count=2, candidates=4)
            self.assertEqual(result["selected_tasks"], 2)
            for task in result["tasks"]:
                frozen = load_user_design(task["task"])
                self.assertEqual(set(frozen.sampling.initial_poses), {"alpha", "beta"})
                self.assertFalse(pose_plan_is_stochastic(compile_sampling_plan(frozen)))


if __name__ == "__main__":
    unittest.main()
