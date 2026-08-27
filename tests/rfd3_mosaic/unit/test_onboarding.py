import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import yaml

from rfd3_mosaic.cli import _parser, main
from rfd3_mosaic.onboarding import (
    available_examples,
    available_profiles,
    copy_example,
    copy_slurm_profile,
    initialize_design,
)
from rfd3_mosaic.schema import load_user_design


class OnboardingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.structure = self.root / "input.pdb"
        self.structure.write_text("REMARK schema-only fixture\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_init_central_motif_writes_short_valid_public_design(self) -> None:
        path = initialize_design(
            self.root / "central.yaml",
            task="central-motif",
            input_path=self.structure,
            symmetry="C4",
            name="guided-central",
            profile="local",
            run_root=self.root / "runs",
            motif_selector="A12-20",
            component_motion="guided",
            designs=24,
        )

        design = load_user_design(path)

        self.assertEqual(design.name, "guided-central")
        self.assertEqual(str(design.symmetry), "C4")
        self.assertEqual(design.task.value, "create_symmetric_interface")
        self.assertEqual(design.fixed_arrangement.value, "optimize_components")
        self.assertEqual(design.preferences.component_motion.value, "guided")
        self.assertEqual(design.sampling.designs, 24)
        self.assertEqual(len(design.generation), 2)
        self.assertEqual(design.resources.profile, "local")

    def test_init_supplied_interface_preserves_one_joint_seed(self) -> None:
        path = initialize_design(
            self.root / "interface.yaml",
            task="supplied-interface",
            input_path=self.structure,
            symmetry="C3",
            name=None,
            profile="local",
            run_root=Path("runs"),
            side_a="A165-194",
            side_b="B211-241",
        )

        design = load_user_design(path)

        self.assertEqual(design.task.value, "preserve_supplied_geometry")
        self.assertEqual(
            {constraint.coupling_group for constraint in design.constraints},
            {"supplied_interface"},
        )
        self.assertEqual(design.generation[0].length.minimum, 70)
        self.assertEqual(design.generation[0].length.maximum, 100)
        self.assertEqual(design.generation[0].orbit_offset, "nearest_adjacent")

    def test_init_supplied_interface_requires_explicit_noncyclic_graph(self) -> None:
        with self.assertRaisesRegex(ValueError, "cyclic Cn symmetry only"):
            initialize_design(
                self.root / "interface-t.yaml",
                task="supplied-interface",
                input_path=self.structure,
                symmetry="T",
                name=None,
                profile="local",
                run_root=Path("runs"),
                side_a="A165-194",
                side_b="B211-241",
            )

    def test_init_supplied_interface_higher_oligomer_is_non_covalent(self) -> None:
        path = initialize_design(
            self.root / "higher-oligomer.yaml",
            task="supplied-interface",
            input_path=self.structure,
            symmetry="C3",
            name=None,
            profile="local",
            run_root=self.root / "runs",
            side_a="A1-10",
            side_b="B1-20",
            interface_scaffold="terminal-extensions",
            new_oligomer_interface=True,
            sequence_conditioning="masked",
            redesign_motif_sidechains=True,
            ligand_selectors=("L1",),
            designs=8,
            pose_radius_minimum=20.0,
            pose_radius_maximum=32.0,
            pose_axial_minimum=-4.0,
            pose_axial_maximum=4.0,
            pose_orientation="uniform_so3",
            pose_seed=1000,
        )

        design = load_user_design(path)

        self.assertEqual(len(design.generation), 4)
        self.assertTrue(all(item.kind == "terminal" for item in design.generation))
        self.assertEqual(design.sampling.scaffold_packing, "symmetric_generated")
        self.assertEqual(
            {item.mode.value for item in design.conditioning.sequence},
            {"masked"},
        )
        self.assertTrue(design.conditioning.redesign_motif_sidechains)
        self.assertEqual(design.conditioning.ligands[0].selector, "L1")
        self.assertEqual(design.sampling.replicates_per_pose, 1)
        self.assertEqual(design.sampling.initial_pose.radius.minimum, 20.0)
        self.assertEqual(
            design.sampling.initial_pose.orientation.method,
            "uniform_so3",
        )

    def test_init_pose_options_are_explicit_and_complete(self) -> None:
        with self.assertRaisesRegex(ValueError, "both minimum and maximum"):
            initialize_design(
                self.root / "bad-pose.yaml",
                task="central-motif",
                input_path=self.structure,
                symmetry="C3",
                name=None,
                profile="local",
                run_root=self.root / "runs",
                motif_selector="A1",
                pose_radius_minimum=10.0,
            )
        with self.assertRaisesRegex(ValueError, "require an explicit"):
            initialize_design(
                self.root / "bad-orientation.yaml",
                task="central-motif",
                input_path=self.structure,
                symmetry="C3",
                name=None,
                profile="local",
                run_root=self.root / "runs",
                motif_selector="A1",
                pose_orientation="uniform_so3",
            )

    def test_init_refuses_unsafe_or_destructive_requests(self) -> None:
        output = self.root / "design.yaml"
        initialize_design(
            output,
            task="central-motif",
            input_path=self.structure,
            symmetry="C3",
            name=None,
            profile="local",
            run_root=Path("runs"),
            motif_selector="A1",
        )
        with self.assertRaises(FileExistsError):
            initialize_design(
                output,
                task="central-motif",
                input_path=self.structure,
                symmetry="C3",
                name=None,
                profile="local",
                run_root=Path("runs"),
                motif_selector="A1",
            )
        mobile = initialize_design(
            self.root / "mobile-interface.yaml",
            task="supplied-interface",
            input_path=self.structure,
            symmetry="C3",
            name=None,
            profile="local",
            run_root=Path("runs"),
            side_a="A1",
            side_b="B1",
            component_motion="free",
        )
        mobile_design = load_user_design(mobile)
        self.assertEqual(
            mobile_design.fixed_arrangement.value,
            "optimize_components",
        )
        self.assertEqual(
            mobile_design.preferences.component_motion.value,
            "free",
        )

    def test_examples_are_listable_and_copied_portably(self) -> None:
        identifiers = {item["id"] for item in available_examples()}
        self.assertEqual(
            identifiers,
            {
                "central-motif",
                "supplied-interface",
                "supplied-interface-oligomer",
            },
        )

        path = copy_example(
            "central-motif",
            self.root / "copied.yaml",
            overwrite=False,
        )
        design = load_user_design(path)

        self.assertTrue(design.input.is_absolute())
        self.assertEqual(design.resources.profile, "local")
        self.assertEqual(design.output.root, (self.root / "runs").resolve())

    def test_profiles_list_public_options_and_copy_slurm_template(self) -> None:
        profiles = available_profiles()
        public = {item["id"] for item in profiles if item["scope"] == "public"}
        self.assertEqual(public, {"local", "slurm-example"})

        destination = copy_slurm_profile(
            self.root / "cluster.yaml",
            overwrite=False,
        )
        payload = yaml.safe_load(destination.read_text(encoding="utf-8"))

        self.assertEqual(payload["executor"], "slurm")

    def test_cli_exposes_guided_init_and_machine_readable_discovery(self) -> None:
        arguments = _parser().parse_args(
            [
                "init",
                "design.yaml",
                "--task",
                "central-motif",
                "--input",
                "input.pdb",
                "--motif-selector",
                "A1-10",
                "--component-motion",
                "guided",
                "--pose-radius-minimum",
                "18",
                "--pose-radius-maximum",
                "30",
                "--pose-orientation",
                "uniform_so3",
            ]
        )
        self.assertEqual(arguments.profile, "local")
        self.assertEqual(arguments.component_motion, "guided")
        self.assertEqual(arguments.designs, 1)
        self.assertEqual(arguments.pose_radius_minimum, 18.0)
        self.assertEqual(arguments.pose_orientation, "uniform_so3")

        output = StringIO()
        with redirect_stdout(output):
            main(["examples", "--format", "json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(
            {item["id"] for item in payload},
            {
                "central-motif",
                "supplied-interface",
                "supplied-interface-oligomer",
            },
        )

    def test_cli_init_reports_the_complete_next_lifecycle(self) -> None:
        destination = self.root / "design.yaml"
        output = StringIO()
        with redirect_stdout(output):
            main(
                [
                    "init",
                    str(destination),
                    "--task",
                    "central-motif",
                    "--input",
                    str(self.structure),
                    "--motif-selector",
                    "A1-10",
                ]
            )

        text = output.getvalue()
        self.assertIn("RFD3-Mosaic design created", text)
        self.assertIn("rfd3-mosaic plan", text)
        self.assertIn("rfd3-mosaic validate", text)
        self.assertIn("rfd3-mosaic run", text)
        self.assertTrue(destination.is_file())


if __name__ == "__main__":
    unittest.main()
