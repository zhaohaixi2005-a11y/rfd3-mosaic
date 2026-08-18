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
        with self.assertRaisesRegex(ValueError, "requires.*locked"):
            initialize_design(
                self.root / "unsafe.yaml",
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

    def test_examples_are_listable_and_copied_portably(self) -> None:
        identifiers = {item["id"] for item in available_examples()}
        self.assertEqual(identifiers, {"central-motif", "supplied-interface"})

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
            ]
        )
        self.assertEqual(arguments.profile, "local")
        self.assertEqual(arguments.component_motion, "guided")
        self.assertEqual(arguments.designs, 1)

        output = StringIO()
        with redirect_stdout(output):
            main(["examples", "--format", "json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(
            {item["id"] for item in payload},
            {"central-motif", "supplied-interface"},
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
