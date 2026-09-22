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
from rfd3_mosaic.schema import UserDesignSpec, load_user_design


class OnboardingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.structure = self.root / "input.pdb"
        self.structure.write_text("REMARK schema-only fixture\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_inferred_compact_tasks_preserve_explicit_design_semantics(self) -> None:
        for task in ("central-motif", "supplied-interface"):
            for motion in ("locked", "guided", "free"):
                with self.subTest(task=task, motion=motion):
                    name = f"{task}-{motion}"
                    selectors = (
                        {"motif_selector": "A12-20"}
                        if task == "central-motif"
                        else {"side_a": "A12-20", "side_b": "B30-40"}
                    )
                    path = initialize_design(
                        self.root / f"{name}.yaml",
                        input_path=self.structure,
                        symmetry="C3",
                        name=name,
                        profile="local",
                        run_root=self.root / "runs",
                        component_motion=motion,
                        designs=1000,
                        **selectors,
                    )
                    compact = yaml.safe_load(path.read_text(encoding="utf-8"))
                    # This explicit declaration is the previous initializer's
                    # contract, independent of its new serialization strategy.
                    expected = {
                        "schema_version": 1,
                        "name": name,
                        "input": str(self.structure.resolve()),
                        "symmetry": "C3",
                        "preferences": {
                            "packing": "balanced",
                            "cavity": "auto",
                            "diversity": "medium",
                            "interface_area": "auto",
                            "component_motion": motion,
                        },
                        "sampling": {"timesteps": 200, "designs": 1000, "seed": 42},
                        "resources": {"profile": "local"},
                        "output": {
                            "root": str((self.root / "runs").resolve()),
                            "campaign": name,
                        },
                    }
                    if task == "central-motif":
                        expected.update(
                            task="create_symmetric_interface",
                            generation=[
                                {
                                    "kind": "terminal",
                                    "anchor": "A12-20",
                                    "terminus": terminus,
                                    "length": 35,
                                }
                                for terminus in ("n", "c")
                            ],
                            constraints=[{"kind": "fixed_xyz", "selector": "A12-20"}],
                        )
                    else:
                        expected.update(
                            task="preserve_supplied_geometry",
                            generation=[
                                {
                                    "kind": "between",
                                    "from_selector": "B30-40",
                                    "to_selector": "A12-20",
                                    "orbit_offset": "nearest_adjacent",
                                    "length": {"minimum": 70, "maximum": 100},
                                }
                            ],
                            constraints=[
                                {
                                    "kind": "fixed_xyz",
                                    "selector": selector,
                                    "coupling_group": "supplied_interface",
                                }
                                for selector in ("A12-20", "B30-40")
                            ],
                        )
                    self.assertEqual(
                        UserDesignSpec.model_validate(compact),
                        UserDesignSpec.model_validate(expected),
                    )
                    self.assertEqual(
                        compact["preferences"], {"component_motion": motion}
                    )
                    self.assertEqual(compact["sampling"], {"designs": 1000})

    def test_task_inference_rejects_missing_mixed_and_conflicting_selectors(
        self,
    ) -> None:
        cases = (
            ({}, "Provide --motif-selector"),
            ({"side_a": "A1-10"}, "requires both"),
            ({"side_b": "B1-10"}, "requires both"),
            ({"motif_selector": "A1-10", "side_a": "A1-10"}, "ambiguous"),
            ({"motif_selector": "A1-10", "side_b": "B1-10"}, "ambiguous"),
            (
                {"task": "central-motif", "side_a": "A1-10", "side_b": "B1-10"},
                "conflicts",
            ),
            ({"task": "supplied-interface", "motif_selector": "A1-10"}, "conflicts"),
            (
                {
                    "task": "supplied-interface",
                    "motif_selector": "A1-10",
                    "side_a": "A1-10",
                    "side_b": "B1-10",
                },
                "ambiguous",
            ),
        )
        for index, (options, message) in enumerate(cases):
            with self.subTest(options=options):
                output = self.root / f"invalid-{index}.yaml"
                with self.assertRaisesRegex(ValueError, message):
                    initialize_design(
                        output,
                        input_path=self.structure,
                        symmetry="C3",
                        name=None,
                        profile="local",
                        run_root=self.root / "runs",
                        **options,
                    )
                self.assertFalse(output.exists())

    def test_central_initializer_does_not_silently_discard_interface_options(
        self,
    ) -> None:
        options = (
            {"interface_scaffold": "terminal-extensions"},
            {"new_oligomer_interface": True},
            {"sequence_conditioning": "masked"},
            {"redesign_motif_sidechains": True},
            {"ligand_selectors": ("L1",)},
        )
        for option in options:
            with (
                self.subTest(option=option),
                self.assertRaisesRegex(ValueError, "require a supplied interface"),
            ):
                initialize_design(
                    self.root / "central.yaml",
                    input_path=self.structure,
                    symmetry="C3",
                    name=None,
                    profile="local",
                    run_root=self.root / "runs",
                    motif_selector="A1-10",
                    **option,
                )

    def test_compact_design_retains_nondefault_sampling_preferences_and_pose(
        self,
    ) -> None:
        path = initialize_design(
            self.root / "nondefault.yaml",
            input_path=self.structure,
            symmetry="C3",
            name=None,
            profile="custom-cluster",
            run_root=self.root / "runs",
            side_a="A1-10",
            side_b="B1-20",
            component_motion="free",
            packing="loose",
            cavity="open",
            diversity="high",
            interface_area="large",
            timesteps=50,
            seed=123,
            designs=50,
            linker_minimum=90,
            linker_maximum=120,
            pose_radius_minimum=20.0,
            pose_radius_maximum=32.0,
            pose_orientation="principal_axis_cone",
            pose_maximum_tilt_deg=15.0,
            pose_seed=7,
        )
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["preferences"],
            {
                "packing": "loose",
                "cavity": "open",
                "diversity": "high",
                "interface_area": "large",
                "component_motion": "free",
            },
        )
        self.assertEqual(
            payload["sampling"],
            {
                "timesteps": 50,
                "seed": 123,
                "designs": 50,
                "initial_pose": {
                    "radius": {"minimum": 20.0, "maximum": 32.0},
                    "axial_offset": {"minimum": 0.0, "maximum": 0.0},
                    "orientation": {
                        "method": "principal_axis_cone",
                        "maximum_tilt_deg": 15.0,
                    },
                    "seed": 7,
                },
            },
        )
        design = load_user_design(path)
        self.assertEqual(design.fixed_arrangement.value, "optimize_components")
        self.assertEqual(design.resources.profile, "custom-cluster")
        self.assertEqual(design.generation[0].length.minimum, 90)
        self.assertEqual(design.generation[0].length.maximum, 120)

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
        self.assertIsNone(design.sampling.replicates_per_pose)
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
        self.assertEqual(
            design.output.root, (self.root / "runs" / "rfd3-mosaic").resolve()
        )

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
        self.assertEqual(
            load_user_design(destination).task.value, "create_symmetric_interface"
        )

    def test_cli_init_infers_supplied_interface_from_both_sides(self) -> None:
        destination = self.root / "interface-cli.yaml"
        with redirect_stdout(StringIO()):
            main(
                [
                    "init",
                    str(destination),
                    "--input",
                    str(self.structure),
                    "--side-a",
                    "A12-20",
                    "--side-b",
                    "B30-40",
                    "--component-motion",
                    "free",
                    "--designs",
                    "1000",
                ]
            )
        design = load_user_design(destination)
        self.assertEqual(design.task.value, "preserve_supplied_geometry")
        self.assertEqual(design.preferences.component_motion.value, "free")
        self.assertEqual(design.sampling.designs, 1000)
        self.assertEqual(design.generation[0].orbit_offset, "nearest_adjacent")

    def test_full_help_keeps_advanced_and_compatibility_commands_available(
        self,
    ) -> None:
        for arguments, documented in (
            (["--help-all"], "prepare-scaffold"),
            (["init", "--help-all"], "--pose-orientation"),
        ):
            with self.subTest(arguments=arguments):
                output = StringIO()
                with (
                    redirect_stdout(output),
                    self.assertRaises(SystemExit) as exit_context,
                ):
                    _parser().parse_args(arguments)
                self.assertEqual(exit_context.exception.code, 0)
                self.assertIn(documented, output.getvalue())
        # Progressive help changes discovery, not previously accepted input.
        legacy = _parser().parse_args(
            [
                "central",
                "--input",
                "input.pdb",
                "--motif",
                "A1-10",
                "--output",
                "runs",
            ]
        )
        self.assertEqual(legacy.command, "central")
        expert = _parser().parse_args(
            [
                "prepare-scaffold",
                "design.yaml",
                "--blueprint",
                "scaffold.yaml",
                "--output-dir",
                "prepared",
            ]
        )
        self.assertEqual(expert.command, "prepare-scaffold")


if __name__ == "__main__":
    unittest.main()
