from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from rfd3_mosaic.constraint_plan import (
    ConstraintStage,
    compile_constraint_plan,
)
from rfd3_mosaic.cli import _write_public_experiment, main
from rfd3_mosaic.schema import UserDesignSpec, load_user_design


def design(**updates: object) -> UserDesignSpec:
    payload: dict[str, object] = {
        "name": "cage-design",
        "input": "motif.pdb",
        "symmetry": "C5",
    }
    payload.update(updates)
    return UserDesignSpec.model_validate(payload)


class UserDesignConstraintTestCase(unittest.TestCase):
    def test_unconstrained_design_has_empty_plan(self) -> None:
        plan = compile_constraint_plan(design())

        self.assertEqual(plan.operators, ())
        self.assertEqual(plan.required_operator_kinds, ())

    def test_fixed_constraints_preserve_explicit_coupling_group(self) -> None:
        plan = compile_constraint_plan(
            design(
                constraints=[
                    {
                        "kind": "fixed_xyz",
                        "selector": "A12-20",
                        "coupling_group": "active_site",
                    },
                    {
                        "kind": "fixed_xyz",
                        "selector": "B4-18",
                        "coupling_group": "active_site",
                    },
                    {
                        "kind": "fixed_xyz",
                        "selector": "C2-8",
                    },
                ]
            )
        )

        self.assertEqual(
            tuple(item.coupling_group for item in plan.operators),
            ("active_site", "active_site", None),
        )

    def test_fixed_component_pose_is_user_controlled(self) -> None:
        plan = compile_constraint_plan(
            design(
                constraints=[
                    {
                        "kind": "fixed_xyz",
                        "selector": "A12-20",
                        "coupling_group": "mobile_site",
                        "pose": {
                            "mode": "bounded_mobile",
                            "proposal": "scaffold_objectives",
                            "max_translation": 4.0,
                            "max_rotation_deg": 12.0,
                        },
                    }
                ]
            )
        )

        pose = plan.operators[0].parameters["pose"]
        self.assertEqual(pose["mode"], "bounded_mobile")
        self.assertEqual(pose["proposal"], "scaffold_objectives")
        self.assertEqual(pose["max_translation"], 4.0)
        self.assertEqual(pose["max_rotation_deg"], 12.0)

    def test_omitted_task_does_not_change_fixed_xyz_pose(self) -> None:
        declared = design(
            constraints=[{"kind": "fixed_xyz", "selector": "A12-20"}]
        )

        plan = compile_constraint_plan(declared)

        self.assertIsNone(declared.task)
        self.assertEqual(
            plan.operators[0].parameters["pose"]["mode"],
            "fixed",
        )

    def test_create_interface_task_derives_safe_orbit_pose_only_when_explicit(
        self,
    ) -> None:
        plan = compile_constraint_plan(
            design(
                task="create_symmetric_interface",
                fixed_arrangement="optimize_components",
                symmetry="C3",
                generation=[
                    {
                        "kind": "terminal",
                        "anchor": "A12-20",
                        "terminus": "n",
                        "length": 35,
                    }
                ],
                constraints=[
                    {"kind": "fixed_xyz", "selector": "A12-20"}
                ],
            )
        )

        operator = plan.operators[0]
        pose = operator.parameters["pose"]
        self.assertEqual(operator.operator, "fixed_xyz")
        self.assertEqual(operator.stage, ConstraintStage.HARD_PROJECTOR)
        self.assertEqual(pose["mode"], "bounded_mobile")
        self.assertEqual(pose["subspace"], "radial_axial_rotation")
        self.assertEqual(pose["proposal"], "scaffold_objectives")
        self.assertEqual(pose["max_translation"], 4.0)
        self.assertEqual(pose["max_rotation_deg"], 10.0)

    def test_create_interface_task_defaults_to_locked_fixed_arrangement(
        self,
    ) -> None:
        declared = design(
            task="create_symmetric_interface",
            symmetry="C3",
            generation=[
                {
                    "kind": "terminal",
                    "anchor": "A12-20",
                    "terminus": "n",
                    "length": 35,
                }
            ],
            constraints=[{"kind": "fixed_xyz", "selector": "A12-20"}],
        )

        plan = compile_constraint_plan(declared)

        self.assertEqual(declared.fixed_arrangement.value, "locked")
        self.assertEqual(plan.operators[0].parameters["pose"]["mode"], "fixed")

    def test_locked_create_interface_is_topology_neutral(self) -> None:
        declared = design(
            task="create_symmetric_interface",
            symmetry="T",
            generation=[
                {
                    "kind": "terminal",
                    "anchor": "A12-20",
                    "terminus": "n",
                    "length": 35,
                }
            ],
            constraints=[{"kind": "fixed_xyz", "selector": "A12-20"}],
        )
        self.assertEqual(declared.fixed_arrangement.value, "locked")

    def test_polyhedral_component_optimization_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "optimize_components currently supports Cn and Dn",
        ):
            design(
                task="create_symmetric_interface",
                fixed_arrangement="optimize_components",
                symmetry="T",
                generation=[
                    {
                        "kind": "terminal",
                        "anchor": "A12-20",
                        "terminus": "n",
                        "length": 35,
                    }
                ],
                constraints=[
                    {"kind": "fixed_xyz", "selector": "A12-20"}
                ],
            )

    def test_preserve_task_rejects_custom_mobile_pose(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "owns the ordinary motif-orbit pose contract",
        ):
            design(
                task="preserve_supplied_geometry",
                generation=[
                    {
                        "kind": "between",
                        "from_selector": "A1-2",
                        "to_selector": "B1-2",
                        "length": 30,
                    }
                ],
                constraints=[
                    {
                        "kind": "fixed_xyz",
                        "selector": "A1-2,B1-2",
                        "pose": {
                            "mode": "bounded_mobile",
                            "max_translation": 3.0,
                            "max_rotation_deg": 10.0,
                        },
                    }
                ],
            )

    def test_create_interface_task_rejects_between_generation(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "requires terminal generation",
        ):
            design(
                task="create_symmetric_interface",
                symmetry="C3",
                generation=[
                    {
                        "kind": "between",
                        "from_selector": "A1-2",
                        "to_selector": "B1-2",
                        "length": 30,
                    }
                ],
                constraints=[
                    {"kind": "fixed_xyz", "selector": "A1-2,B1-2"}
                ],
            )

    def test_fixed_component_rejects_scaffold_proposal(self) -> None:
        with self.assertRaises(ValidationError):
            design(
                constraints=[
                    {
                        "kind": "fixed_xyz",
                        "selector": "A12-20",
                        "pose": {
                            "mode": "fixed",
                            "proposal": "scaffold_objectives",
                        },
                    }
                ]
            )

    def test_bounded_fixed_component_requires_translation_and_rotation(self) -> None:
        with self.assertRaises(ValidationError):
            design(
                constraints=[
                    {
                        "kind": "fixed_xyz",
                        "selector": "A12-20",
                        "pose": {
                            "mode": "bounded_mobile",
                            "max_translation": 4.0,
                        },
                    }
                ]
            )

    def test_radial_component_uses_translation_only_scaffold_signal(
        self,
    ) -> None:
        plan = compile_constraint_plan(
            design(
                constraints=[
                    {
                        "kind": "fixed_xyz",
                        "selector": "A12-20",
                        "pose": {
                            "mode": "bounded_mobile",
                            "subspace": "radial",
                            "proposal": "scaffold_objectives",
                            "max_translation": 4.0,
                        },
                    }
                ]
            )
        )

        pose = plan.operators[0].parameters["pose"]
        self.assertEqual(pose["subspace"], "radial")
        self.assertIsNone(pose["max_rotation_deg"])

    def test_radial_component_rejects_rotation_or_denoiser_signal(
        self,
    ) -> None:
        for pose in (
            {
                "mode": "bounded_mobile",
                "subspace": "radial",
                "proposal": "scaffold_objectives",
                "max_translation": 4.0,
                "max_rotation_deg": 5.0,
            },
            {
                "mode": "bounded_mobile",
                "subspace": "radial",
                "proposal": "denoiser_fit",
                "max_translation": 4.0,
            },
        ):
            with self.subTest(pose=pose), self.assertRaises(ValidationError):
                design(
                    constraints=[
                        {
                            "kind": "fixed_xyz",
                            "selector": "A12-20",
                            "pose": pose,
                        }
                    ]
                )

    def test_radial_rotation_component_keeps_rotation_enabled(self) -> None:
        plan = compile_constraint_plan(
            design(
                constraints=[
                    {
                        "kind": "fixed_xyz",
                        "selector": "A12-20",
                        "pose": {
                            "mode": "bounded_mobile",
                            "subspace": "radial_rotation",
                            "proposal": "scaffold_objectives",
                            "max_translation": 4.0,
                            "max_rotation_deg": 12.0,
                        },
                    }
                ]
            )
        )

        pose = plan.operators[0].parameters["pose"]
        self.assertEqual(pose["subspace"], "radial_rotation")
        self.assertEqual(pose["max_translation"], 4.0)
        self.assertEqual(pose["max_rotation_deg"], 12.0)

    def test_compatibility_constraint_names_compile_canonically(self) -> None:
        plan = compile_constraint_plan(
            design(
                constraints=[
                    {
                        "kind": "full_xyz_fixed",
                        "selector": "A12-20",
                    },
                    {
                        "kind": "ca_cylindrical_fixed",
                        "selector": "A26-37",
                        "keep": ["radius", "azimuth"],
                    },
                    {
                        "kind": "bounded_mobile_interface",
                        "selector": "B4-18",
                        "radial": {"minimum": 60.0, "maximum": 90.0},
                    },
                ]
            )
        )

        self.assertEqual(
            tuple(item.operator for item in plan.operators),
            ("fixed_xyz", "cylindrical", "bounded_mobile"),
        )
        self.assertEqual(
            tuple(item.id for item in plan.operators),
            ("constraint_001", "constraint_002", "constraint_003"),
        )
        self.assertEqual(
            plan.operators[-1].stage,
            ConstraintStage.BOUNDED_PROJECTOR,
        )

    def test_cylindrical_constraint_preserves_only_declared_dofs(self) -> None:
        plan = compile_constraint_plan(
            design(
                constraints=[
                    {
                        "kind": "cylindrical",
                        "selector": "A12-20",
                        "keep": ["radius", "azimuth"],
                    }
                ]
            )
        )

        operator = plan.operators[0]
        self.assertEqual(operator.atoms, "ca")
        self.assertEqual(operator.reference_frame, "symmetry_axis")
        self.assertEqual(operator.controlled_dofs, ("radius", "azimuth"))

    def test_cylindrical_constraint_rejects_duplicate_dofs(self) -> None:
        with self.assertRaises(ValidationError):
            design(
                constraints=[
                    {
                        "kind": "cylindrical",
                        "selector": "A12-20",
                        "keep": ["radius", "radius"],
                    }
                ]
            )

    def test_bounded_mobile_requires_an_explicit_bound(self) -> None:
        with self.assertRaises(ValidationError):
            design(
                constraints=[
                    {
                        "kind": "bounded_mobile",
                        "selector": "A12-20",
                    }
                ]
            )

    def test_bounded_mobile_rejects_reversed_range(self) -> None:
        with self.assertRaises(ValidationError):
            design(
                constraints=[
                    {
                        "kind": "bounded_mobile",
                        "selector": "A12-20",
                        "tilt_deg": {"minimum": 20.0, "maximum": 5.0},
                    }
                ]
            )

    def test_fixed_xyz_conflicts_with_pose_constraint_on_same_selection(self) -> None:
        declared = design(
            constraints=[
                {"kind": "fixed_xyz", "selector": "A12-20"},
                {
                    "kind": "cylindrical",
                    "selector": "A12-20",
                    "keep": ["radius"],
                },
            ]
        )

        with self.assertRaisesRegex(ValueError, "Conflicting constraints"):
            compile_constraint_plan(declared)

    def test_disjoint_dofs_on_same_selection_can_be_composed(self) -> None:
        plan = compile_constraint_plan(
            design(
                constraints=[
                    {
                        "kind": "cylindrical",
                        "selector": "A12-20",
                        "keep": ["radius"],
                    },
                    {
                        "kind": "bounded_mobile",
                        "selector": "A12-20",
                        "tilt_deg": {"minimum": 0.0, "maximum": 15.0},
                    },
                ]
            )
        )

        self.assertEqual(len(plan.operators), 2)

    def test_backend_capability_check_is_fail_closed(self) -> None:
        plan = compile_constraint_plan(
            design(
                constraints=[
                    {"kind": "fixed_xyz", "selector": "A12-20"},
                    {
                        "kind": "cylindrical",
                        "selector": "A26-37",
                        "keep": ["radius"],
                    },
                ]
            )
        )

        with self.assertRaisesRegex(ValueError, "cylindrical"):
            plan.require_backend_support({"fixed_xyz"})
        plan.require_backend_support({"fixed_xyz", "cylindrical"})

    def test_generation_is_topology_neutral(self) -> None:
        declared = design(
            generation=[
                {
                    "kind": "terminal",
                    "anchor": "A12-37",
                    "terminus": "n",
                    "length": 35,
                },
                {
                    "kind": "between",
                    "from_selector": "A12-37",
                    "to_selector": "B4-18",
                    "length": {"minimum": 80, "maximum": 100},
                },
            ]
        )

        self.assertEqual(tuple(item.kind for item in declared.generation), (
            "terminal",
            "between",
        ))

    def test_invalid_symmetry_name_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            design(symmetry="C0")

    def test_loader_resolves_relative_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "motif.pdb").write_text("END\n", encoding="utf-8")
            (root / "design.yaml").write_text(
                """\
schema_version: 1
name: portable-design
input: motif.pdb
symmetry: D3
constraints:
  - kind: fixed_xyz
    selector: A1-10
""",
                encoding="utf-8",
            )

            loaded = load_user_design(root / "design.yaml")

        self.assertEqual(loaded.input, (root / "motif.pdb").resolve())

    def test_plan_cli_reads_public_design_without_legacy_topology(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "motif.pdb").write_text(
                "ATOM      1   CA ALA A   1       0.000   0.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            config = root / "design.yaml"
            config.write_text(
                """\
schema_version: 1
name: public-plan
input: motif.pdb
symmetry: C7
constraints:
  - kind: cylindrical
    selector: A1
    keep: [radius, azimuth]
""",
                encoding="utf-8",
            )
            output = StringIO()

            with redirect_stdout(output):
                main(["plan", str(config)])

        text = output.getvalue()
        self.assertIn("RFD3-Mosaic public design plan", text)
        self.assertIn("user mode:  simple", text)
        self.assertIn("cylindrical [hard_projector]", text)
        self.assertIn("assembly lowering: blocked", text)

    def test_plan_cli_explains_create_interface_task_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "motif.pdb").write_text(
                "ATOM      1   CA ALA A   1       0.000   0.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            config = root / "design.yaml"
            config.write_text(
                """\
schema_version: 1
name: create-interface-plan
input: motif.pdb
symmetry: C3
task: create_symmetric_interface
fixed_arrangement: optimize_components
generation:
  - kind: terminal
    anchor: A1
    terminus: c
    length: 20
constraints:
  - kind: fixed_xyz
    selector: A1
""",
                encoding="utf-8",
            )
            output = StringIO()

            with redirect_stdout(output):
                main(["plan", str(config)])

        text = output.getvalue()
        self.assertIn("task:       create symmetric interface", text)
        self.assertIn("fixed arrangement=optimize_components", text)
        self.assertIn("exact rigid components", text)
        self.assertIn("bounded joint pose/packing optimization", text)
        self.assertIn("pose=bounded_mobile", text)
        self.assertIn("assembly lowering: ready", text)

    def test_plan_cli_explains_locked_generated_only_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "motif.pdb").write_text(
                "ATOM      1   CA ALA A   1       0.000   0.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            config = root / "design.yaml"
            config.write_text(
                """\
schema_version: 1
name: locked-create-interface-plan
input: motif.pdb
symmetry: C3
task: create_symmetric_interface
generation:
  - kind: terminal
    anchor: A1
    terminus: c
    length: 20
constraints:
  - kind: fixed_xyz
    selector: A1
""",
                encoding="utf-8",
            )
            output = StringIO()

            with redirect_stdout(output):
                main(["plan", str(config)])

        text = output.getvalue()
        self.assertIn("fixed arrangement=locked", text)
        self.assertIn("complete fixed arrangement", text)
        self.assertIn("generated-only packing guidance", text)
        self.assertIn("pose=fixed", text)

    def test_plan_cli_explains_joint_seed_copies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "motif.pdb").write_text(
                "ATOM      1   CA ALA A   1       0.000   0.000   0.000"
                "  1.00 20.00           C\n"
                "ATOM      2   CA ALA A   2       3.800   0.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            config = root / "design.yaml"
            config.write_text(
                """\
schema_version: 1
name: complete-seed-plan
input: motif.pdb
symmetry: C3
constraints:
  - kind: fixed_xyz
    selector: A1
    coupling_group: complete_interface_seed
  - kind: fixed_xyz
    selector: A2
    coupling_group: complete_interface_seed
""",
                encoding="utf-8",
            )
            output = StringIO()

            with redirect_stdout(output):
                main(["plan", str(config)])

        text = output.getvalue()
        self.assertIn("complete_interface_seed", text)
        self.assertIn(
            "2 selected region(s) and 2 atom(s) per copy",
            text,
        )
        self.assertIn("x 3 symmetry copies", text)
        self.assertIn("selectors: A1, A2", text)

    def test_plan_reports_physical_quotient_copy_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "motif.pdb").write_text(
                "ATOM      1   CA ALA A   1       1.000   0.000   0.000"
                "  1.00 20.00           C\n"
                "ATOM      2   CA ALA A   2      -1.000   0.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            config = root / "quotient.yaml"
            config.write_text(
                """\
schema_version: 1
name: quotient-plan
input: motif.pdb
symmetry: C4
finite_orbit_action:
  coset_representative_ids: [C4:e, C4:r1]
  stabilizer_transform_ids: [C4:e, C4:r2]
  transform_to_coset_representative:
    C4:e: C4:e
    C4:r1: C4:r1
    C4:r2: C4:e
    C4:r3: C4:r1
constraints:
  - kind: fixed_xyz
    selector: A1-2
    coupling_group: c2_seed
""",
                encoding="utf-8",
            )
            output = StringIO()

            with redirect_stdout(output):
                main(["plan", str(config)])

        text = output.getvalue()
        self.assertIn("x 2 symmetry copies", text)
        self.assertNotIn("x 4 symmetry copies", text)

    def test_submit_cli_rejects_unlowered_public_design(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "motif.pdb").write_text("END\n", encoding="utf-8")
            config = root / "design.yaml"
            config.write_text(
                """\
schema_version: 1
name: public-submit
input: motif.pdb
symmetry: C3
""",
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit):
                main(["submit", str(config), "--dry-run"])

    def test_validate_preflights_complete_expanded_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "motif.pdb").write_text(
                "ATOM      1   CA ALA A   1       0.000   0.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            config = root / "design.yaml"
            config.write_text(
                """\
schema_version: 1
name: geometry-preflight
input: motif.pdb
symmetry: C3
generation:
  - kind: terminal
    anchor: A1
    terminus: n
    length: 20
constraints:
  - kind: fixed_xyz
    selector: A1
sampling:
  initial_pose:
    radius: {minimum: 24.0, maximum: 24.0}
    seed: 3000
""",
                encoding="utf-8",
            )
            output = StringIO()

            with redirect_stdout(output):
                main(["validate", str(config)])

        text = output.getvalue()
        self.assertIn("User design validation: PASSED", text)
        self.assertIn("geometry:    PASSED", text)
        self.assertIn("RFD3 input:  PASSED", text)
        self.assertIn("3 atoms", text)

    def test_validate_automatically_positions_degenerate_simple_motif(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "motif.pdb").write_text(
                "ATOM      1   CA ALA A   1       0.000   0.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            config = root / "design.yaml"
            config.write_text(
                """\
schema_version: 1
name: clashing-preflight
input: motif.pdb
symmetry: C3
generation:
  - kind: terminal
    anchor: A1
    terminus: n
    length: 20
constraints:
  - kind: fixed_xyz
    selector: A1
""",
                encoding="utf-8",
            )

            output = StringIO()
            with redirect_stdout(output):
                main(["validate", str(config)])

        self.assertIn("User design validation: PASSED", output.getvalue())

    def test_executable_public_design_materializes_internal_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            structure = root / "motif.pdb"
            structure.write_text(
                "ATOM      1   CA ALA A   1       0.000   0.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            source = root / "design.yaml"
            source.write_text("placeholder\n", encoding="utf-8")
            declared = UserDesignSpec.model_validate(
                {
                    "name": "public-fixed-run",
                    "input": str(structure),
                    "symmetry": "C3",
                    "generation": [
                        {
                            "kind": "terminal",
                            "anchor": "A1",
                            "terminus": "n",
                            "length": 20,
                        }
                    ],
                    "constraints": [
                        {"kind": "fixed_xyz", "selector": "A1"},
                    ],
                    "sampling": {
                        "timesteps": 50,
                        "seed": 7,
                        "initial_pose": {
                            "radius": {
                                "minimum": 24.0,
                                "maximum": 24.0,
                            },
                            "seed": 3000,
                        },
                    },
                    "resources": {"profile": "h100"},
                    "output": {
                        "root": str(root / "runs"),
                        "campaign": "public-tests",
                    },
                }
            )

            envelope = _write_public_experiment(declared, source)
            payload = yaml.safe_load(
                envelope.read_text(encoding="utf-8")
            )

        self.assertEqual(payload["topology"]["kind"], "user_design")
        self.assertEqual(payload["topology"]["config"], str(source))
        self.assertEqual(payload["sampling"]["timesteps"], 50)
        self.assertNotIn("initial_pose", payload["sampling"])
        self.assertEqual(payload["resources"]["profile"], "h100")


if __name__ == "__main__":
    unittest.main()
