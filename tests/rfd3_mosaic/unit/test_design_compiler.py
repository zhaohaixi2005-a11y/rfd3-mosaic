import tempfile
import unittest
from pathlib import Path

import numpy as np

from rfd3_mosaic.compile import (
    build_master_group_transforms,
    expand_symmetry_instances,
)
from rfd3_mosaic.constraint_plan import compile_constraint_plan
from rfd3_mosaic.design_compiler import (
    bind_constraint_plan,
    lower_user_design,
    parse_public_selector,
)
from rfd3_mosaic.geometry import build_transform_registry
from rfd3_mosaic.schema import AssemblySpecification, UserDesignSpec
from rfd3_mosaic.topology.component_incidence import (
    enumerate_binary_interface_incidence_plans,
)
from rfd3_mosaic.topology.symmetry_connectivity import finite_symmetry_spec


def _atom_line(
    serial: int,
    atom: str,
    chain: str,
    residue: int,
    x: float,
) -> str:
    element = atom[0]
    return (
        f"ATOM  {serial:5d} {atom:>4s} ALA {chain:1s}{residue:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
        f"          {element:>2s}\n"
    )


def _write_structure(path: Path) -> None:
    lines: list[str] = []
    serial = 1
    for chain_index, chain in enumerate(("A", "B")):
        for residue in range(1, 5):
            for atom_index, atom in enumerate(("N", "CA", "C", "O")):
                lines.append(
                    _atom_line(
                        serial,
                        atom,
                        chain,
                        residue,
                        chain_index * 20.0 + residue * 4.0 + atom_index,
                    )
                )
                serial += 1
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


class DesignCompilerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.structure = self.root / "motif.pdb"
        _write_structure(self.structure)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _design(self, **updates: object) -> UserDesignSpec:
        payload: dict[str, object] = {
            "name": "compiled-design",
            "input": str(self.structure),
            "symmetry": "C3",
        }
        payload.update(updates)
        return UserDesignSpec.model_validate(payload)

    def test_public_selector_accepts_compact_and_assembly_syntax(self) -> None:
        compact = parse_public_selector("A1-2,B3-4")
        assembly = parse_public_selector("A/1-2/*,B/3-4/*")

        self.assertEqual(compact, assembly)
        self.assertEqual(compact[0].assembly_expression, "A/1-2/*")

    def test_binding_detects_partial_selector_atom_overlap(self) -> None:
        declared = self._design(
            constraints=[
                {"kind": "fixed_xyz", "selector": "A1-2"},
                {
                    "kind": "cylindrical",
                    "selector": "A2-3",
                    "atoms": "ca",
                    "keep": ["radius"],
                },
            ]
        )
        plan = compile_constraint_plan(declared)

        with self.assertRaisesRegex(ValueError, "overlap on resolved atoms"):
            bind_constraint_plan(declared, plan)

    def test_binding_rejects_missing_residue_in_selector(self) -> None:
        declared = self._design(
            constraints=[
                {"kind": "fixed_xyz", "selector": "A3-5"},
            ]
        )

        with self.assertRaisesRegex(ValueError, "missing residues"):
            bind_constraint_plan(declared)

    def test_lowers_bidirectional_terminal_generation(self) -> None:
        lowered = lower_user_design(
            self._design(
                generation=[
                    {
                        "kind": "terminal",
                        "anchor": "A1-2",
                        "terminus": "n",
                        "length": 20,
                    },
                    {
                        "kind": "terminal",
                        "anchor": "A1-2",
                        "terminus": "c",
                        "length": 25,
                    },
                ],
                constraints=[
                    {"kind": "fixed_xyz", "selector": "A1-2"},
                ],
            )
        )

        spec = lowered.specification
        self.assertEqual(len(spec.fragments), 1)
        self.assertEqual(len(spec.generated_segments), 2)
        self.assertEqual(len(spec.interfaces), 1)
        automatic = spec.interfaces["auto_generated_interface_001"]
        self.assertEqual(automatic.satisfaction_stage, "output")
        self.assertEqual(automatic.copy_relation.transform, "C3:r1")
        self.assertEqual(automatic.target_geometry.coverage.mode, "auto")
        self.assertEqual(automatic.target_geometry.contacts.cutoff, 4.5)
        self.assertEqual(spec.fragments["motif_001"].fixed_atoms, "all")
        self.assertEqual(
            spec.symmetry.transform_sets["declared"].order,
            3,
        )

    def test_preserves_fixed_fragment_not_used_as_linker_endpoint(
        self,
    ) -> None:
        lowered = lower_user_design(
            self._design(
                generation=[
                    {
                        "kind": "terminal",
                        "anchor": "A1-2",
                        "terminus": "c",
                        "length": 20,
                    }
                ],
                constraints=[
                    {"kind": "fixed_xyz", "selector": "A1-2"},
                    {"kind": "fixed_xyz", "selector": "B1-2"},
                ],
            )
        )

        self.assertEqual(len(lowered.specification.fragments), 2)
        self.assertEqual(
            {
                fragment.selection
                for fragment in lowered.specification.fragments.values()
            },
            {"A/1-2/*", "B/1-2/*"},
        )
        self.assertEqual(len(lowered.specification.generated_segments), 1)
        instances = expand_symmetry_instances(lowered.specification)
        self.assertEqual(len(instances.fragments), 6)

    def test_lowers_assembly_size_intent_to_required_static_objectives(
        self,
    ) -> None:
        lowered = lower_user_design(
            self._design(
                assembly_shape={
                    "diameter_angstrom": {
                        "minimum": 40.0,
                        "maximum": 80.0,
                    },
                    "cavity_diameter_angstrom": {
                        "minimum": 10.0,
                        "maximum": 30.0,
                    },
                },
                generation=[
                    {
                        "kind": "terminal",
                        "anchor": "A1-2",
                        "terminus": "c",
                        "length": 20,
                    }
                ],
                constraints=[
                    {"kind": "fixed_xyz", "selector": "A1-2"},
                ],
            )
        )

        objectives = lowered.specification.objectives
        outer = objectives["assembly_outer_diameter"]
        cavity = objectives["assembly_cavity_diameter"]
        self.assertEqual(outer.metric, "assemblies.outer_diameter")
        self.assertEqual(outer.mode.value, "range")
        self.assertEqual((outer.minimum, outer.maximum), (40.0, 80.0))
        self.assertTrue(outer.required)
        self.assertEqual(
            cavity.metric,
            "cavities.minimum_central_void_diameter",
        )
        self.assertEqual(
            (cavity.minimum, cavity.maximum),
            (10.0, 30.0),
        )
        self.assertTrue(cavity.required)

    def test_optimized_task_automatically_moves_axis_degenerate_motif(
        self,
    ) -> None:
        structure = self.root / "axis_motif.pdb"
        structure.write_text(
            _atom_line(1, "CA", "A", 1, 0.0) + "END\n",
            encoding="utf-8",
        )
        design = UserDesignSpec.model_validate(
            {
                "name": "automatic-simple-pose",
                "input": str(structure),
                "symmetry": "C3",
                "task": "create_symmetric_interface",
                "fixed_arrangement": "optimize_components",
                "generation": [
                    {
                        "kind": "terminal",
                        "anchor": "A1",
                        "terminus": "c",
                        "length": 20,
                    }
                ],
                "constraints": [
                    {"kind": "fixed_xyz", "selector": "A1"},
                ],
            }
        )

        lowered = lower_user_design(design)
        initialization = lowered.specification.initialization[
            "fixed_component_001"
        ]

        self.assertEqual(design.user_mode, "simple")
        self.assertGreater(initialization.placement.radius.mean, 0.0)
        self.assertEqual(
            lowered.specification.interfaces[
                "auto_generated_interface_001"
            ].copy_relation.transform,
            "C3:r1",
        )

    def test_locked_task_freezes_automatically_resolved_initial_orbit(
        self,
    ) -> None:
        structure = self.root / "locked_axis_motif.pdb"
        structure.write_text(
            _atom_line(1, "CA", "A", 1, 0.0) + "END\n",
            encoding="utf-8",
        )
        lowered = lower_user_design(
            UserDesignSpec.model_validate(
                {
                    "name": "locked-simple-pose",
                    "input": str(structure),
                    "symmetry": "C3",
                    "task": "create_symmetric_interface",
                    "generation": [
                        {
                            "kind": "terminal",
                            "anchor": "A1",
                            "terminus": "c",
                            "length": 20,
                        }
                    ],
                    "constraints": [
                        {"kind": "fixed_xyz", "selector": "A1"},
                    ],
                }
            )
        )

        initialization = lowered.specification.initialization[
            "fixed_component_001"
        ]
        self.assertGreater(initialization.placement.radius.mean, 0.0)
        self.assertEqual(
            lowered.specification.symmetry.orbits[
                "motif_orbit"
            ].component_mobility,
            {},
        )

    def test_between_generation_does_not_invent_output_interface(self) -> None:
        lowered = lower_user_design(
            self._design(
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
                        "selector": "A1-2",
                        "coupling_group": "supplied_interface",
                    },
                    {
                        "kind": "fixed_xyz",
                        "selector": "B1-2",
                        "coupling_group": "supplied_interface",
                    },
                ],
            )
        )

        self.assertEqual(lowered.specification.interfaces, {})

    def test_preserve_task_keeps_complete_supplied_geometry_orbit_fixed(
        self,
    ) -> None:
        lowered = lower_user_design(
            self._design(
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
                        "selector": "A1-2",
                        "coupling_group": "supplied_interface",
                    },
                    {
                        "kind": "fixed_xyz",
                        "selector": "B1-2",
                        "coupling_group": "supplied_interface",
                    },
                ],
            )
        )

        orbit = lowered.specification.symmetry.orbits["motif_orbit"]
        group = lowered.specification.motion_groups["fixed_component_001"]
        self.assertEqual(group.mode.value, "fixed")
        self.assertEqual(group.members, ["motif_001", "motif_002"])
        self.assertEqual(orbit.component_mobility, {})

    def test_create_interface_task_keeps_internal_geometry_exact_and_moves_orbit(
        self,
    ) -> None:
        lowered = lower_user_design(
            self._design(
                task="create_symmetric_interface",
                fixed_arrangement="optimize_components",
                generation=[
                    {
                        "kind": "terminal",
                        "anchor": "A1-2",
                        "terminus": "n",
                        "length": 20,
                    },
                    {
                        "kind": "terminal",
                        "anchor": "A1-2",
                        "terminus": "c",
                        "length": 20,
                    },
                ],
                constraints=[
                    {"kind": "fixed_xyz", "selector": "A1-2"}
                ],
            )
        )

        operator = lowered.constraint_plan.operators[0]
        orbit = lowered.specification.symmetry.orbits["motif_orbit"]
        group = lowered.specification.motion_groups["fixed_component_001"]
        mobility = orbit.component_mobility["fixed_component_001"]
        self.assertEqual(operator.operator, "fixed_xyz")
        self.assertEqual(operator.controlled_dofs, ("cartesian_xyz",))
        self.assertEqual(group.mode.value, "fixed")
        self.assertEqual(group.members, ["motif_001"])
        self.assertEqual(mobility.mode.value, "orbit_rigid")
        self.assertEqual(
            mobility.effective_subspace.value,
            "radial_axial_rotation",
        )
        self.assertEqual(
            mobility.effective_proposal.value,
            "scaffold_objectives",
        )
        self.assertEqual(mobility.bounds.max_translation, 4.0)
        self.assertEqual(mobility.bounds.max_rotation_deg, 10.0)

    def test_simple_create_interface_requires_explicit_multi_component_graph(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Simple mode found terminal generation on multiple",
        ):
            lower_user_design(
                self._design(
                    task="create_symmetric_interface",
                    generation=[
                        {
                            "kind": "terminal",
                            "anchor": "A1-2",
                            "terminus": "c",
                            "length": 20,
                        },
                        {
                            "kind": "terminal",
                            "anchor": "B1-2",
                            "terminus": "c",
                            "length": 20,
                        },
                    ],
                    constraints=[
                        {"kind": "fixed_xyz", "selector": "A1-2"},
                        {"kind": "fixed_xyz", "selector": "B1-2"},
                    ],
                )
            )

    def test_locked_create_interface_has_guidance_without_mobility(self) -> None:
        lowered = lower_user_design(
            self._design(
                task="create_symmetric_interface",
                generation=[
                    {
                        "kind": "terminal",
                        "anchor": "A1-2",
                        "terminus": "c",
                        "length": 20,
                    }
                ],
                constraints=[{"kind": "fixed_xyz", "selector": "A1-2"}],
            )
        )

        orbit = lowered.specification.symmetry.orbits["motif_orbit"]
        self.assertEqual(orbit.component_mobility, {})
        self.assertEqual(
            lowered.constraint_plan.operators[0].parameters["pose"]["mode"],
            "fixed",
        )
        self.assertEqual(len(lowered.specification.interfaces), 1)

    def test_fixed_coupling_groups_lower_to_motion_components(self) -> None:
        lowered = lower_user_design(
            self._design(
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
                        "selector": "A1-2",
                        "coupling_group": "joint_site",
                    },
                    {
                        "kind": "fixed_xyz",
                        "selector": "B1-2",
                        "coupling_group": "joint_site",
                    },
                ],
            )
        )

        groups = lowered.specification.motion_groups
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            next(iter(groups.values())).members,
            ["motif_001", "motif_002"],
        )

    def test_independent_fixed_declarations_lower_separately(self) -> None:
        lowered = lower_user_design(
            self._design(
                generation=[
                    {
                        "kind": "between",
                        "from_selector": "A1-2",
                        "to_selector": "B1-2",
                        "length": 30,
                    }
                ],
                constraints=[
                    {"kind": "fixed_xyz", "selector": "A1-2"},
                    {"kind": "fixed_xyz", "selector": "B1-2"},
                ],
            )
        )

        groups = lowered.specification.motion_groups
        self.assertEqual(len(groups), 2)
        self.assertEqual(
            sorted(tuple(group.members) for group in groups.values()),
            [("motif_001",), ("motif_002",)],
        )
        self.assertEqual(
            len(
                lowered.specification.symmetry.orbits[
                    "motif_orbit"
                ].master_groups
            ),
            2,
        )

    def test_component_pose_lowers_to_per_group_orbit_mobility(self) -> None:
        lowered = lower_user_design(
            self._design(
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
                        "selector": "A1-2",
                        "coupling_group": "mobile_component",
                        "pose": {
                            "mode": "bounded_mobile",
                            "proposal": "scaffold_objectives",
                            "max_translation": 3.0,
                            "max_rotation_deg": 10.0,
                        },
                    },
                    {
                        "kind": "fixed_xyz",
                        "selector": "B1-2",
                        "coupling_group": "fixed_component",
                    },
                ],
            )
        )

        orbit = lowered.specification.symmetry.orbits["motif_orbit"]
        self.assertEqual(len(orbit.component_mobility), 1)
        mobility = orbit.component_mobility["fixed_component_001"]
        self.assertEqual(mobility.mode.value, "orbit_rigid")
        self.assertEqual(
            mobility.effective_proposal.value,
            "scaffold_objectives",
        )
        self.assertEqual(mobility.bounds.max_translation, 3.0)
        self.assertEqual(mobility.bounds.max_rotation_deg, 10.0)
        self.assertEqual(
            orbit.component_mobility.get("fixed_component_002"), None
        )

    def test_two_independent_mobile_interfaces_lower_to_two_components(
        self,
    ) -> None:
        pose = {
            "mode": "bounded_mobile",
            "proposal": "scaffold_objectives",
            "max_translation": 3.0,
            "max_rotation_deg": 10.0,
        }
        lowered = lower_user_design(
            self._design(
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
                        "selector": "A1-2",
                        "coupling_group": "interface_alpha",
                        "pose": pose,
                    },
                    {
                        "kind": "fixed_xyz",
                        "selector": "B1-2",
                        "coupling_group": "interface_beta",
                        "pose": pose,
                    },
                ],
            )
        )

        orbit = lowered.specification.symmetry.orbits["motif_orbit"]
        self.assertEqual(len(orbit.master_groups), 2)
        self.assertEqual(
            set(orbit.component_mobility),
            {"fixed_component_001", "fixed_component_002"},
        )
        self.assertTrue(
            all(
                mobility.mode.value == "orbit_rigid"
                and mobility.effective_proposal.value
                == "scaffold_objectives"
                for mobility in orbit.component_mobility.values()
            )
        )

    def test_radial_component_lowers_axis_aware_mobility(self) -> None:
        lowered = lower_user_design(
            self._design(
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
                        "selector": "A1-2",
                        "coupling_group": "mobile_component",
                        "pose": {
                            "mode": "bounded_mobile",
                            "subspace": "radial",
                            "proposal": "scaffold_objectives",
                            "max_translation": 3.0,
                        },
                    },
                    {
                        "kind": "fixed_xyz",
                        "selector": "B1-2",
                        "coupling_group": "fixed_component",
                    },
                ],
            )
        )

        orbit = lowered.specification.symmetry.orbits["motif_orbit"]
        mobility = orbit.component_mobility["fixed_component_001"]
        self.assertEqual(mobility.effective_subspace.value, "radial")
        self.assertEqual(
            mobility.effective_proposal.value,
            "scaffold_objectives",
        )
        self.assertEqual(mobility.bounds.max_translation, 3.0)
        self.assertIsNone(mobility.bounds.max_rotation_deg)

    def test_tilt_only_component_lowers_rotation_only_mobility(self) -> None:
        lowered = lower_user_design(
            self._design(
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
                        "selector": "A1-2",
                        "coupling_group": "tilting_component",
                        "pose": {
                            "mode": "bounded_mobile",
                            "subspace": "tilt_only",
                            "proposal": "scaffold_objectives",
                            "max_rotation_deg": 12.0,
                        },
                    },
                    {
                        "kind": "fixed_xyz",
                        "selector": "B1-2",
                        "coupling_group": "fixed_component",
                    },
                ],
            )
        )

        orbit = lowered.specification.symmetry.orbits["motif_orbit"]
        mobility = orbit.component_mobility["fixed_component_001"]
        self.assertEqual(mobility.effective_subspace.value, "tilt_only")
        self.assertEqual(
            mobility.effective_proposal.value,
            "scaffold_objectives",
        )
        self.assertIsNone(mobility.bounds.max_translation)
        self.assertEqual(mobility.bounds.max_rotation_deg, 12.0)

    def test_joint_mobile_interface_fragments_lower_as_one_orbit_component(
        self,
    ) -> None:
        pose = {
            "mode": "bounded_mobile",
            "proposal": "scaffold_objectives",
            "max_translation": 3.0,
            "max_rotation_deg": 10.0,
        }
        lowered = lower_user_design(
            self._design(
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
                        "selector": "A1-2",
                        "coupling_group": "complete_interface_seed",
                        "pose": pose,
                    },
                    {
                        "kind": "fixed_xyz",
                        "selector": "B1-2",
                        "coupling_group": "complete_interface_seed",
                        "pose": pose,
                    },
                ],
            )
        )

        specification = lowered.specification
        self.assertEqual(len(specification.motion_groups), 1)
        group = specification.motion_groups["fixed_component_001"]
        self.assertEqual(set(group.members), {"motif_001", "motif_002"})
        orbit = specification.symmetry.orbits["motif_orbit"]
        self.assertEqual(orbit.master_groups, ["fixed_component_001"])
        self.assertEqual(set(orbit.component_mobility), {"fixed_component_001"})

    def test_joint_component_rejects_conflicting_pose_modes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same pose settings"):
            lower_user_design(
                self._design(
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
                            "selector": "A1-2",
                            "coupling_group": "joint_site",
                        },
                        {
                            "kind": "fixed_xyz",
                            "selector": "B1-2",
                            "coupling_group": "joint_site",
                            "pose": {
                                "mode": "bounded_mobile",
                                "max_translation": 3.0,
                                "max_rotation_deg": 10.0,
                            },
                        },
                    ],
                )
            )

    def test_lowers_declared_initial_pose_as_one_rigid_group(self) -> None:
        lowered = lower_user_design(
            self._design(
                generation=[
                    {
                        "kind": "terminal",
                        "anchor": "A1-2",
                        "terminus": "n",
                        "length": 20,
                    }
                ],
                constraints=[
                    {"kind": "fixed_xyz", "selector": "A1-2"},
                ],
                sampling={
                    "initial_pose": {
                        "radius": {"minimum": 20.0, "maximum": 30.0},
                        "orientation": {"method": "uniform_so3"},
                        "seed": 3000,
                    }
                },
            )
        )

        spec = lowered.specification
        self.assertEqual(spec.random_seed, 3000)
        self.assertEqual(set(spec.initialization), {"fixed_component_001"})
        self.assertEqual(
            spec.initialization["fixed_component_001"].orientation.method,
            "uniform_so3",
        )
        self.assertEqual(
            spec.initialization[
                "fixed_component_001"
            ].placement.radius.mean,
            25.0,
        )
        self.assertEqual(
            lowered.sampling_plan.diffusion.seed,
            42,
        )

    def test_initial_pose_is_relative_to_declared_symmetry_center(self) -> None:
        lowered = lower_user_design(
            self._design(
                symmetry={
                    "id": "C3",
                    "axis": [0.0, 0.0, 1.0],
                    "center": [10.0, 20.0, 30.0],
                },
                generation=[
                    {
                        "kind": "terminal",
                        "anchor": "A1-2",
                        "terminus": "n",
                        "length": 20,
                    }
                ],
                constraints=[
                    {"kind": "fixed_xyz", "selector": "A1-2"},
                ],
                sampling={
                    "initial_pose": {
                        "radius": {"minimum": 5.0, "maximum": 5.0},
                        "axial_offset": {
                            "minimum": 2.0,
                            "maximum": 2.0,
                        },
                        "orientation": {"method": "fixed"},
                    }
                },
            )
        )
        metadata: dict[str, object] = {}

        build_master_group_transforms(
            lowered.specification,
            base_directory=self.root,
            sample_metadata=metadata,
        )

        np.testing.assert_allclose(
            metadata["fixed_component_001"]["target_center"],
            [15.0, 20.0, 32.0],
            atol=1e-12,
        )

    def test_lowers_between_generation_with_orbit_relation(self) -> None:
        lowered = lower_user_design(
            self._design(
                symmetry={
                    "id": "D3",
                    "axis": [0.0, 0.0, 1.0],
                    "secondary_axis": [1.0, 0.0, 0.0],
                },
                generation=[
                    {
                        "kind": "between",
                        "from_selector": "A1-2",
                        "to_selector": "B1-2",
                        "length": {"minimum": 70, "maximum": 90},
                        "orbit_offset": 1,
                    }
                ],
                constraints=[
                    {
                        "kind": "fixed_xyz",
                        "selector": "A1-2,B1-2",
                    },
                ],
            )
        )

        spec = lowered.specification
        segment = spec.generated_segments["generated_001"]
        self.assertEqual(segment.copy_relation.orbit_offset, 1)
        transform = spec.symmetry.transform_sets["declared"]
        self.assertEqual(transform.type, "dihedral")
        self.assertEqual(transform.order, 3)

    def test_lowers_two_independent_mobile_components_for_d3(self) -> None:
        pose = {
            "mode": "bounded_mobile",
            "proposal": "scaffold_objectives",
            "max_translation": 3.0,
            "max_rotation_deg": 10.0,
        }
        lowered = lower_user_design(
            self._design(
                symmetry={
                    "id": "D3",
                    "axis": [0.0, 0.0, 1.0],
                    "secondary_axis": [1.0, 0.0, 0.0],
                },
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
                        "selector": "A1-2",
                        "coupling_group": "d3_orbit_alpha",
                        "pose": pose,
                    },
                    {
                        "kind": "fixed_xyz",
                        "selector": "B1-2",
                        "coupling_group": "d3_orbit_beta",
                        "pose": pose,
                    },
                ],
            )
        )

        spec = lowered.specification
        transform = spec.symmetry.transform_sets["declared"]
        orbit = spec.symmetry.orbits["motif_orbit"]
        self.assertEqual(transform.type, "dihedral")
        self.assertEqual(transform.order, 3)
        self.assertEqual(len(orbit.master_groups), 2)
        self.assertEqual(
            set(orbit.component_mobility),
            {"fixed_component_001", "fixed_component_002"},
        )

    def test_lowers_public_polyhedral_groups_into_assembly_ir(self) -> None:
        expected = {
            "T": ("tetrahedral", 12),
            "O": ("octahedral", 24),
            "I": ("icosahedral", 60),
        }
        for symmetry_id, (symmetry_type, order) in expected.items():
            with self.subTest(symmetry=symmetry_id):
                lowered = lower_user_design(
                    self._design(
                        symmetry=symmetry_id,
                        generation=[
                            {
                                "kind": "terminal",
                                "anchor": "A1-2",
                                "terminus": "n",
                                "length": 20,
                            }
                        ],
                        constraints=[
                            {
                                "kind": "fixed_xyz",
                                "selector": "A1-2",
                            }
                        ],
                    )
                )
                transform = lowered.specification.symmetry.transform_sets[
                    "declared"
                ]
                self.assertEqual(transform.type, symmetry_type)
                self.assertEqual(transform.order, order)
                instances = expand_symmetry_instances(
                    lowered.specification
                )
                self.assertEqual(len(instances.fragments), order)
                self.assertEqual(len(instances.generated_segments), order)

    def test_component_finite_actions_lower_as_independent_orbits(
        self,
    ) -> None:
        plan = next(
            item
            for item in enumerate_binary_interface_incidence_plans(
                symmetry="T",
                interface_id="natural_interface",
                left_participant="alpha",
                right_participant="beta",
            )
            if (item.left.valency, item.right.valency) == (2, 3)
        )

        def action_payload(participant_plan):
            action = participant_plan.action
            return {
                "coset_representative_ids": (
                    action.coset_representative_ids
                ),
                "stabilizer_transform_ids": (
                    action.stabilizer_transform_ids
                ),
                "transform_to_coset_representative": dict(
                    action.transform_to_coset_representative
                ),
            }

        registry = build_transform_registry(finite_symmetry_spec("T"))
        lines: list[str] = []
        serial = 1
        component_chains = (
            ("alpha", ("A", "B"), plan.left),
            ("beta", ("C", "D", "E"), plan.right),
        )
        for component_index, (_, chains, participant_plan) in enumerate(
            component_chains
        ):
            for chain, transform_id in zip(
                chains,
                participant_plan.action.stabilizer_transform_ids,
                strict=True,
            ):
                transform = registry.transform(transform_id)
                for residue in range(1, 5):
                    for atom_index, atom_name in enumerate(
                        ("N", "CA", "C", "O")
                    ):
                        coordinate = np.asarray(
                            (
                                18.0 + component_index * 7.0 + residue * 1.2,
                                5.0 + component_index * 11.0 + atom_index * 0.2,
                                3.0 + residue * 1.4 + atom_index * 0.1,
                            )
                        )
                        coordinate = (
                            coordinate @ transform[:3, :3].T
                            + transform[:3, 3]
                        )
                        lines.append(
                            f"ATOM  {serial:5d} {atom_name:>4s} ALA "
                            f"{chain:1s}{residue:4d}    "
                            f"{coordinate[0]:8.3f}{coordinate[1]:8.3f}"
                            f"{coordinate[2]:8.3f}{1.0:6.2f}{20.0:6.2f}"
                            f"          {atom_name[0]:>2s}\n"
                        )
                        serial += 1
        lines.append("END\n")
        self.structure.write_text("".join(lines), encoding="utf-8")

        component_selectors = {
            component_id: [
                selector
                for chain in chains
                for selector in (f"{chain}1-2", f"{chain}3-4")
            ]
            for component_id, chains, _ in component_chains
        }
        connections = []
        for component_id, chains, _ in component_chains:
            for chain in chains:
                connections.append({
                    "id": f"{component_id}_path_{chain}",
                    "from": {
                        "component": component_id,
                        "selector": f"{chain}1-2",
                        "terminus": "c",
                    },
                    "to": {
                        "component": component_id,
                        "selector": f"{chain}3-4",
                        "terminus": "n",
                    },
                    "length": 20,
                })

        design = self._design(
            symmetry="T",
            components={
                "alpha": {
                    "selectors": component_selectors["alpha"],
                    "geometry": "joint_rigid",
                    "finite_orbit_action": action_payload(plan.left),
                },
                "beta": {
                    "selectors": component_selectors["beta"],
                    "geometry": "joint_rigid",
                    "finite_orbit_action": action_payload(plan.right),
                },
            },
            ports={
                "alpha_face": {
                    "component": "alpha",
                    "selectors": component_selectors["alpha"],
                },
                "beta_face": {
                    "component": "beta",
                    "selectors": component_selectors["beta"],
                },
            },
            interfaces=[{
                "id": "natural_interface",
                "between": ["alpha_face", "beta_face"],
                "relation": {"mode": "preserve_input"},
                "use": {"exact": 12},
            }],
            connections=connections,
        )

        lowered = lower_user_design(design)
        self.assertEqual(
            lowered.specification.constraint_group_strategy,
            "interface_edges",
        )
        orbits = lowered.specification.symmetry.orbits
        self.assertEqual(
            set(orbits),
            {"motif_orbit__alpha", "motif_orbit__beta"},
        )
        instances = expand_symmetry_instances(lowered.specification)
        self.assertEqual(len(instances.motion_groups), 10)
        self.assertEqual(len(instances.interfaces), 12)
        self.assertEqual(
            lowered.interface_usage[0].physical_instance_count,
            12,
        )

    def test_tetrahedral_joint_seed_accepts_one_generic_initial_pose(
        self,
    ) -> None:
        lowered = lower_user_design(
            self._design(
                symmetry={
                    "id": "T",
                    "axis": [0.0, 0.0, 1.0],
                    "secondary_axis": [1.0, 0.0, 0.0],
                    "center": [0.0, 0.0, 0.0],
                },
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
                        "selector": "A1-2",
                        "coupling_group": "joint_seed",
                    },
                    {
                        "kind": "fixed_xyz",
                        "selector": "B1-2",
                        "coupling_group": "joint_seed",
                    },
                ],
                sampling={
                    "initial_pose": {
                        "radius": {"minimum": 80.0, "maximum": 80.0},
                        "orientation": {"method": "uniform_so3"},
                        "seed": 920,
                    }
                },
            )
        )

        spec = lowered.specification
        self.assertEqual(
            list(spec.motion_groups),
            ["fixed_component_001"],
        )
        self.assertEqual(
            list(spec.motion_groups["fixed_component_001"].members),
            ["motif_001", "motif_002"],
        )
        self.assertEqual(
            spec.initialization[
                "fixed_component_001"
            ].placement.radius.mean,
            80.0,
        )
        instances = expand_symmetry_instances(spec)
        self.assertEqual(len(instances.fragments), 24)
        self.assertEqual(len(instances.generated_segments), 12)

    def test_tetrahedral_independent_orbits_accept_component_poses(
        self,
    ) -> None:
        lowered = lower_user_design(
            self._design(
                symmetry={
                    "id": "T",
                    "axis": [0.0, 0.0, 1.0],
                    "secondary_axis": [1.0, 0.0, 0.0],
                    "center": [0.0, 0.0, 0.0],
                },
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
                        "selector": "A1-2",
                        "coupling_group": "site_alpha",
                    },
                    {
                        "kind": "fixed_xyz",
                        "selector": "B1-2",
                        "coupling_group": "site_beta",
                    },
                ],
                sampling={
                    "initial_poses": {
                        "site_alpha": {
                            "radius": {
                                "minimum": 70.0,
                                "maximum": 70.0,
                            },
                            "orientation": {"method": "uniform_so3"},
                            "seed": 101,
                        },
                        "site_beta": {
                            "radius": {
                                "minimum": 90.0,
                                "maximum": 90.0,
                            },
                            "radial_direction": [0.0, 1.0, 0.0],
                            "orientation": {"method": "uniform_so3"},
                            "seed": 202,
                        },
                    }
                },
            )
        )

        spec = lowered.specification
        self.assertIsNone(spec.random_seed)
        self.assertEqual(
            list(spec.initialization),
            ["fixed_component_001", "fixed_component_002"],
        )
        self.assertEqual(
            spec.initialization["fixed_component_001"].random_seed,
            101,
        )
        self.assertEqual(
            spec.initialization["fixed_component_002"].random_seed,
            202,
        )
        self.assertEqual(
            spec.initialization[
                "fixed_component_001"
            ].placement.radius.mean,
            70.0,
        )
        self.assertEqual(
            spec.initialization[
                "fixed_component_002"
            ].placement.radius.mean,
            90.0,
        )

        metadata: dict[str, object] = {}
        transforms = build_master_group_transforms(
            spec,
            base_directory=self.root,
            sample_metadata=metadata,
        )
        self.assertEqual(
            metadata["fixed_component_001"]["random_seed"],
            101,
        )
        self.assertEqual(
            metadata["fixed_component_002"]["random_seed"],
            202,
        )

        reversed_payload = spec.model_dump(mode="json")
        reversed_payload["initialization"] = dict(
            reversed(list(reversed_payload["initialization"].items()))
        )
        reversed_spec = AssemblySpecification.model_validate(
            reversed_payload
        )
        reversed_transforms = build_master_group_transforms(
            reversed_spec,
            base_directory=self.root,
        )
        for group_id in transforms:
            np.testing.assert_allclose(
                transforms[group_id],
                reversed_transforms[group_id],
                atol=1e-12,
            )

        resampled_metadata: dict[str, object] = {}
        resampled = build_master_group_transforms(
            spec,
            base_directory=self.root,
            random_seed=303,
            sample_metadata=resampled_metadata,
        )
        repeated_resample = build_master_group_transforms(
            spec,
            base_directory=self.root,
            random_seed=303,
        )
        self.assertNotEqual(
            resampled_metadata["fixed_component_001"]["random_seed"],
            101,
        )
        self.assertNotEqual(
            resampled_metadata["fixed_component_002"]["random_seed"],
            202,
        )
        for group_id in resampled:
            np.testing.assert_allclose(
                resampled[group_id],
                repeated_resample[group_id],
                atol=1e-12,
            )

    def test_component_initial_pose_rejects_unknown_coupling_group(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "unknown fixed coupling_group"):
            lower_user_design(
                self._design(
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
                            "selector": "A1-2",
                            "coupling_group": "site_alpha",
                        },
                        {
                            "kind": "fixed_xyz",
                            "selector": "B1-2",
                            "coupling_group": "site_beta",
                        },
                    ],
                    sampling={
                        "initial_poses": {
                            "missing_site": {
                                "radius": {
                                    "minimum": 50.0,
                                    "maximum": 50.0,
                                }
                            }
                        }
                    },
                )
            )

    def test_polyhedral_lowering_rejects_cyclic_orbit_offsets(self) -> None:
        design = self._design(
            symmetry="T",
            generation=[
                {
                    "kind": "between",
                    "from_selector": "A1-2",
                    "to_selector": "B1-2",
                    "length": 30,
                    "orbit_offset": 1,
                }
            ],
            constraints=[
                {
                    "kind": "fixed_xyz",
                    "selector": "A1-2,B1-2",
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "cyclic orbit offsets"):
            lower_user_design(design)

    def test_lowering_emits_runtime_cylindrical_atom_contract(self) -> None:
        declared = self._design(
            generation=[
                {
                    "kind": "terminal",
                    "anchor": "A1-2",
                    "terminus": "n",
                    "length": 20,
                }
            ],
            constraints=[
                {
                    "kind": "cylindrical",
                    "selector": "A1-2",
                    "keep": ["radius"],
                }
            ],
        )

        lowered = lower_user_design(declared)

        self.assertEqual(
            lowered.specification.fragments["motif_001"].fixed_atoms,
            "none",
        )
        runtime = lowered.runtime_constraint_metadata[
            "cylindrical_constraints"
        ]
        self.assertEqual(len(runtime), 1)
        self.assertEqual(runtime[0]["keep"], ["radius"])
        self.assertEqual(runtime[0]["axis"], [0.0, 0.0, 1.0])
        self.assertEqual(
            {
                atom["atom_name"]
                for atom in runtime[0]["members"][0]["source_atoms"]
            },
            {"CA"},
        )

    def test_lowering_rejects_implicit_endpoint_fixing(self) -> None:
        declared = self._design(
            generation=[
                {
                    "kind": "terminal",
                    "anchor": "A1-2",
                    "terminus": "n",
                    "length": 20,
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "explicit fixed_xyz"):
            lower_user_design(declared)


if __name__ == "__main__":
    unittest.main()
