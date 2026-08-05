import tempfile
import unittest
from pathlib import Path

import numpy as np

from rfd3_mosaic.compile import build_master_group_transforms
from rfd3_mosaic.constraint_plan import compile_constraint_plan
from rfd3_mosaic.design_compiler import (
    bind_constraint_plan,
    lower_user_design,
    parse_public_selector,
)
from rfd3_mosaic.schema import UserDesignSpec


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
        self.assertEqual(spec.fragments["motif_001"].fixed_atoms, "all")
        self.assertEqual(
            spec.symmetry.transform_sets["declared"].order,
            3,
        )

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
        self.assertEqual(mobility.bounds.max_translation, 3.0)
        self.assertEqual(mobility.bounds.max_rotation_deg, 10.0)
        self.assertEqual(
            orbit.component_mobility.get("fixed_component_002"), None
        )

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

    def test_lowering_rejects_unimplemented_cylindrical_backend(self) -> None:
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

        with self.assertRaisesRegex(ValueError, "cylindrical"):
            lower_user_design(declared)

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
